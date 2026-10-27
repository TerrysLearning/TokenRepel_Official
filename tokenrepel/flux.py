"""FLUX.1-dev with TokenRepel.

Every attention layer pushes its image tokens (before the output projection). With preservation, two extra
branches (clean prompt, negative prompt) are run next to the repel branch.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from .repel import TokenRepel


class FluxRepelAttnProcessor:
    """FLUX attention (same math as diffusers' FluxAttnProcessor) + TokenRepel on the image tokens."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None, repel=None, step=None, n_txt=None):
        B = hidden_states.shape[0]
        heads = attn.heads

        def split_heads(x):
            return x.view(B, -1, heads, x.shape[-1] // heads).transpose(1, 2)

        query = split_heads(attn.to_q(hidden_states))
        key   = split_heads(attn.to_k(hidden_states))
        value = split_heads(attn.to_v(hidden_states))
        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key   = attn.norm_k(key)

        if encoder_hidden_states is not None:              # double-stream: prepend the text stream
            n_txt = encoder_hidden_states.shape[1]
            eq = split_heads(attn.add_q_proj(encoder_hidden_states))
            ek = split_heads(attn.add_k_proj(encoder_hidden_states))
            ev = split_heads(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], dim=2)
            key   = torch.cat([ek, key],   dim=2)
            value = torch.cat([ev, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, heads * query.shape[-1]).to(query.dtype)

        if repel is not None and repel.is_active(step):    # output is [text | image]
            hidden_states = torch.cat([hidden_states[:, :n_txt], repel.push(hidden_states[:, n_txt:])], dim=1)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states[:, :n_txt], hidden_states[:, n_txt:]
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


class FluxTokenRepelPipeline(FluxPipeline):

    @torch.no_grad()
    def __call__(self, prompt: str, latents: torch.Tensor, height=512, width=512,
                 num_inference_steps=25, guidance_scale=3.5, repel: Optional[TokenRepel] = None):
        """latents: (B, N, 64) packed initial noise; all B images share `prompt` and repel each other."""
        if not isinstance(self.transformer.transformer_blocks[0].attn.processor, FluxRepelAttnProcessor):
            self.transformer.set_attn_processor(FluxRepelAttnProcessor())
        device = self._execution_device
        B = latents.shape[0]

        prompt_embeds, pooled, txt_ids = self.encode_prompt(prompt=[prompt] * B, prompt_2=None, device=device)
        preserved = repel is not None and repel.preservation != 'none'
        if preserved:
            neg_embeds, neg_pooled, _ = self.encode_prompt(prompt=repel.neg_prompt, prompt_2=None, device=device)
            neg_embeds, neg_pooled = neg_embeds.expand(B, -1, -1), neg_pooled.expand(B, -1)

        latents, img_ids = self.prepare_latents(B, self.transformer.config.in_channels // 4, height, width,
                                                prompt_embeds.dtype, device, None, latents)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(latents.shape[1], self.scheduler.config.base_image_seq_len,
                             self.scheduler.config.max_image_seq_len,
                             self.scheduler.config.base_shift, self.scheduler.config.max_shift)
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        guidance = torch.full([B], guidance_scale, device=device, dtype=torch.float32)

        for i, t in enumerate(timesteps):
            timestep = t.expand(B).to(latents.dtype) / 1000
            jkw = {'repel': repel, 'step': i, 'n_txt': prompt_embeds.shape[1]}
            if preserved and repel.is_active(i):
                v = self._forward_branches(latents, timestep, guidance, prompt_embeds, pooled,
                                           neg_embeds, neg_pooled, txt_ids, img_ids, repel, jkw)
            else:
                v = self.transformer(hidden_states=latents, timestep=timestep, guidance=guidance,
                                     pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                                     txt_ids=txt_ids, img_ids=img_ids, return_dict=False,
                                     joint_attention_kwargs=jkw)[0]
            latents = self.scheduler.step(v, t, latents, return_dict=False)[0]

        x = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        x = x / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.image_processor.postprocess(self.vae.decode(x, return_dict=False)[0])

    def _forward_branches(self, latents, timestep, guidance, prompt_embeds, pooled, neg_embeds, neg_pooled,
                          txt_ids, img_ids, repel, jkw):
        """Repel / clean / negative-prompt branches block by block; returns the preserved velocity.
        'layer'    -> after each preserving block, repel <- clean + correct(repel - clean, clean - neg);
                      once no preserving block is ahead, only the repel branch is run.
        'velocity' -> v_clean + correct(v_repel - v_clean, v_clean - v_neg)."""
        tr = self.transformer
        x = tr.x_embedder(latents)
        t = timestep.to(x.dtype) * 1000
        g = guidance.to(x.dtype) * 1000
        temb, temb_neg = tr.time_text_embed(t, g, pooled), tr.time_text_embed(t, g, neg_pooled)
        rope = tr.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

        rep = clean = (tr.context_embedder(prompt_embeds), x)    # (text stream, image stream)
        neg = (tr.context_embedder(neg_embeds), x)
        velocity = repel.preservation == 'velocity'
        for b, block in enumerate([*tr.transformer_blocks, *tr.single_transformer_blocks]):
            rep = block(rep[1], rep[0], temb, rope, jkw)
            if velocity or repel.preserves(b):
                clean = block(clean[1], clean[0], temb, rope)
                neg   = block(neg[1], neg[0], temb_neg, rope)
            if repel.preserves(b):
                rep = (rep[0], clean[1] + repel.correct(rep[1] - clean[1], clean[1] - neg[1]))

        def head(h, tb):
            return tr.proj_out(tr.norm_out(h, tb))

        if not velocity:
            return head(rep[1], temb)
        v_r, v, v_neg = head(rep[1], temb), head(clean[1], temb), head(neg[1], temb_neg)
        return v + repel.correct(v_r - v, v - v_neg)

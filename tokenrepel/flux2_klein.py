"""FLUX.2-klein with TokenRepel.

The repel processors wrap the stock FLUX.2 attention processors and push the image tokens of their output.
Prompt encoding, latent preparation and decoding are inherited from diffusers.
"""
import numpy as np
import torch
from typing import Optional

from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline, compute_empirical_mu, retrieve_timesteps,
)

from .repel import TokenRepel


class Flux2RepelAttnProcessor:
    """Wraps a stock FLUX.2 processor (whose positional signature differs between double- and
    single-stream blocks) and pushes the image tokens of its output."""

    def __init__(self, base, is_single: bool):
        self.base = base
        self.is_single = is_single

    def __call__(self, attn, hidden_states, *args, repel: Optional[TokenRepel] = None, step=None, n_txt=None):
        out = self.base(attn, hidden_states, *args)
        if repel is not None and repel.is_active(step):
            if self.is_single:      # output is [text | image]
                out = torch.cat([out[:, :n_txt], repel.push(out[:, n_txt:])], dim=1)
            else:                   # output is (image, text)
                out = (repel.push(out[0]), out[1])
        return out


class Flux2KleinTokenRepelPipeline(Flux2KleinPipeline):

    def _set_repel_processors(self):
        tr = self.transformer
        if isinstance(tr.transformer_blocks[0].attn.processor, Flux2RepelAttnProcessor):
            return
        for block in tr.transformer_blocks:
            block.attn.processor = Flux2RepelAttnProcessor(block.attn.processor, False)
        for block in tr.single_transformer_blocks:
            block.attn.processor = Flux2RepelAttnProcessor(block.attn.processor, True)

    @torch.no_grad()
    def __call__(self, prompt: str, latents: torch.Tensor, height=512, width=512,
                 num_inference_steps=4, repel: Optional[TokenRepel] = None):
        """latents: (B, N, 128) packed initial noise; all B images share `prompt` and repel each other."""
        self._set_repel_processors()
        device = self._execution_device
        B = latents.shape[0]

        prompt_embeds, txt_ids = self.encode_prompt(prompt=[prompt] * B, device=device)
        preserved = repel is not None and repel.preservation != 'none'
        if preserved:
            neg_embeds = self.encode_prompt(prompt=repel.neg_prompt, device=device)[0].expand(B, -1, -1)

        latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        h, w = 2 * (height // (self.vae_scale_factor * 2)), 2 * (width // (self.vae_scale_factor * 2))
        img_ids = self._prepare_latent_ids(torch.zeros(B, latents.shape[-1], h // 2, w // 2)).to(device)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        self.scheduler.set_begin_index(0)

        def velocity(timestep, embeds, jkw=None):
            return self.transformer(hidden_states=latents, timestep=timestep / 1000, guidance=None,
                                    encoder_hidden_states=embeds, txt_ids=txt_ids, img_ids=img_ids,
                                    joint_attention_kwargs=jkw, return_dict=False)[0][:, :latents.shape[1]]

        for i, t in enumerate(timesteps):
            timestep = t.expand(B).to(latents.dtype)
            jkw = {'repel': repel, 'step': i, 'n_txt': prompt_embeds.shape[1]}
            if preserved and repel.is_active(i) and repel.preservation == 'layer':
                v = self._forward_layer_preserved(latents, timestep, prompt_embeds, neg_embeds, txt_ids, img_ids, repel, jkw)
            else:
                v = velocity(timestep, prompt_embeds, jkw)
                if preserved and repel.is_active(i):            # preservation == 'velocity'
                    v_clean, v_neg = velocity(timestep, prompt_embeds), velocity(timestep, neg_embeds)
                    v = v_clean + repel.correct(v - v_clean, v_clean - v_neg)
            latents = self.scheduler.step(v, t, latents, return_dict=False)[0]

        x = self._unpack_latents_with_ids(latents, img_ids, h // 2, w // 2)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
        bn_std  = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(x.device, x.dtype)
        x = self._unpatchify_latents(x * bn_std + bn_mean)
        return self.image_processor.postprocess(self.vae.decode(x, return_dict=False)[0])

    def _forward_layer_preserved(self, latents, timestep, prompt_embeds, neg_embeds, txt_ids, img_ids, repel, jkw):
        """Repel / clean / negative-prompt branches side by side; after each preserving block the repel branch
        is replaced by clean + correct(repel - clean, clean - neg); once no preserving block is ahead, only the
        repel branch is run. Returns the velocity."""
        tr = self.transformer
        n_txt = prompt_embeds.shape[1]
        temb = tr.time_guidance_embed(timestep.to(tr.dtype), None)
        mod_img = tr.double_stream_modulation_img(temb)
        mod_txt = tr.double_stream_modulation_txt(temb)
        mod_single = tr.single_stream_modulation(temb)
        if img_ids.ndim == 3: img_ids = img_ids[0]
        if txt_ids.ndim == 3: txt_ids = txt_ids[0]
        rope_img, rope_txt = tr.pos_embed(img_ids), tr.pos_embed(txt_ids)
        rope = (torch.cat([rope_txt[0], rope_img[0]], dim=0), torch.cat([rope_txt[1], rope_img[1]], dim=0))

        def double(block, branch, kw=None):
            return block(hidden_states=branch[1], encoder_hidden_states=branch[0], temb_mod_img=mod_img,
                         temb_mod_txt=mod_txt, image_rotary_emb=rope, joint_attention_kwargs=kw)

        def single(block, hs, kw=None):
            return block(hidden_states=hs, encoder_hidden_states=None, temb_mod=mod_single,
                         image_rotary_emb=rope, joint_attention_kwargs=kw)

        x = tr.x_embedder(latents.to(tr.dtype))
        rep = clean = (tr.context_embedder(prompt_embeds.to(tr.dtype)), x)    # (text stream, image stream)
        neg = (tr.context_embedder(neg_embeds.to(tr.dtype)), x)
        for b, block in enumerate(tr.transformer_blocks):
            rep = double(block, rep, jkw)
            if repel.preserves(b):
                clean, neg = double(block, clean), double(block, neg)
                rep = (rep[0], clean[1] + repel.correct(rep[1] - clean[1], clean[1] - neg[1]))

        rep, clean, neg = (torch.cat(s, dim=1) for s in (rep, clean, neg))   # single-stream: [text | image]
        for b, block in enumerate(tr.single_transformer_blocks, start=len(tr.transformer_blocks)):
            rep = single(block, rep, jkw)
            if repel.preserves(b):
                clean, neg = single(block, clean), single(block, neg)
                rep = clean + repel.correct(rep - clean, clean - neg)
        return tr.proj_out(tr.norm_out(rep[:, n_txt:], temb))

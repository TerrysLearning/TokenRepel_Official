"""SDXL with TokenRepel.

Every UNet self-attention layer pushes its output tokens, in the conditional CFG branch only. The high-density
direction is the CFG direction eps(x, c) - eps(x, uncond).
"""
import torch
import torch.nn.functional as F
from typing import Optional

from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps

from .repel import TokenRepel


class SDXLRepelAttnProcessor:
    """Standard SDPA attention; self-attention layers push the last `num_images` rows of the batch
    (the conditional images) apart after the output projection."""

    def __init__(self, is_self: bool):
        self.is_self = is_self

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 repel: Optional[TokenRepel] = None, step=None, num_images=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, c, hh * ww).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads
        query, key, value = (x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for x in (query, key, value))
        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key   = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                       dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, hh, ww)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        if self.is_self and repel is not None and repel.is_active(step):
            hidden_states = torch.cat([hidden_states[:-num_images], repel.push(hidden_states[-num_images:])], dim=0)
        return hidden_states


class SDXLTokenRepelPipeline(StableDiffusionXLPipeline):

    @torch.no_grad()
    def __call__(self, prompt: str, latents: torch.Tensor, height=512, width=512, num_inference_steps=50,
                 guidance_scale=7.5, repel: Optional[TokenRepel] = None):
        """latents: (B, 4, H/8, W/8) initial noise; all B images share `prompt` and repel each other."""
        if not any(isinstance(p, SDXLRepelAttnProcessor) for p in self.unet.attn_processors.values()):
            self.unet.set_attn_processor({name: SDXLRepelAttnProcessor('attn1' in name) for name in self.unet.attn_processors})
        device = self._execution_device
        B = latents.shape[0]

        pe, npe, pooled, npooled = self.encode_prompt(
            prompt=[prompt] * B, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=[''] * B)
        time_ids = self._get_add_time_ids((height, width), (0, 0), (height, width), dtype=pe.dtype,
                                          text_encoder_projection_dim=self.text_encoder_2.config.projection_dim)
        prompt_embeds = torch.cat([npe, pe]).to(device)                       # [uncond | cond]
        added = {'text_embeds': torch.cat([npooled, pooled]).to(device),
                 'time_ids': time_ids.to(device).repeat(2 * B, 1)}

        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        latents = self.prepare_latents(B, self.unet.config.in_channels, height, width, pe.dtype, device, None, latents)
        extra = self.prepare_extra_step_kwargs(None, 0.0)
        channels_last = lambda x: x.permute(0, 2, 3, 1)                       # one token per latent pixel

        for i, t in enumerate(timesteps):
            x_in = self.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            cak = {'repel': repel, 'step': i, 'num_images': B}
            active = repel is not None and repel.is_active(i)
            if active and repel.preservation == 'layer':
                eps = self._unet_layer_preserved(x_in, t, prompt_embeds, added, cak, repel)
            else:
                eps = self.unet(x_in, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cak,
                                added_cond_kwargs=added, return_dict=False)[0]
            eps_u, eps_c = eps.chunk(2)
            if active and repel.preservation == 'velocity':
                eps_clean = self.unet(x_in[B:], t, encoder_hidden_states=prompt_embeds[B:],
                                      added_cond_kwargs={k: v[B:] for k, v in added.items()}, return_dict=False)[0]
                d, h = channels_last(eps_c - eps_clean), channels_last(eps_clean - eps_u)
                eps_c = eps_clean + repel.correct(d, h).permute(0, 3, 1, 2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            latents = self.scheduler.step(eps, t, latents, **extra, return_dict=False)[0]

        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:   # the fp16 SDXL VAE overflows
            self.vae.to(torch.float32)
        image = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        return self.image_processor.postprocess(image)

    def _unet_layer_preserved(self, x_in, t, prompt_embeds, added, cak, repel):
        """Run the 7 top-level UNet blocks (3 down, mid, 3 up) on three branches: repel (cond, pushed), clean (cond)
        and uncond. After each preserving block, repel <- clean + correct(repel - clean, clean - uncond) on the
        (B, C, H, W) feature map with channels as the token dimension; once no preserving block is ahead, the
        clean branch is skipped (uncond always runs, it is the uncond half of the CFG output)."""
        unet = self.unet
        B = x_in.shape[0] // 2
        ehs = {'u': prompt_embeds[:B], 'c': prompt_embeds[B:]}
        add = {'u': {k: v[:B] for k, v in added.items()}, 'c': {k: v[B:] for k, v in added.items()}}
        emb = unet.time_embedding(unet.get_time_embed(sample=x_in[B:], timestep=t), None)
        emb = {k: emb + unet.get_aug_embed(emb=emb, encoder_hidden_states=ehs[k], added_cond_kwargs=add[k]) for k in 'uc'}
        cond = {'r': 'c', 'c': 'c', 'u': 'u'}                                # branch -> its conditioning
        ck = {'r': cak, 'c': None, 'u': None}

        s = {'u': unet.conv_in(x_in[:B])}
        s['r'] = s['c'] = unet.conv_in(x_in[B:])
        res = {k: (v,) for k, v in s.items()}                                # skip connections per branch
        blocks = [('down', blk) for blk in unet.down_blocks] + [('mid', unet.mid_block)] + [('up', blk) for blk in unet.up_blocks]
        for b, (part, blk) in enumerate(blocks):
            for k in ('r', 'c', 'u') if repel.preserves(b) else ('r', 'u'):
                kw = {'temb': emb[cond[k]]}
                if getattr(blk, 'has_cross_attention', False):
                    kw.update(encoder_hidden_states=ehs[cond[k]], cross_attention_kwargs=ck[k])
                if part == 'down':
                    s[k], r = blk(s[k], **kw)
                    res[k] += r
                elif part == 'up':
                    n = len(blk.resnets)
                    s[k] = blk(s[k], res_hidden_states_tuple=res[k][-n:], **kw)
                    res[k] = res[k][:-n]
                else:
                    s[k] = blk(s[k], **kw)
            if repel.preserves(b):
                d, h = (x.permute(0, 2, 3, 1) for x in (s['r'] - s['c'], s['c'] - s['u']))
                s['r'] = s['c'] + repel.correct(d, h).permute(0, 3, 1, 2)

        post = lambda x: unet.conv_out(unet.conv_act(unet.conv_norm_out(x)))
        return torch.cat([post(s['u']), post(s['r'])], dim=0)

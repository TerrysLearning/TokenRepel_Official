import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


NEG_PROMPT = (
    "flat background, simple environment, cartoon, anime, low detail, disfigured, disgusting, "
    "blurry, low quality, unrealistic, distorted, deformed, bad anatomy, ugly, watermark, oversaturated, "
    "noise, overexposed, underexposed, compressed artifacts, plastic skin, airbrushed, "
    "doll-like, fake, low resolution, illustration"
)


@dataclass
class TokenRepel:
    """TokenRepel settings and its two operations.

    push(h)        token repulsion: rotate every image token by `delta` degrees away from its nearest token
                   in the *other* images of the batch, if the angle between them is below `theta_bar` degrees.
    correct(d, h)  token density preservation: project every displacement d_j caused by the repulsion that
                   opposes the high-density direction h_j onto the half-space {x : <x, h_j> >= 0}.

    preservation: 'none'     -> TokenRepel w/o P (push only)
                  'velocity' -> w/ P(V): correct the final velocity / noise prediction
                  'layer'    -> w/ P(L): correct the output of the first `preserve_blocks` transformer / UNet blocks
    """
    delta: float = 20.0             # push angle in degrees (< 90)
    theta_bar: float = 45.0         # only token pairs closer than this angle (degrees) are pushed
    step_start: int = 0             # repel on denoising steps [step_start, step_end)
    step_end: int = 3
    preservation: str = 'layer'     # 'none' | 'velocity' | 'layer'
    preserve_blocks: int = -1       # 'layer': correct after the first N blocks only (-1 = all)
    neg_prompt: str = NEG_PROMPT    # c_neg of the high-density direction (FLUX; SDXL uses its unconditional branch)
    strength: float = field(init=False)     # tan(delta)
    threshold: float = field(init=False)    # cos(theta_bar)

    def __post_init__(self):
        assert self.preservation in ('none', 'velocity', 'layer'), self.preservation
        assert 0 <= self.delta < 90 and 0 <= self.theta_bar <= 180, (self.delta, self.theta_bar)
        self.strength = math.tan(math.radians(self.delta))
        self.threshold = math.cos(math.radians(self.theta_bar))

    def is_active(self, step: int) -> bool:
        return self.step_start <= step < self.step_end

    def preserves(self, block: int) -> bool:
        """Whether the output of block `block` (counted over all blocks of the network) is corrected."""
        return self.preservation == 'layer' and (self.preserve_blocks < 0 or block < self.preserve_blocks)

    def push(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) image tokens of the B images in the batch. Returns the pushed tokens."""
        B, N, D = h.shape
        h_flat = h.reshape(B * N, D)
        h_hat  = F.normalize(h_flat, dim=-1)
        ids    = torch.arange(B, device=h.device).repeat_interleave(N)

        sim = h_hat @ h_hat.T
        sim.masked_fill_(ids.unsqueeze(1) == ids.unsqueeze(0), float('-inf'))   # exclude own image
        max_sim, argmax = sim.max(dim=-1)                                       # nearest other-image token
        mask = (max_sim > self.threshold).unsqueeze(-1)

        h_ref = F.normalize(h_flat[argmax], dim=-1)
        c     = (h_hat * h_ref).sum(dim=-1, keepdim=True)                       # cos(theta)
        t_dir = F.normalize(c * h_hat - h_ref, dim=-1)                          # unit tangent away from h_ref
        near  = (c > 1 - 1e-2).squeeze(-1)                                      # (near-)parallel: tangent undefined
        if near.any():
            # one random direction per token pair (i, nearest j), +r for the smaller index and -r for the other
            idx = near.nonzero().squeeze(-1)
            pair = torch.minimum(idx, argmax[idx])
            keys, inv = torch.unique(pair, return_inverse=True)
            r = F.normalize(torch.randn(len(keys), D, device=h.device, dtype=t_dir.dtype), dim=-1)[inv]
            r = torch.where(idx == pair, 1.0, -1.0).to(t_dir.dtype).unsqueeze(-1) * r
            h_i = h_hat[idx]
            t_dir[idx] = F.normalize(r - (r * h_i).sum(dim=-1, keepdim=True) * h_i, dim=-1)   # tangent at h_i
        h_new = h_hat + self.strength * t_dir
        overshoot = (h_new * h_ref).sum(dim=-1, keepdim=True) < 0               # never rotate past 90 degrees
        h_new  = torch.where(overshoot, h_hat - c * h_ref, h_new)
        pushed = F.normalize(h_new, dim=-1) * h_flat.norm(dim=-1, keepdim=True) # keep the original norm
        return torch.where(mask, pushed, h_flat).reshape(B, N, D)

    def correct(self, delta_r: torch.Tensor, delta_h: torch.Tensor) -> torch.Tensor:
        """delta_r: displacement caused by the repulsion; delta_h: high-density direction. Last dim = channels."""
        h_hat = F.normalize(delta_h, dim=-1)
        cos  = (F.normalize(delta_r, dim=-1) * h_hat).sum(dim=-1, keepdim=True)
        proj = (delta_r * h_hat).sum(dim=-1, keepdim=True) * h_hat
        return torch.where(cos < 0, delta_r - proj, delta_r)

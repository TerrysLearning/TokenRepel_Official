"""Backbones and their default (paper) settings."""
import torch

REPOS = {
    'flux':   ('black-forest-labs/FLUX.1-dev', torch.bfloat16),
    'flux2k': ('black-forest-labs/FLUX.2-klein-9B', torch.bfloat16),
    'sdxl':   ('RunDiffusion/Juggernaut-XI-v11', torch.float16),
}

# preserve_blocks counts all blocks: FLUX.1 19 double + 38 single, FLUX.2-klein 8 double + 24 single,
# SDXL 3 down + mid + 3 up (4 = down and middle blocks)
DEFAULTS = {
    'flux':   dict(inference_steps=25, guidance_scale=3.5, delta=20.0, theta_bar=45.0, step_end=3, preserve_blocks=30),
    'flux2k': dict(inference_steps=4, delta=40.0, theta_bar=60.0, step_end=1, preserve_blocks=20),
    'sdxl':   dict(inference_steps=50, guidance_scale=7.5, delta=10.0, theta_bar=45.0, step_end=5, preserve_blocks=4),
}


def load_pipeline(model: str, device='cuda'):
    if model == 'flux':
        from .flux import FluxTokenRepelPipeline as cls
    elif model == 'flux2k':
        from .flux2_klein import Flux2KleinTokenRepelPipeline as cls
    else:
        from .sdxl import SDXLTokenRepelPipeline as cls
    repo, dtype = REPOS[model]
    pipe = cls.from_pretrained(repo, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def init_noise(model: str, num_images: int, size: int, seed: int) -> torch.Tensor:
    """Initial noise on the CPU (packed latents for FLUX); the same seed gives the same noise with and without TokenRepel."""
    torch.manual_seed(seed)
    dtype = REPOS[model][1]
    if model == 'sdxl':
        return torch.randn(num_images, 4, size // 8, size // 8, dtype=dtype)
    return torch.randn(num_images, (size // 16) ** 2, 64 if model == 'flux' else 128, dtype=dtype)

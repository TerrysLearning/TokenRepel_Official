"""Generate a batch of images per prompt with TokenRepel (or the plain baseline).

    python generate.py --model flux2k --prompts assets/example_prompts.json
    python generate.py --model flux --prompts "A cat wearing a tiny crown." --baseline

Images go to <out_dir>/p{prompt_id:03d}/img{i:02d}.png, the settings to <out_dir>/config.json.
Unset options fall back to the paper settings of the model (tokenrepel/models.py).
"""
import argparse
import json
import os

import torch.multiprocessing as mp

from tokenrepel import DEFAULTS, TokenRepel, init_noise, load_pipeline


def worker(rank, args, prompts):
    pipe = load_pipeline(args.model, device=f'cuda:{rank}')
    repel = None if args.baseline else TokenRepel(
        delta=args.delta, theta_bar=args.theta_bar, step_start=args.step_start, step_end=args.step_end,
        preservation=args.preservation, preserve_blocks=args.preserve_blocks)
    kwargs = dict(height=args.resolution, width=args.resolution, num_inference_steps=args.inference_steps, repel=repel)
    if args.guidance_scale is not None:
        kwargs['guidance_scale'] = args.guidance_scale
    latents = init_noise(args.model, args.num_images_per_prompt, args.resolution, args.seed)

    for pid in range(rank, len(prompts), args.num_gpus):
        d = os.path.join(args.out_dir, f'p{pid:03d}')
        if os.path.isdir(d) and len(os.listdir(d)) >= args.num_images_per_prompt:
            continue                                            # already done
        images = []
        for i in range(0, len(latents), args.max_chunk):        # images in the same chunk repel each other
            images += pipe(prompts[pid], latents[i:i + args.max_chunk], **kwargs)
        os.makedirs(d, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(d, f'img{i:02d}.png'))
        print(f'[gpu {rank}] p{pid:03d} {prompts[pid]!r}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='flux2k', choices=list(DEFAULTS), help='flux = FLUX.1-dev, flux2k = FLUX.2-klein-9B, sdxl = Juggernaut-XI')
    p.add_argument('--prompts', default='assets/example_prompts.json', help='.json list of prompts, or a single prompt')
    p.add_argument('--out_dir', default=None, help='default: outputs/<model>[_baseline]')
    p.add_argument('--num_gpus', type=int, default=1, help='prompts are split over this many GPUs')
    p.add_argument('--num_images_per_prompt', type=int, default=4)
    p.add_argument('--max_chunk', type=int, default=8, help='max images per forward pass; only images in one chunk repel each other')
    p.add_argument('--seed', type=int, default=0, help='initial noise, the same for every prompt and with / without TokenRepel')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--inference_steps', type=int, default=None, help='default: flux 25, flux2k 4, sdxl 50')
    p.add_argument('--guidance_scale', type=float, default=None, help='default: flux 3.5, sdxl 7.5 (flux2k has none)')
    p.add_argument('--baseline', action='store_true', help='plain sampling without TokenRepel')
    p.add_argument('--delta', type=float, default=None, help='push angle in degrees; default: flux 20, flux2k 40, sdxl 10')
    p.add_argument('--theta_bar', type=float, default=None, help='only tokens closer than this angle (degrees) are pushed; default: flux 45, flux2k 60, sdxl 45')
    p.add_argument('--step_start', type=int, default=0, help='TokenRepel is applied on denoising steps [step_start, step_end)')
    p.add_argument('--step_end', type=int, default=None, help='default: flux 3, flux2k 1, sdxl 5')
    p.add_argument('--preservation', default='layer', choices=['none', 'velocity', 'layer'],
                   help='token density preservation: none = w/o P, velocity = w/ P(V), layer = w/ P(L) (paper)')
    p.add_argument('--preserve_blocks', type=int, default=None,
                   help='layer preservation after the first N blocks only (-1 = all); default: flux 30 of 57, flux2k 20 of 32, sdxl 4 of 7 (down + mid)')
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.out_dir = args.out_dir or f'outputs/{args.model}{"_baseline" if args.baseline else ""}'

    if args.prompts.endswith('.json'):
        with open(args.prompts) as f:
            prompts = json.load(f)
    else:
        prompts = [args.prompts]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    if args.num_gpus > 1:
        mp.spawn(worker, args=(args, prompts), nprocs=args.num_gpus, join=True)
    else:
        worker(0, args, prompts)


if __name__ == '__main__':
    main()

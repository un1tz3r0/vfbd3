"""
Bits shared by both engines: device/dtype selection and the final encode step.
"""

import gc

from . import video


def cuda_vram_gb():
	import torch
	if not torch.cuda.is_available():
		return 0.0
	return torch.cuda.get_device_properties(0).total_memory / 2**30


def pick_device():
	import torch
	return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device):
	''' bfloat16 where the GPU supports it natively, float16 on older cards such as the T4
	(where bfloat16 is emulated and slow), float32 on CPU. '''
	import torch
	if not str(device).startswith("cuda"):
		return torch.float32
	try:
		native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
	except TypeError:
		native_bf16 = torch.cuda.is_bf16_supported()
	return torch.bfloat16 if native_bf16 else torch.float16


def free_memory():
	import torch
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def apply_preset(cfg, presets, defaults):
	''' fill the config fields left at None from the `fast` preset, then from `defaults`.
	A preset's LoRA goes in front of the user's own. '''
	if cfg.fast and cfg.fast not in presets:
		raise ValueError(f"unknown fast preset {cfg.fast!r}; choices: {', '.join(sorted(presets))}")
	preset = presets.get(cfg.fast, {})
	for key, value in {**defaults, **preset}.items():
		if hasattr(cfg, key) and getattr(cfg, key) is None:
			setattr(cfg, key, value)
	cfg.lora = [*([preset["lora"]] if "lora" in preset else []), *cfg.lora]


def load_loras(pipe, specs):
	''' each spec is "repo_or_path[:weight_file][@scale]". '''
	names, scales = [], []
	for i, spec in enumerate(specs or []):
		spec, _, scale = spec.partition("@")
		repo, _, weight_name = spec.partition(":")
		name = f"lora{i}"
		pipe.load_lora_weights(repo, weight_name=weight_name or None, adapter_name=name)
		names.append(name)
		scales.append(float(scale) if scale else 1.0)
	if names:
		pipe.set_adapters(names, scales)


def finish(work, output, fps, audio_from=None, audio_start=None, upscale_to=None, upscale_model=None, crf=16):
	''' encode the work directory's frames to `output`, upscaling on the way if asked. '''
	frames = work.frames()
	if upscale_to:
		from .upscale import Upscaler
		upscaler = Upscaler(upscale_model) if upscale_model else Upscaler()
		frames = (upscaler(f, short_side=upscale_to) for f in frames)
	count = video.write_video(output, frames, fps, audio_from=audio_from, audio_start=audio_start, crf=crf)
	print(f"wrote {count} frames to {output}")
	return output

"""
Per-frame img2img with motion-compensated feedback, the technique from controlnetvideo.

Each frame: estimate the input's motion since the previous frame, carry the previous
*output* along that motion, blend it with the current input, and denoise the blend under a
control signal extracted from the input. The result is fed back for the next frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from . import common, control as controls, filters, motion, video

# control-type indices of the SDXL ControlNet-Union model
UNION_MODES = {"depth": 1, "canny": 3}


@dataclass
class FrameConfig:
	prompt: str
	negative_prompt: str = ""
	model: str = "stabilityai/stable-diffusion-xl-base-1.0"
	controlnet: str = "xinsir/controlnet-union-sdxl-1.0"
	control: str = "depth"         # depth or canny
	control_scale: float = 1.0
	short_side: int = 720
	max_area: int = 1280 * 720
	fps: float | None = None       # None keeps the input's frame rate
	steps: int = 25
	guidance: float = 7.5
	denoise: float = 0.5           # img2img strength: how much of each frame is re-generated
	first_denoise: float = 1.0     # same, for the first frame (nothing to feed back yet)
	feedthrough: float = 0.0       # share of the raw input in the init image; the rest is feedback
	motion_estimator: str = "dis"
	motion_alpha: float = 0.0      # temporal flow smoothing
	motion_sigma: float = 0.0      # spatial flow smoothing
	occlusion: bool = False        # fill areas the flow cannot explain from the input instead
	ip_adapter: float = 0.0        # > 0: also feed the warped previous output in as an image prompt
	color_fix: float = 0.0         # pull each output's colours toward the first output
	seed: int | None = None        # None: fresh noise every frame; int: the same noise every frame
	input_filter: str | None = None
	feedback_filter: str | None = None
	control_filter: str | None = None
	output_filter: str | None = None
	lora: tuple = ()
	start: float | None = None
	duration: float | None = None
	upscale: int | None = None
	upscale_model: str | None = None
	restart: bool = False


class SDXLUnionBackend:
	''' SDXL + ControlNet-Union img2img. `generate` is the whole interface the frame loop
	needs, so other still-image models can be dropped in behind it. '''

	def __init__(self, cfg, device=None):
		import torch
		from diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionImg2ImgPipeline

		if cfg.control not in UNION_MODES:
			raise ValueError(f"control {cfg.control!r} is not supported by the SDXL union backend; use one of {sorted(UNION_MODES)}")
		self.cfg = cfg
		self.torch = torch
		device = device or common.pick_device()
		dtype = torch.float16 if device.startswith("cuda") else torch.float32
		controlnet = ControlNetUnionModel.from_pretrained(cfg.controlnet, torch_dtype=dtype)
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
		self.pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
			cfg.model, controlnet=controlnet, vae=vae, torch_dtype=dtype, use_safetensors=True,
			**({"variant": "fp16"} if cfg.model == FrameConfig.model else {}))
		common.load_loras(self.pipe, cfg.lora)
		if cfg.ip_adapter > 0:
			self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
		if device.startswith("cuda") and common.cuda_vram_gb() < 14:
			self.pipe.enable_model_cpu_offload()
		else:
			self.pipe.to(device)
		self.pipe.set_progress_bar_config(disable=True)

	def generate(self, init, control, denoise, params, reference=None):
		cfg = self.cfg
		kwargs = {}
		if cfg.ip_adapter > 0:
			self.pipe.set_ip_adapter_scale(params.get("ip_adapter", cfg.ip_adapter))
			kwargs["ip_adapter_image"] = Image.fromarray(reference if reference is not None else init)
		generator = None
		if cfg.seed is not None:
			generator = self.torch.Generator("cpu").manual_seed(cfg.seed)
		# img2img runs int(steps * strength) steps; keep that at one or more
		denoise = min(1.0, max(denoise, 1.0 / cfg.steps + 1e-3))
		image = self.pipe(
			prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
			image=Image.fromarray(init), strength=denoise,
			control_image=[Image.fromarray(control)], control_mode=[UNION_MODES[cfg.control]],
			controlnet_conditioning_scale=params.get("control_scale", cfg.control_scale),
			guidance_scale=params.get("guidance", cfg.guidance),
			num_inference_steps=cfg.steps, width=init.shape[1], height=init.shape[0],
			generator=generator, **kwargs,
		).images[0]
		return np.asarray(image.convert("RGB"))


def run(input_path, output_path, cfg, backend=None, control_fn=None, on_frame=None):
	''' restyle `input_path` into `output_path`.

	`on_frame(index, seconds)` may return a dict overriding `denoise`, `feedthrough`,
	`control_scale`, `guidance` or `ip_adapter` for that frame; this is the hook parameter
	animation (audio-reactive or otherwise) plugs into. '''
	info = video.probe(input_path)
	fps = cfg.fps or info.fps
	size = video.fit_size(info.width, info.height, short_side=cfg.short_side, max_area=cfg.max_area, multiple=16)
	work = video.Workdir(output_path, f"frame {input_path} {size[0]}x{size[1]}@{fps}", restart=cfg.restart)
	print(f"frame: {info.width}x{info.height} -> {size[0]}x{size[1]} at {fps:g} fps, resuming at frame {work.done}")

	backend = backend or SDXLUnionBackend(cfg)
	control_fn = control_fn or controls.create_control(cfg.control)
	compensate = motion.MotionCompensator(cfg.motion_estimator, cfg.motion_alpha, cfg.motion_sigma, cfg.occlusion)
	chains = {
		name: filters.parse_video_filter_spec(getattr(cfg, f"{name}_filter"), fps=fps)
		for name in ("input", "feedback", "control", "output")}

	total = max(1, int((cfg.duration or info.duration) * fps))
	prev_input = prev_output = first_output = None
	if work.done:
		prev_output = work.load(work.done - 1)
		first_output = work.load(0)

	frames_in = video.read_frames(input_path, size, fps=cfg.fps, start=cfg.start, duration=cfg.duration)
	for index, frame in enumerate(tqdm(frames_in, total=total, unit="frame")):
		position = dict(x=index / total, n=index, fps=fps)
		frame = chains["input"](frame, **position)
		if index < work.done:
			prev_input = frame
			continue
		params = (on_frame(index, index / fps) if on_frame else None) or {}
		feedthrough = params.get("feedthrough", cfg.feedthrough)

		if prev_output is None:
			feedback = None
			init = frame
			denoise = cfg.first_denoise
		else:
			feedback, occluded = compensate(prev_input, frame, prev_output)
			if occluded is not None:
				soft = cv2.GaussianBlur(occluded, (0, 0), 3)[..., None]
				feedback = (feedback * (1 - soft) + frame * soft).astype(np.uint8)
			feedback = chains["feedback"](feedback, **position)
			init = cv2.addWeighted(frame, feedthrough, feedback, 1.0 - feedthrough, 0)
			denoise = params.get("denoise", cfg.denoise)

		control = chains["control"](control_fn([frame])[0], **position)
		output = backend.generate(init, control, denoise, params, reference=feedback)
		if first_output is None:
			first_output = output
		else:
			output = controls.match_color(output, first_output, cfg.color_fix)
		output = chains["output"](output, **position)

		work.save(index, output)
		work.commit(index + 1)
		prev_input, prev_output = frame, output

	return common.finish(work, output_path, fps, audio_from=input_path if info.has_audio else None,
		audio_start=cfg.start, upscale_to=cfg.upscale, upscale_model=cfg.upscale_model)

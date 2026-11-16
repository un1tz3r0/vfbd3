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

DEFAULTS = dict(model="stabilityai/stable-diffusion-xl-base-1.0", steps=25, guidance=7.5)

# few-step setups: a distillation LoRA (or distilled checkpoint) with the scheduler it expects and
# no classifier-free guidance. `steps` is the full schedule; a frame runs steps * denoise of them.
FAST = {
	"lcm": dict(steps=8, guidance=1.0, scheduler=("LCMScheduler", {}),
		lora="latent-consistency/lcm-lora-sdxl:pytorch_lora_weights.safetensors"),
	"lightning": dict(steps=8, guidance=1.0, scheduler=("EulerDiscreteScheduler", {"timestep_spacing": "trailing"}),
		lora="ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors"),
	"dmd2": dict(steps=4, guidance=1.0, scheduler=("LCMScheduler", {}),
		lora="tianweiy/DMD2:dmd2_sdxl_4step_lora_fp16.safetensors"),
	# SDXL-Turbo is trained at 512 px and for 1-4 steps: pair it with short_side 512 and an upscale
	"turbo": dict(steps=4, guidance=1.0, model="stabilityai/sdxl-turbo",
		scheduler=("EulerAncestralDiscreteScheduler", {"timestep_spacing": "trailing"})),
}


@dataclass
class FrameConfig:
	prompt: str
	negative_prompt: str = ""
	model: str | None = None       # default stabilityai/stable-diffusion-xl-base-1.0
	controlnet: str = "xinsir/controlnet-union-sdxl-1.0"
	control: str = "depth"         # depth or canny
	control_scale: float = 1.0
	short_side: int = 720
	max_area: int = 1280 * 720
	fps: float | None = None       # None keeps the input's frame rate
	fast: str | None = None        # a FAST preset; sets steps, guidance (and model) unless given
	steps: int | None = None       # default 25
	guidance: float | None = None  # default 7.5
	tiny_vae: bool = False         # TAESDXL instead of the full VAE: far faster, slightly softer
	denoise: float = 0.5           # img2img strength: how much of each frame is re-generated
	first_denoise: float = 1.0     # same, for the first frame (nothing to feed back yet)
	feedthrough: float = 0.0       # share of the raw input in the init image; the rest is feedback
	motion_estimator: str = "dis"
	motion_alpha: float = 0.0      # temporal flow smoothing
	motion_sigma: float = 0.0      # spatial flow smoothing
	occlusion: bool = True         # re-generate areas the flow cannot explain (newly revealed ones)
	occlusion_denoise: float = 1.0 # denoise inside those areas; the rest of the frame gets `denoise`
	ip_adapter: float = 0.0        # > 0: also feed the warped previous output in as an image prompt
	color_fix: float = 0.0         # pull each output's colours toward the first output
	noise: str = "warp"            # warp: one noise field carried along the motion; fixed; random
	seed: int | None = None
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

	def __post_init__(self):
		common.apply_preset(self, FAST, DEFAULTS)
		if self.noise not in ("warp", "fixed", "random"):
			raise ValueError(f"noise must be warp, fixed or random, got {self.noise!r}")


class SDXLUnionBackend:
	''' SDXL + ControlNet-Union img2img. `generate` is the whole interface the frame loop
	needs, so other still-image models can be dropped in behind it. '''

	def __init__(self, cfg, device=None, pipe=None):
		import torch
		self.cfg = cfg
		self.torch = torch
		if cfg.control not in UNION_MODES:
			raise ValueError(f"control {cfg.control!r} is not supported by the SDXL union backend; use one of {sorted(UNION_MODES)}")
		device = device or common.pick_device()
		self.pipe = pipe = pipe or self.load(cfg, device)
		if cfg.fast and "scheduler" in FAST[cfg.fast]:
			import diffusers
			name, args = FAST[cfg.fast]["scheduler"]
			pipe.scheduler = getattr(diffusers, name).from_config(pipe.scheduler.config, **args)
		common.load_loras(pipe, cfg.lora)
		if cfg.ip_adapter > 0:
			pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
		if device.startswith("cuda") and common.cuda_vram_gb() < 14:
			pipe.enable_model_cpu_offload()
		else:
			pipe.to(device)
		pipe.set_progress_bar_config(disable=True)
		# the prompt never changes: encode it once, then drop the text encoders to free their memory
		names = ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds")
		with torch.no_grad():
			self.embeds = dict(zip(names, pipe.encode_prompt(
				cfg.prompt, negative_prompt=cfg.negative_prompt, device=pipe._execution_device)))
		pipe.text_encoder = pipe.text_encoder_2 = None
		common.free_memory()
		self.generator = torch.Generator("cpu")
		if cfg.seed is not None:
			self.generator.manual_seed(cfg.seed)

	@staticmethod
	def load(cfg, device):
		import torch
		from diffusers import AutoencoderKL, AutoencoderTiny, ControlNetUnionModel, StableDiffusionXLControlNetUnionImg2ImgPipeline
		dtype = torch.float16 if device.startswith("cuda") else torch.float32
		controlnet = ControlNetUnionModel.from_pretrained(cfg.controlnet, torch_dtype=dtype)
		if cfg.tiny_vae:
			vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=dtype)
		else:
			vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
		return StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
			cfg.model, controlnet=controlnet, vae=vae, torch_dtype=dtype, use_safetensors=True,
			**({"variant": "fp16"} if cfg.model.startswith("stabilityai/") else {}))

	def start(self, init, denoise, noise, regen):
		''' the latents a frame starts from: `init` encoded, plus `noise` at the level `denoise`
		calls for. With a `regen` mask, the run starts at the level of the highest per-pixel
		denoise instead, and the returned step callback holds every pixel at its noised original
		until the schedule comes down to that pixel's own denoise (differential diffusion). '''
		from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_latents
		torch, pipe, steps = self.torch, self.pipe, self.cfg.steps
		device, dtype = pipe._execution_device, pipe.unet.dtype
		with torch.no_grad():
			image = pipe.image_processor.preprocess(Image.fromarray(init)).to(device, pipe.vae.dtype)
			clean = retrieve_latents(pipe.vae.encode(image), sample_mode="argmax").to(dtype) * pipe.vae.config.scaling_factor
		if noise is None:
			noise = torch.randn(clean.shape, generator=self.generator)
		else:
			noise = torch.from_numpy(np.ascontiguousarray(noise))[None]
		noise = noise.to(device, dtype)

		strength, own_steps = denoise, None
		if regen is not None:
			area = torch.nn.functional.interpolate(torch.from_numpy(regen)[None, None].float(), size=clean.shape[-2:], mode="area")
			per_pixel = denoise + (max(self.cfg.occlusion_denoise, denoise) - denoise) * area
			strength = float(per_pixel.max())
			# how many steps img2img would run for each pixel at its own denoise
			own_steps = (per_pixel * steps + 1e-4).floor().clamp(min=1).to(device)
		pipe.scheduler.set_timesteps(steps, device=device)
		timesteps, _ = pipe.get_timesteps(steps, strength, device)
		latents = pipe.scheduler.add_noise(clean, noise, timesteps[:1])

		def hold(pipe, i, t, tensors):
			remaining = len(timesteps) - i - 1
			if remaining:
				held = pipe.scheduler.add_noise(clean, noise, timesteps[i + 1:i + 2])
				tensors["latents"] = torch.where(own_steps <= remaining, held, tensors["latents"])
			return tensors

		return dict(latents=latents, strength=strength, callback_on_step_end=hold if regen is not None else None)

	def generate(self, init, control, denoise, params, reference=None, noise=None, regen=None):
		''' `noise`: (4, h/8, w/8) starting noise, or None for the backend's own.
		`regen`: (h, w) float mask, 1 where the frame is to be generated anew. '''
		cfg, pipe = self.cfg, self.pipe
		kwargs = {}
		if cfg.ip_adapter > 0:
			pipe.set_ip_adapter_scale(params.get("ip_adapter", cfg.ip_adapter))
			kwargs["ip_adapter_image"] = Image.fromarray(reference if reference is not None else init)
		# img2img runs int(steps * strength) steps; keep that at one or more
		denoise = min(1.0, max(denoise, 1.0 / cfg.steps + 1e-3))
		image = pipe(
			**self.embeds, **self.start(init, denoise, noise, regen), image=Image.fromarray(init),
			control_image=[Image.fromarray(control)], control_mode=[UNION_MODES[cfg.control]],
			controlnet_conditioning_scale=params.get("control_scale", cfg.control_scale),
			guidance_scale=params.get("guidance", cfg.guidance),
			num_inference_steps=cfg.steps, width=init.shape[1], height=init.shape[0],
			generator=self.generator, **kwargs,
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
	field = None if cfg.noise == "random" else motion.NoiseField(size[1], size[0], seed=cfg.seed)
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

		regen = None
		if prev_output is None:
			feedback = None
			init = frame
			denoise = cfg.first_denoise
		else:
			feedback, occluded = compensate(prev_input, frame, prev_output)
			if cfg.noise == "warp":
				field.advance(compensate.flow, occluded)
			if occluded is not None:
				# drop speckle, then feather: those areas start from the input and are re-generated
				regen = cv2.GaussianBlur(cv2.morphologyEx(occluded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), (0, 0), 3)
				soft = regen[..., None]
				feedback = (feedback * (1 - soft) + frame * soft).astype(np.uint8)
				if regen.max() < 0.05:
					regen = None
			feedback = chains["feedback"](feedback, **position)
			init = cv2.addWeighted(frame, feedthrough, feedback, 1.0 - feedthrough, 0)
			denoise = params.get("denoise", cfg.denoise)

		control = chains["control"](control_fn([frame])[0], **position)
		output = backend.generate(init, control, denoise, params, reference=feedback,
			noise=field.latent() if field else None, regen=regen)
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

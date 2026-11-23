"""
Image models behind the frame engine. A backend turns (init image, control image, denoise,
starting noise, regeneration mask) into an output frame; the frame loop doesn't care which
model does it.

    sdxl  SDXL + ControlNet-Union (one ControlNet for every control family), optional
          reference-only attention and IP-Adapter
    flux  FLUX.1 with a GGUF-quantized transformer + ControlNet-Union-Pro-2.0; fits a T4
    sd3   Stable Diffusion 3 Medium + the per-family InstantX / alimama SD3 ControlNets

All three start from latents the backend builds itself: the init image encoded, plus the
caller's noise at the level `denoise` calls for. With a regeneration mask the run starts at
the highest per-pixel denoise instead, and a step callback holds each pixel at its noised
original until the schedule comes down to that pixel's own level (differential diffusion).
"""

import inspect

import numpy as np
from PIL import Image

from . import common, control as controls

# ControlNet-Union (xinsir, SDXL) mode per control family
SDXL_UNION_MODES = {"pose": 0, "depth": 1, "softedge": 2, "lines": 3, "normal": 4}

SD3_CONTROLNETS = {
	"depth": "InstantX/SD3-Controlnet-Depth",
	"lines": "InstantX/SD3-Controlnet-Canny",
	"pose": "InstantX/SD3-Controlnet-Pose",
	"softedge": "alimama-creative/SD3-Controlnet-Softedge",
}

DEFAULTS = {
	"sdxl": dict(model="stabilityai/stable-diffusion-xl-base-1.0", controlnet="xinsir/controlnet-union-sdxl-1.0", steps=25, guidance=7.5),
	# FLUX.1-dev is guidance-distilled: `guidance` is its embedded guidance, there is no CFG pass
	"flux": dict(model="black-forest-labs/FLUX.1-dev", controlnet="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
		transformer="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf", steps=20, guidance=3.5),
	"sd3": dict(model="stabilityai/stable-diffusion-3-medium-diffusers", steps=28, guidance=7.0),
}

# few-step setups: a distillation LoRA (or distilled checkpoint) with the scheduler it expects and
# no classifier-free guidance. `steps` is the full schedule; a frame runs steps * denoise of them.
FAST = {
	"sdxl": {
		"lcm": dict(steps=8, guidance=1.0, scheduler=("LCMScheduler", {}),
			lora="latent-consistency/lcm-lora-sdxl:pytorch_lora_weights.safetensors"),
		"lightning": dict(steps=8, guidance=1.0, scheduler=("EulerDiscreteScheduler", {"timestep_spacing": "trailing"}),
			lora="ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors"),
		"dmd2": dict(steps=4, guidance=1.0, scheduler=("LCMScheduler", {}),
			lora="tianweiy/DMD2:dmd2_sdxl_4step_lora_fp16.safetensors"),
		# SDXL-Turbo is trained at 512 px and for 1-4 steps: pair it with short_side 512 and an upscale
		"turbo": dict(steps=4, guidance=1.0, model="stabilityai/sdxl-turbo",
			scheduler=("EulerAncestralDiscreteScheduler", {"timestep_spacing": "trailing"})),
	},
	"flux": {
		"schnell": dict(steps=4, guidance=0.0, model="black-forest-labs/FLUX.1-schnell",
			transformer="https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K_S.gguf"),
	},
	"sd3": {},
}


class Backend:
	''' the shared part: prompt encoding, LoRAs, starting latents, masked regeneration.
	Subclasses provide `load`, `encode_prompt`, `call` and the noise/schedule specifics. '''

	latent_channels = 4
	offloaded = False
	visualize = frozenset()   # preview panels being shown; backends may record extra data for some

	def __init__(self, cfg, device=None, pipe=None):
		import torch
		self.cfg, self.torch = cfg, torch
		self.family = controls.family(cfg.control)
		self.check_control()
		device = device or common.pick_device()
		self.embeds = None
		self.pipe = pipe = pipe or self.load(device)
		if cfg.fast and "scheduler" in FAST[cfg.backend][cfg.fast]:
			import diffusers
			name, args = FAST[cfg.backend][cfg.fast]["scheduler"]
			pipe.scheduler = getattr(diffusers, name).from_config(pipe.scheduler.config, **args)
		common.load_loras(pipe, cfg.lora)
		self.setup(device)
		if not self.offloaded:
			pipe.to(device)
		pipe.set_progress_bar_config(disable=True)
		if self.embeds is None:
			with torch.no_grad():
				self.embeds = self.encode_prompt(pipe, pipe._execution_device)
		for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
			if hasattr(pipe, name):
				setattr(pipe, name, None)
		common.free_memory()
		self.encoder = pipe.vae
		if cfg.tiny_vae:
			self.use_tiny_vae(device)
		self.latent_scale = pipe.vae_scale_factor
		self.generator = torch.Generator("cpu")
		if cfg.seed is not None:
			self.generator.manual_seed(cfg.seed)

	def check_control(self):
		pass

	def setup(self, device):
		''' after loading and LoRAs, before the prompt is encoded: adapters, offload. '''

	def use_tiny_vae(self, device):
		raise ValueError(f"tiny_vae is not supported by the {self.cfg.backend} backend")

	def _tiny(self, repo, device):
		from diffusers import AutoencoderTiny
		return AutoencoderTiny.from_pretrained(repo, torch_dtype=self.pipe.vae.dtype).to(device)

	def encode(self, image):
		''' RGB uint8 -> latents as the denoiser sees them (scaled and shifted). '''
		from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_latents
		pipe, vae = self.pipe, self.encoder
		with self.torch.no_grad():
			pixels = pipe.image_processor.preprocess(Image.fromarray(image)).to(pipe._execution_device, vae.dtype)
			latents = retrieve_latents(vae.encode(pixels), sample_mode="argmax")
		shift = getattr(vae.config, "shift_factor", None) or 0.0
		return ((latents - shift) * vae.config.scaling_factor).to(self.dtype)

	@property
	def dtype(self):
		return self.denoiser.dtype

	def pack(self, latents):
		return latents

	def start(self, init, denoise, noise, regen):
		''' (pipeline kwargs, step callback or None) for one frame. '''
		torch = self.torch
		clean = self.encode(init)
		device = clean.device
		if noise is None:
			noise = torch.randn(clean.shape, generator=self.generator)
		else:
			noise = torch.from_numpy(np.ascontiguousarray(noise))[None]
		noise = noise.to(device, clean.dtype)

		strength, own_steps = denoise, None
		steps = self.cfg.steps
		if regen is not None:
			area = torch.nn.functional.interpolate(torch.from_numpy(regen)[None, None].float(), size=clean.shape[-2:], mode="area")
			per_pixel = denoise + (max(self.cfg.occlusion_denoise, denoise) - denoise) * area
			strength = float(per_pixel.max())
			# how many steps img2img would run for each pixel at its own denoise
			own_steps = self.pack((per_pixel * steps + 1e-4).floor().clamp(min=1).expand_as(clean).to(device, clean.dtype))
		levels, n, kwargs = self.schedule(strength, clean)
		clean_p, noise_p = self.pack(clean), self.pack(noise)
		kwargs["latents"] = self.mix(clean_p, noise_p, levels[0])

		def hold(pipe, i, t, tensors):
			remaining = n - i - 1
			if remaining > 0:
				held = self.mix(clean_p, noise_p, levels[i + 1])
				tensors["latents"] = torch.where(own_steps <= remaining, held, tensors["latents"])
			return tensors

		self.noise, self.steps_run = noise, n
		return kwargs, (hold if regen is not None else None)

	def generate(self, init, control, denoise, params, images=None, noise=None, regen=None):
		''' `noise`: (C, h, w) starting noise in latent space, or None for the backend's own.
		`regen`: (h, w) float mask, 1 where the frame is to be generated anew. `images`: the
		loop's other images, for image conditioning: input (current input frame), feedback
		(motion-compensated previous output), previous (previous output as it was). '''
		cfg = self.cfg
		# img2img runs int(steps * denoise) steps; keep that at one or more
		denoise = min(1.0, max(denoise, 1.0 / cfg.steps + 1e-3))
		kwargs, callback = self.start(init, denoise, noise, regen)
		kwargs.update(self.embeds, callback_on_step_end=callback,
			controlnet_conditioning_scale=params.get("control_scale", cfg.control_scale),
			guidance_scale=params.get("guidance", cfg.guidance),
			width=init.shape[1], height=init.shape[0], generator=self.generator)
		image = self.call(kwargs, init, control, params, images or {})
		return np.asarray(image.convert("RGB"))


class SDXLBackend(Backend):
	@property
	def denoiser(self):
		return self.pipe.unet

	def check_control(self):
		if self.family not in SDXL_UNION_MODES:
			raise ValueError(f"control {self.cfg.control!r} ({self.family}) has no ControlNet-Union mode; "
				f"use a {', '.join(SDXL_UNION_MODES)} control")

	def load(self, device):
		import torch
		from diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionImg2ImgPipeline
		cfg = self.cfg
		dtype = torch.float16 if device.startswith("cuda") else torch.float32
		controlnet = ControlNetUnionModel.from_pretrained(cfg.controlnet, torch_dtype=dtype)
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
		return StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
			cfg.model, controlnet=controlnet, vae=vae, torch_dtype=dtype, use_safetensors=True,
			**({"variant": "fp16"} if cfg.model.startswith("stabilityai/") else {}))

	def setup(self, device):
		cfg, pipe = self.cfg, self.pipe
		if cfg.ip_adapter > 0:
			pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
		# installed on first use: the reference preview panels can call for it too
		self.reference, self.correspondence, self.attention = None, None, None
		if device.startswith("cuda") and common.cuda_vram_gb() < 14:
			pipe.enable_model_cpu_offload()
			self.offloaded = True

	def encode_prompt(self, pipe, device):
		names = ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds")
		return dict(zip(names, pipe.encode_prompt(self.cfg.prompt, negative_prompt=self.cfg.negative_prompt, device=device)))

	def use_tiny_vae(self, device):
		# TAESDXL's decoder pushes colour up a little on every round trip, which a feedback loop
		# compounds; its encoder doesn't. "encoder" keeps the full decoder.
		tiny = self._tiny("madebyollin/taesdxl", device)
		self.encoder = tiny
		if self.cfg.tiny_vae == "both":
			self.pipe.vae = tiny

	def schedule(self, strength, clean):
		pipe, device = self.pipe, clean.device
		pipe.scheduler.set_timesteps(self.cfg.steps, device=device)
		timesteps, _ = pipe.get_timesteps(self.cfg.steps, strength, device)
		return timesteps, len(timesteps), dict(strength=strength, num_inference_steps=self.cfg.steps)

	def mix(self, clean, noise, t):
		return self.pipe.scheduler.add_noise(clean, noise, t.reshape(1))

	@property
	def wants_correspondence(self):
		return self.cfg.flow_remap > 0 or bool(self.visualize & {"correspondence", "confidence"})

	def references(self, params, images):
		''' ([(latents, strength)], [name]) for shared attention: the previous output first (it
		also drives AdaIN), the current input last (it also gives the correspondence, for flow
		remap and the preview, which add it even at strength 0). '''
		cfg, refs, names = self.cfg, [], []
		previous = images.get("feedback" if cfg.reference_warp else "previous")
		if cfg.reference > 0 and previous is not None:
			refs.append((self.encode(previous), params.get("reference", cfg.reference)))
			names.append("previous")
		if (cfg.reference_input > 0 or self.wants_correspondence) and images.get("input") is not None:
			refs.append((self.encode(images["input"]), params.get("reference_input", cfg.reference_input)))
			names.append("input")
		return refs, names

	def call(self, kwargs, init, control, params, images):
		import contextlib
		cfg, pipe = self.cfg, self.pipe
		if cfg.ip_adapter > 0:
			pipe.set_ip_adapter_scale(params.get("ip_adapter", cfg.ip_adapter))
			kwargs["ip_adapter_image"] = Image.fromarray(images.get("feedback") if images.get("feedback") is not None else init)
		active = contextlib.nullcontext()
		refs, names = self.references(params, images)
		correspond, shares = self.wants_correspondence and names[-1:] == ["input"], "attention" in self.visualize
		if refs:
			if self.reference is None:
				from .reference import Reference
				self.reference = Reference(pipe.unet)
			active = self.reference.active(pipe.scheduler, refs, self.noise, adain=cfg.reference_adain,
				record=correspond or shares, steps=self.steps_run)
		with active:
			image = pipe(**kwargs, image=Image.fromarray(init), control_image=[Image.fromarray(control)],
				control_mode=[SDXL_UNION_MODES[self.family]]).images[0]
		self.correspondence = self.reference.correspondence if refs and correspond else None
		# (h, w, 1 + references) attention split, and what each share is of
		self.attention = (self.reference.shares, ("self", *names)) if refs and shares else None
		return image


class FlowBackend(Backend):
	''' rectified-flow models (Flux, SD3): x_s = (1 - s) * clean + s * noise. The run is
	handed to the text-to-image ControlNet pipeline as pre-noised latents plus the tail of the
	sigma schedule, which is exactly what img2img would run. '''

	latent_channels = 16

	@property
	def denoiser(self):
		return self.pipe.transformer

	def load(self, device):
		''' text encoders first (T5 alone is 9 GB): encode the prompt on the GPU, free them,
		then load the denoiser. '''
		import torch
		cfg, cls = self.cfg, self.pipeline_class()
		dtype = common.pick_dtype(device)
		on_gpu = device.startswith("cuda")
		text = cls.from_pretrained(cfg.model, transformer=None, vae=None, controlnet=None, torch_dtype=dtype,
			**({"device_map": "balanced"} if on_gpu else {}))
		with torch.no_grad():
			self.embeds = self.encode_prompt(text, device)
		del text
		common.free_memory()
		no_text = {name: None for name in ("text_encoder", "text_encoder_2", "text_encoder_3")
			if name in inspect.signature(cls.__init__).parameters}
		return cls.from_pretrained(cfg.model, transformer=self.load_transformer(dtype), controlnet=self.load_controlnet(dtype),
			torch_dtype=dtype, **no_text)

	def load_transformer(self, dtype):
		cfg = self.cfg
		cls = self.transformer_class()
		if not cfg.transformer:
			return cls.from_pretrained(cfg.model, subfolder="transformer", torch_dtype=dtype)
		args = {}
		if cfg.transformer.endswith(".gguf"):
			from diffusers import GGUFQuantizationConfig
			args["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
		return cls.from_single_file(cfg.transformer, config=cfg.model, subfolder="transformer", torch_dtype=dtype, **args)

	def schedule(self, strength, clean):
		from diffusers.pipelines.flux.pipeline_flux import calculate_shift
		scheduler, steps = self.pipe.scheduler, self.cfg.steps
		sigmas = np.linspace(1.0, 1 / steps, steps)[steps - int(steps * strength):]
		args = {}
		if scheduler.config.get("use_dynamic_shifting"):
			c = scheduler.config
			seq = self.pack(clean).shape[1] if self.packed else clean.shape[-1] * clean.shape[-2] // 4
			args["mu"] = calculate_shift(seq, c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096),
				c.get("base_shift", 0.5), c.get("max_shift", 1.15))
		# the pipeline sets the same schedule again; doing it here first gives the exact levels it will use
		scheduler.set_timesteps(sigmas=sigmas, device=clean.device, **args)
		return scheduler.sigmas.to(clean.device), len(sigmas), dict(sigmas=list(sigmas), num_inference_steps=len(sigmas))

	packed = False

	def mix(self, clean, noise, sigma):
		return clean + (noise - clean) * sigma


class FluxBackend(FlowBackend):
	packed = True

	@staticmethod
	def pipeline_class():
		from diffusers import FluxControlNetPipeline
		return FluxControlNetPipeline

	@staticmethod
	def transformer_class():
		from diffusers import FluxTransformer2DModel
		return FluxTransformer2DModel

	def load_controlnet(self, dtype):
		from diffusers import FluxControlNetModel
		return FluxControlNetModel.from_pretrained(self.cfg.controlnet, torch_dtype=dtype)

	def encode_prompt(self, pipe, device):
		prompt_embeds, pooled, _ = pipe.encode_prompt(self.cfg.prompt, prompt_2=None, device=device)
		return dict(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled)

	def use_tiny_vae(self, device):
		tiny = self._tiny("madebyollin/taef1", device)
		self.encoder = tiny
		if self.cfg.tiny_vae == "both":
			self.pipe.vae = tiny

	def pack(self, latents):
		b, c, h, w = latents.shape
		return self.pipe._pack_latents(latents, b, c, h, w)

	def call(self, kwargs, init, control, params, images):
		controlnet = self.pipe.controlnet
		modes = getattr(controlnet.config, "num_mode", None)
		mode = {"canny": 0, "lines": 0, "depth": 2, "pose": 4, "gray": 5}.get(self.family) if modes else None
		return self.pipe(**kwargs, control_image=Image.fromarray(control), control_mode=mode).images[0]


class SD3Backend(FlowBackend):
	@staticmethod
	def pipeline_class():
		from diffusers import StableDiffusion3ControlNetPipeline
		return StableDiffusion3ControlNetPipeline

	@staticmethod
	def transformer_class():
		from diffusers import SD3Transformer2DModel
		return SD3Transformer2DModel

	def check_control(self):
		if not self.cfg.controlnet and self.family not in SD3_CONTROLNETS:
			raise ValueError(f"no default SD3 ControlNet for control {self.cfg.control!r} ({self.family}); "
				f"use a {', '.join(SD3_CONTROLNETS)} control or pass --controlnet")

	def load_controlnet(self, dtype):
		from diffusers import SD3ControlNetModel
		return SD3ControlNetModel.from_pretrained(self.cfg.controlnet or SD3_CONTROLNETS[self.family], torch_dtype=dtype)

	def encode_prompt(self, pipe, device):
		names = ("prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds")
		return dict(zip(names, pipe.encode_prompt(self.cfg.prompt, None, None, device=device,
			do_classifier_free_guidance=self.cfg.guidance > 1, negative_prompt=self.cfg.negative_prompt or None)))

	def use_tiny_vae(self, device):
		# the SD3 ControlNet pipeline also encodes the control image with pipe.vae, so only the
		# frame encoder can be swapped
		self.encoder = self._tiny("madebyollin/taesd3", device)

	def call(self, kwargs, init, control, params, images):
		return self.pipe(**kwargs, control_image=Image.fromarray(control)).images[0]


BACKENDS = {"sdxl": SDXLBackend, "flux": FluxBackend, "sd3": SD3Backend}


def create(cfg, device=None, pipe=None):
	return BACKENDS[cfg.backend](cfg, device=device, pipe=pipe)

"""
Per-frame img2img with motion-compensated feedback, the technique from controlnetvideo.

Each frame: estimate the input's motion since the previous frame, carry the previous
*output* along that motion, blend it with the current input, and denoise the blend under a
control signal extracted from the input. The result is fed back for the next frame. The
image model is a backend (see backends.py): SDXL, Flux or SD3.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import backends, common, control as controls, filters, motion, preview as previews, video

FAST = backends.FAST


@dataclass
class FrameConfig:
	prompt: str
	negative_prompt: str = ""
	backend: str = "sdxl"          # sdxl, flux or sd3
	model: str | None = None       # base model repo; default depends on the backend
	transformer: str | None = None # flux/sd3: single-file or GGUF transformer checkpoint (path or URL)
	controlnet: str | None = None  # default depends on the backend (and, for sd3, the control)
	control: str = "depth"         # see control.CONTROLS
	control_scale: float = 1.0
	short_side: int = 720
	max_area: int = 1280 * 720
	fps: float | None = None       # None keeps the input's frame rate
	fast: str | None = None        # a FAST preset; sets steps, guidance (and model) unless given
	steps: int | None = None       # default depends on the backend
	guidance: float | None = None
	tiny_vae: str | None = None    # "encoder": tiny VAE encodes frames; "both": decodes too (faster, but drifts colour in feedback)
	denoise: float = 0.5           # img2img strength: how much of each frame is re-generated
	first_denoise: float = 1.0     # same, for the first frame (nothing to feed back yet)
	feedthrough: float = 0.0       # share of the raw input in the init image; the rest is feedback
	motion_estimator: str = "dis"
	motion_alpha: float = 0.0      # temporal flow smoothing
	motion_sigma: float = 0.0      # spatial flow smoothing
	occlusion: bool = True         # re-generate areas the flow cannot explain (newly revealed ones)
	occlusion_denoise: float = 1.0 # denoise inside those areas; the rest of the frame gets `denoise`
	ip_adapter: float = 0.0        # > 0: also feed the warped previous output in as an image prompt
	reference: float = 0.0         # sdxl: shared self-attention with the previous output, at this strength
	reference_warp: bool = True    # ... motion-compensated (True) or as it was (False)
	reference_input: float = 0.0   # sdxl: shared self-attention with the current input frame
	reference_adain: float = 0.0   # 0..1, match feature statistics to the first reference (holds colour)
	flow_remap: float = 0.0        # sdxl, experimental: 0..1, move the output by its own objects' input motion
	color_fix: float = 0.0         # pull each output's colours toward the first output
	noise: str = "warp"            # warp: one noise field carried along the motion; fixed; random
	seed: int | None = None
	input_filter: str | None = None
	feedback_filter: str | None = None
	control_filter: str | None = None
	output_filter: str | None = None
	preview: str | None = None     # image file rewritten (atomically) after every frame
	preview_panels: str = "output" # comma-separated preview.PANELS, or "all", for a mosaic
	lora: tuple = ()
	start: float | None = None
	duration: float | None = None
	upscale: int | None = None
	upscale_model: str | None = None
	restart: bool = False

	def __post_init__(self):
		if self.backend not in backends.BACKENDS:
			raise ValueError(f"backend must be one of {', '.join(backends.BACKENDS)}, got {self.backend!r}")
		common.apply_preset(self, FAST[self.backend], backends.DEFAULTS[self.backend])
		if self.tiny_vae not in (None, "encoder", "both"):
			raise ValueError(f"tiny_vae must be encoder or both, got {self.tiny_vae!r}")
		if self.backend != "sdxl" and (self.reference or self.reference_input or self.flow_remap or self.ip_adapter):
			raise ValueError("reference, reference_input, flow_remap and ip_adapter need the sdxl backend")
		if self.noise not in ("warp", "fixed", "random"):
			raise ValueError(f"noise must be warp, fixed or random, got {self.noise!r}")


def confidence(spread, scale):
	''' 0..1 per coarse token: near 1 where attention was concentrated within about one token. '''
	return np.exp(-(spread * scale / 64.0) ** 2)


def run(input_path, output_path, cfg, backend=None, control_fn=None, on_frame=None, on_preview=None):
	''' restyle `input_path` into `output_path`.

	`on_frame(index, seconds)` may return a dict overriding `denoise`, `feedthrough`,
	`control_scale`, `guidance`, `ip_adapter`, `reference` or `reference_input` for that frame;
	this is the hook parameter animation (audio-reactive or otherwise) plugs into.
	`on_preview(image)` receives each preview image (see `cfg.preview_panels`). '''
	info = video.probe(input_path)
	fps = cfg.fps or info.fps
	size = video.fit_size(info.width, info.height, short_side=cfg.short_side, max_area=cfg.max_area, multiple=16)
	work = video.Workdir(output_path, f"frame {input_path} {size[0]}x{size[1]}@{fps}", restart=cfg.restart)
	print(f"frame: {info.width}x{info.height} -> {size[0]}x{size[1]} at {fps:g} fps, resuming at frame {work.done}")

	show = previews.Preview(cfg.preview, cfg.preview_panels, on_preview)
	backend = backend or backends.create(cfg)
	control_fn = control_fn or controls.create_control(cfg.control)
	compensate = motion.MotionCompensator(cfg.motion_estimator, cfg.motion_alpha, cfg.motion_sigma, cfg.occlusion)
	scale = getattr(backend, "latent_scale", 8)
	field = None if cfg.noise == "random" else motion.NoiseField(
		size[1], size[0], channels=getattr(backend, "latent_channels", 4), block=scale, seed=cfg.seed)
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

		regen = feedback = warped = None
		if prev_output is None:
			init = frame
			denoise = cfg.first_denoise
		else:
			correspondence = getattr(backend, "correspondence", None)
			remap = None
			if cfg.flow_remap > 0 and correspondence is not None:
				corr, spread = correspondence
				remap = (corr * scale, cfg.flow_remap * confidence(spread, scale))
			feedback, occluded = compensate(prev_input, frame, prev_output, remap=remap)
			warped = feedback
			if cfg.noise == "warp":
				field.advance(compensate.output_flow, occluded)
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
		noise = field.latent() if field else None
		output = backend.generate(init, control, denoise, params, noise=noise, regen=regen,
			images=dict(input=frame, feedback=feedback, previous=prev_output))
		if first_output is None:
			first_output = output
		else:
			output = controls.match_color(output, first_output, cfg.color_fix)
		output = chains["output"](output, **position)

		work.save(index, output)
		work.commit(index + 1)
		if show.active:
			corr, spread = getattr(backend, "correspondence", None) or (None, None)
			corr = None if corr is None else corr * scale
			conf = None if spread is None else confidence(spread, scale)
			flow = compensate.output_flow if feedback is not None else None
			show(input=frame, control=control, flow=flow, vectors=None if flow is None else (prev_input, flow),
				occlusion=regen, noise=noise, reference=prev_output, warped=warped, feedback=feedback, init=init,
				output=output, attention=None if corr is None else (frame, corr, conf),
				attended=None if corr is None else (frame, corr), confidence=conf)
		prev_input, prev_output = frame, output

	return common.finish(work, output_path, fps, audio_from=input_path if info.has_audio else None,
		audio_start=cfg.start, upscale_to=cfg.upscale, upscale_model=cfg.upscale_model)

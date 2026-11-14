"""
Chunked video-to-video with Wan VACE.

The video is processed in windows of `chunk_frames`. Every frame of a window is either
*known* (handed to the model with a black mask: keep as is) or *generated* (white mask:
follow the control signal). Consistency across windows comes from three places:

  overlap   the last `overlap` output frames of one window are the known head of the next
  anchor    one output frame is passed to every window as a reference image
  warp      (experimental) the frames just after the overlap are pre-filled with the last
            known frame carried along the input's optical flow; only areas the flow cannot
            explain are left for the model to generate
"""

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from . import common, control as controls, motion, video

DEFAULT_NEGATIVE = (
	"blurred details, subtitles, static, overall gray, worst quality, low quality, JPEG compression residue, "
	"ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
	"misshapen limbs, fused fingers, messy background")


@dataclass
class VaceConfig:
	prompt: str
	negative_prompt: str = DEFAULT_NEGATIVE
	model: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
	control: str = "depth"
	control_scale: float = 1.0
	max_area: int = 832 * 480      # pixels per frame the model works at; upscale afterwards
	fps: float = 16.0              # Wan generates 16 fps motion; input is resampled to this
	chunk_frames: int = 81         # must be 4k+1
	overlap: int = 9               # 4k+1 lines up with the VAE's temporal blocks
	steps: int = 30
	guidance: float = 5.0
	flow_shift: float = 3.0        # 3 for 480p, 5 for 720p
	seed: int | None = None
	anchor: str = "first"          # "first" (first output frame), "none", or an image path
	warp_frames: int = 0
	motion_estimator: str = "dis"
	color_fix: float = 0.0
	lora: list = field(default_factory=list)
	start: float | None = None
	duration: float | None = None
	upscale: int | None = None     # short side of the final video, e.g. 720 or 1080
	upscale_model: str | None = None
	restart: bool = False

	def __post_init__(self):
		if (self.chunk_frames - 1) % 4 or self.chunk_frames < 5:
			raise ValueError(f"chunk_frames must be 4k+1 (5, 9, ..., 81), got {self.chunk_frames}")
		if not 0 <= self.overlap < self.chunk_frames:
			raise ValueError("overlap must be >= 0 and smaller than chunk_frames")
		if self.warp_frames and not self.overlap:
			raise ValueError("warp_frames needs a non-zero overlap")


def iter_chunks(frames, chunk_frames, overlap, skip=0):
	''' split a frame stream into windows. Yields (start, new_frames, previous_frame): the
	index of the first new frame, the new frames themselves (a full window for the very first
	chunk, `chunk_frames - overlap` afterwards, fewer at the end), and the frame just before
	them. `skip` frames are consumed first, for resuming. '''
	it = iter(frames)
	previous = None
	for _ in range(skip):
		previous = next(it, None)
		if previous is None:
			return
	index = skip
	while True:
		new = list(itertools.islice(it, chunk_frames if index == 0 else chunk_frames - overlap))
		if not new:
			return
		yield index, new, previous
		index += len(new)
		previous = new[-1]


def padded_length(n):
	''' smallest 4k+1 that holds n frames (the Wan VAE packs 4 frames per latent after the first). '''
	return max(5, 4 * ((n - 1 + 3) // 4) + 1)


def load_pipeline(cfg, device=None):
	''' load Wan VACE sized for the GPU at hand and encode the prompts. The UMT5 text encoder
	(11 GB) is only needed for that, so it is dropped again before the rest moves to the GPU.
	Returns (pipe, prompt_embeds, negative_prompt_embeds). '''
	import torch
	from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
	from transformers import UMT5EncoderModel

	device = device or common.pick_device()
	dtype = common.pick_dtype(device)
	on_gpu = device.startswith("cuda")
	text_encoder = UMT5EncoderModel.from_pretrained(
		cfg.model, subfolder="text_encoder", torch_dtype=torch.bfloat16 if on_gpu else torch.float32,
		**({"device_map": device} if on_gpu else {}))
	vae = AutoencoderKLWan.from_pretrained(cfg.model, subfolder="vae", torch_dtype=torch.float32)
	pipe = WanVACEPipeline.from_pretrained(cfg.model, vae=vae, text_encoder=text_encoder, torch_dtype=dtype)
	pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=cfg.flow_shift)
	embeds = encode_prompts(pipe, cfg, device)
	del text_encoder
	common.free_memory()

	common.load_loras(pipe, cfg.lora)
	vram = common.cuda_vram_gb()
	if on_gpu and vram < 12:
		pipe.enable_model_cpu_offload()
	else:
		pipe.to(device)
	if vram < 30 and hasattr(pipe.vae, "enable_tiling"):
		pipe.vae.enable_tiling()
	return (pipe, *embeds)


def encode_prompts(pipe, cfg, device):
	''' encode the prompts, then detach the text encoder from the pipeline. '''
	import torch
	with torch.no_grad():
		prompt_embeds, negative_embeds = pipe.encode_prompt(
			prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
			do_classifier_free_guidance=cfg.guidance > 1.0, device=device)
	pipe.text_encoder = None
	return prompt_embeds, negative_embeds


def _warp_fill(cfg, estimator, last_input, last_output, new_inputs, control_frames):
	''' experimental: pre-fill the first frames after the overlap with the last known output
	carried along the input's motion. Returns (frames, masks) for those positions. '''
	frames, masks = [], []
	for j in range(min(cfg.warp_frames, len(new_inputs))):
		flow = estimator(new_inputs[j], last_input)
		occ = motion.occlusion_mask(flow, estimator(last_input, new_inputs[j]))
		# the model sees masks on an 8x8 grid, so grow the unexplained areas a little
		occ = cv2.dilate(occ, np.ones((9, 9), np.uint8))
		warped = motion.warp(last_output, flow)
		keep = occ[..., None] < 0.5
		frames.append(np.where(keep, warped, control_frames[j]))
		masks.append((occ * 255).astype(np.uint8))
	return frames, masks


def run(input_path, output_path, cfg, pipe=None, embeds=None, control_fn=None):
	''' restyle `input_path` into `output_path`. `pipe`/`embeds` may be passed in to reuse
	loaded models across runs (and by the tests); otherwise they are loaded here. '''
	import torch

	info = video.probe(input_path)
	size = video.fit_size(info.width, info.height, max_area=cfg.max_area, multiple=16)
	width, height = size
	work = video.Workdir(output_path, f"vace {input_path} {width}x{height}@{cfg.fps}", restart=cfg.restart)
	print(f"vace: {info.width}x{info.height} -> {width}x{height} at {cfg.fps} fps, resuming at frame {work.done}")

	if pipe is None:
		pipe, *embeds = load_pipeline(cfg)
	prompt_embeds, negative_embeds = embeds
	control_fn = control_fn or controls.create_control(cfg.control)
	estimator = motion.create_estimator(cfg.motion_estimator) if cfg.warp_frames else None

	anchor = None
	if cfg.anchor not in ("first", "none"):
		anchor = np.asarray(Image.open(cfg.anchor).convert("RGB"))
	elif cfg.anchor == "first":
		anchor = work.load_image("anchor.png")

	total = int((cfg.duration or info.duration) * cfg.fps)
	frames_in = video.read_frames(input_path, size, fps=cfg.fps, start=cfg.start, duration=cfg.duration)
	progress = tqdm(total=total, initial=work.done, unit="frame")
	for start, new_inputs, last_input in iter_chunks(frames_in, cfg.chunk_frames, cfg.overlap, skip=work.done):
		known = [work.load(i) for i in range(max(0, start - cfg.overlap), start)]
		control_frames = control_fn(new_inputs)

		frames = known + control_frames
		black, white = np.zeros((height, width), np.uint8), np.full((height, width), 255, np.uint8)
		masks = [black] * len(known) + [white] * len(control_frames)
		if estimator is not None and known:
			filled, filled_masks = _warp_fill(cfg, estimator, last_input, known[-1], new_inputs, control_frames)
			frames[len(known):len(known) + len(filled)] = filled
			masks[len(known):len(known) + len(filled)] = filled_masks

		# the model wants 4k+1 frames; repeat the last control frame to fill up, trim afterwards
		real = len(frames)
		pad = padded_length(real) - real
		frames += [frames[-1]] * pad
		masks += [white] * pad

		generator = None
		if cfg.seed is not None:
			generator = torch.Generator("cpu").manual_seed(cfg.seed + start)
		result = pipe(
			video=[Image.fromarray(f) for f in frames],
			mask=[Image.fromarray(m) for m in masks],
			reference_images=[Image.fromarray(anchor)] if anchor is not None else None,
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_embeds,
			conditioning_scale=cfg.control_scale,
			height=height, width=width, num_frames=len(frames),
			num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
			generator=generator, output_type="np",
		).frames[0]
		outputs = (np.clip(result[len(known):real], 0, 1) * 255).round().astype(np.uint8)

		if cfg.anchor == "first" and anchor is None:
			anchor = outputs[0]
			work.save_image("anchor.png", anchor)
		if cfg.color_fix > 0:
			reference = anchor if anchor is not None else (work.load(0) if start else outputs[0])
			outputs = [controls.match_color(frame, reference, cfg.color_fix) for frame in outputs]
		for offset, frame in enumerate(outputs):
			work.save(start + offset, frame)
		work.commit(start + len(outputs))
		progress.update(len(outputs))
	progress.close()

	return common.finish(work, output_path, cfg.fps, audio_from=input_path if info.has_audio else None,
		audio_start=cfg.start, upscale_to=cfg.upscale, upscale_model=cfg.upscale_model)

"""
Control-signal extraction: turns input frames into the structural guide the diffusion model
follows (depth map, edges, pose, ...). Extractors are callables taking and returning a list of
RGB uint8 frames, and may keep state between calls so the signal is stable over time.

Every control belongs to a *family* (depth, normal, softedge, lines, pose, gray), which is what
a backend uses to pick a ControlNet or a union-model mode. The controlnet_aux detectors (hed,
pidi, lineart, lineart_anime, mlsd, pose, normalbae) are loaded only when asked for and need
`pip install controlnet_aux`.
"""

import inspect

import cv2
import numpy as np


class DepthControl:
	''' monocular depth, near = bright. Depth Anything V2 by default; any transformers depth
	model works (ZoeDepth, DPT/MiDaS). Models that predict metric depth rather than disparity
	(Zoe) are inverted. The value range is smoothed over time rather than normalised per frame,
	which would make the depth map pulse. '''

	def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf", device=None, batch_size=8, range_smoothing=0.9, invert=None):
		import torch
		from transformers import AutoImageProcessor, AutoModelForDepthEstimation
		self.torch = torch
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.processor = AutoImageProcessor.from_pretrained(model)
		self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
		self.invert = ("zoe" in model.lower()) if invert is None else invert
		self.batch_size = batch_size
		self.range_smoothing = range_smoothing
		self.lo = self.hi = None

	def maps(self, frames):
		''' float depth maps in 0..1 (near = 1), at the frames' size. '''
		torch = self.torch
		out = []
		for i in range(0, len(frames), self.batch_size):
			batch = list(frames[i:i + self.batch_size])
			inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
			with torch.no_grad():
				result = self.model(**inputs)
			sizes = [f.shape[:2] for f in batch]
			post = self.processor.post_process_depth_estimation
			extra = {"source_sizes": sizes} if "source_sizes" in inspect.signature(post).parameters else {}
			for d in post(result, target_sizes=sizes, **extra):
				d = d["predicted_depth"].float().cpu().numpy()
				d = -d if self.invert else d
				lo, hi = float(d.min()), float(d.max())
				if self.lo is None:
					self.lo, self.hi = lo, hi
				else:
					k = self.range_smoothing
					self.lo, self.hi = self.lo * k + lo * (1 - k), self.hi * k + hi * (1 - k)
				out.append(np.clip((d - self.lo) / max(self.hi - self.lo, 1e-6), 0, 1))
		return out

	def __call__(self, frames):
		return [np.repeat((d * 255).astype(np.uint8)[..., None], 3, axis=2) for d in self.maps(frames)]


class NormalFromDepth:
	''' surface normals from the depth map's slope, in the RGB convention the normal
	ControlNets use (x right = red, y up = green, toward the camera = blue). '''

	def __init__(self, strength=4.0, **depth_args):
		self.depth = DepthControl(**depth_args)
		self.strength = strength

	def __call__(self, frames):
		out = []
		for d in self.depth.maps(frames):
			d = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 1.5) * self.strength * max(d.shape) / 256
			n = np.dstack([-cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3), np.ones_like(d)])
			n /= np.linalg.norm(n, axis=2, keepdims=True)
			out.append(((n + 1) * 127.5).astype(np.uint8))
		return out


class AuxControl:
	''' a controlnet_aux detector, run frame by frame at the frame's own resolution. '''

	def __init__(self, name, **call_args):
		import controlnet_aux
		cls, repo = AUX[name]
		cls = getattr(controlnet_aux, cls)
		self.detect = cls.from_pretrained(repo) if repo else cls()
		self.call_args = call_args

	def __call__(self, frames):
		out = []
		for f in frames:
			h, w = f.shape[:2]
			image = self.detect(f, detect_resolution=min(h, w), image_resolution=min(h, w), output_type="np", **self.call_args)
			out.append(cv2.resize(np.asarray(image)[..., :3], (w, h), interpolation=cv2.INTER_AREA))
		return out


AUX = {
	"hed": ("HEDdetector", "lllyasviel/Annotators"),
	"pidi": ("PidiNetDetector", "lllyasviel/Annotators"),
	"lineart": ("LineartDetector", "lllyasviel/Annotators"),
	"lineart_anime": ("LineartAnimeDetector", "lllyasviel/Annotators"),
	"mlsd": ("MLSDdetector", "lllyasviel/Annotators"),
	"pose": ("OpenposeDetector", "lllyasviel/Annotators"),
	"normalbae": ("NormalBaeDetector", "lllyasviel/Annotators"),
}


class CannyControl:
	def __init__(self, low=100, high=200):
		self.low, self.high = low, high

	def __call__(self, frames):
		return [np.repeat(cv2.Canny(f, self.low, self.high)[..., None], 3, axis=2) for f in frames]


class GrayControl:
	''' luminance only: keeps structure and shading, lets the model recolour. '''

	def __call__(self, frames):
		return [np.repeat(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2) for f in frames]


class IdentityControl:
	def __call__(self, frames):
		return list(frames)


# name -> (factory, family)
CONTROLS = {
	"depth": (DepthControl, "depth"),
	"depth_large": (lambda: DepthControl("depth-anything/Depth-Anything-V2-Large-hf"), "depth"),
	"zoe": (lambda: DepthControl("Intel/zoedepth-nyu-kitti"), "depth"),
	"midas": (lambda: DepthControl("Intel/dpt-hybrid-midas"), "depth"),
	"normal": (NormalFromDepth, "normal"),
	"normalbae": (lambda: AuxControl("normalbae"), "normal"),
	"canny": (CannyControl, "lines"),
	"lineart": (lambda: AuxControl("lineart", coarse=False), "lines"),
	"lineart_anime": (lambda: AuxControl("lineart_anime"), "lines"),
	"mlsd": (lambda: AuxControl("mlsd"), "lines"),
	"hed": (lambda: AuxControl("hed"), "softedge"),
	"pidi": (lambda: AuxControl("pidi"), "softedge"),
	"pose": (lambda: AuxControl("pose", include_hand=True, include_face=True), "pose"),
	"gray": (GrayControl, "gray"),
	"none": (IdentityControl, None),
}


def family(name):
	if name not in CONTROLS:
		raise ValueError(f"unknown control {name!r}; choices: {', '.join(sorted(CONTROLS))}")
	return CONTROLS[name][1]


def create_control(name, **kwargs):
	family(name)
	return CONTROLS[name][0](**kwargs)


def match_color(frame, reference, amount=1.0):
	''' pull `frame`'s colour statistics (per-channel mean and spread in Lab) toward
	`reference`. Counters the slow colour drift that feedback loops and chunk hand-offs build up. '''
	if amount <= 0:
		return frame
	src = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
	ref = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
	s_mean, s_std = src.mean(axis=(0, 1)), src.std(axis=(0, 1)) + 1e-6
	r_mean, r_std = ref.mean(axis=(0, 1)), ref.std(axis=(0, 1))
	matched = (src - s_mean) * (r_std / s_std) + r_mean
	matched = cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
	return cv2.addWeighted(matched, amount, frame, 1.0 - amount, 0)

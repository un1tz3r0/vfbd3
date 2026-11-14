"""
Control-signal extraction: turns input frames into the structural guide the diffusion model
follows (depth map, edges, ...). Extractors are callables taking and returning a list of
RGB uint8 frames, and may keep state between calls so the signal is stable over time.
"""

import cv2
import numpy as np


class DepthControl:
	''' monocular depth (Depth Anything V2), near = bright. The value range is smoothed over
	time rather than normalised per frame, which would make the depth map pulse. '''

	def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf", device=None, batch_size=8, range_smoothing=0.9):
		import torch
		from transformers import AutoImageProcessor, AutoModelForDepthEstimation
		self.torch = torch
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.processor = AutoImageProcessor.from_pretrained(model)
		self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
		self.batch_size = batch_size
		self.range_smoothing = range_smoothing
		self.lo = self.hi = None

	def __call__(self, frames):
		torch = self.torch
		out = []
		for i in range(0, len(frames), self.batch_size):
			batch = frames[i:i + self.batch_size]
			inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
			with torch.no_grad():
				depth = self.model(**inputs).predicted_depth
			h, w = batch[0].shape[:2]
			depth = torch.nn.functional.interpolate(depth[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)
			for d in depth[:, 0].cpu().numpy():
				lo, hi = float(d.min()), float(d.max())
				if self.lo is None:
					self.lo, self.hi = lo, hi
				else:
					k = self.range_smoothing
					self.lo, self.hi = self.lo * k + lo * (1 - k), self.hi * k + hi * (1 - k)
				d = np.clip((d - self.lo) / max(self.hi - self.lo, 1e-6), 0, 1)
				out.append(np.repeat((d * 255).astype(np.uint8)[..., None], 3, axis=2))
		return out


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


CONTROLS = {
	"depth": DepthControl,
	"canny": CannyControl,
	"gray": GrayControl,
	"none": IdentityControl,
}


def create_control(name, **kwargs):
	if name not in CONTROLS:
		raise ValueError(f"unknown control {name!r}; choices: {', '.join(sorted(CONTROLS))}")
	return CONTROLS[name](**kwargs)


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

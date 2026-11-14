"""
Frame super-resolution through spandrel, which loads most published upscaler checkpoints
(Real-ESRGAN, SwinIR, DAT, ...). Large frames are processed as overlapping tiles.
"""

import pathlib

import cv2
import numpy as np

DEFAULT_MODEL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


def _fetch(model):
	''' `model` is a local checkpoint path or a URL (downloaded once into the torch hub cache). '''
	import torch
	if not str(model).startswith(("http://", "https://")):
		return str(model)
	target = pathlib.Path(torch.hub.get_dir()) / "vfbd3" / str(model).rsplit("/", 1)[-1]
	if not target.exists():
		target.parent.mkdir(parents=True, exist_ok=True)
		torch.hub.download_url_to_file(model, str(target))
	return str(target)


class Upscaler:
	def __init__(self, model=DEFAULT_MODEL, device=None, tile=512, tile_overlap=32):
		import torch
		from spandrel import ModelLoader
		self.torch = torch
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.net = ModelLoader().load_from_file(_fetch(model)).to(self.device).eval()
		self.scale = self.net.scale
		self.tile = tile
		self.tile_overlap = tile_overlap

	def _run(self, tile):
		torch = self.torch
		t = torch.from_numpy(tile).to(self.device).permute(2, 0, 1)[None].float() / 255.0
		with torch.no_grad():
			out = self.net(t)
		return out[0].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()

	def enlarge(self, frame):
		''' run the model over the frame, feathering overlapping tiles together. '''
		h, w = frame.shape[:2]
		if max(h, w) <= self.tile:
			return (self._run(frame) * 255).round().astype(np.uint8)
		s, step = self.scale, self.tile - self.tile_overlap
		acc = np.zeros((h * s, w * s, 3), np.float32)
		weight = np.zeros((h * s, w * s, 1), np.float32)
		for y in range(0, max(h - self.tile_overlap, 1), step):
			for x in range(0, max(w - self.tile_overlap, 1), step):
				y0, x0 = min(y, max(h - self.tile, 0)), min(x, max(w - self.tile, 0))
				out = self._run(frame[y0:y0 + self.tile, x0:x0 + self.tile])
				th, tw = out.shape[:2]
				# weight falls off toward tile edges so seams average out
				ramp_y = np.minimum(np.arange(th) + 1, np.arange(th)[::-1] + 1)
				ramp_x = np.minimum(np.arange(tw) + 1, np.arange(tw)[::-1] + 1)
				wgt = np.minimum.outer(ramp_y, ramp_x).clip(max=self.tile_overlap * s).astype(np.float32)[..., None]
				acc[y0 * s:y0 * s + th, x0 * s:x0 * s + tw] += out * wgt
				weight[y0 * s:y0 * s + th, x0 * s:x0 * s + tw] += wgt
		return (acc / weight * 255).round().astype(np.uint8)

	def __call__(self, frame, short_side=None):
		''' upscale, repeating the model until the short side reaches `short_side`, then
		resize down to it exactly. Without `short_side`, one pass at the model's native scale. '''
		out = self.enlarge(frame)
		if not short_side:
			return out
		while min(out.shape[:2]) < short_side:
			out = self.enlarge(out)
		factor = short_side / min(out.shape[:2])
		w, h = (int(round(d * factor / 2)) * 2 for d in (out.shape[1], out.shape[0]))
		return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)

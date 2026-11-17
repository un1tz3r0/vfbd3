"""
Live preview: an image rewritten after every frame, either the output alone or a labelled
mosaic of the loop's intermediate images. The file is replaced atomically (written under a
temporary name in the same folder, then renamed over the old one), so an auto-refreshing
viewer never reads a half-written file.
"""

import math
import os
import pathlib
import secrets

import cv2
import numpy as np

from . import motion

PANELS = ("input", "control", "flow", "occlusion", "noise", "feedback", "init", "output", "confidence")


def write_atomic(path, image):
	path = pathlib.Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(f".{path.stem}.{secrets.token_hex(4)}{path.suffix}")
	try:
		if not cv2.imwrite(str(tmp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
			raise OSError(f"could not write {tmp}")
		os.replace(tmp, path)
	finally:
		tmp.unlink(missing_ok=True)


def render(name, value, size):
	''' turn one intermediate into an RGB uint8 image of `size` (w, h). '''
	if value is None:
		image = np.zeros((size[1], size[0], 3), np.uint8)
	elif name == "flow":
		image = motion.flow_to_image(value)
	elif name == "noise":
		# first three latent channels as colour, +-2 sigma mapped to 0..255
		image = np.clip(value[:3].transpose(1, 2, 0) * 64 + 128, 0, 255).astype(np.uint8)
	elif value.ndim == 2:
		image = np.repeat((np.clip(value, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
	else:
		image = value
	if image.shape[:2] != (size[1], size[0]):
		image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if name == "noise" else cv2.INTER_AREA)
	return image


def mosaic(panels, names, max_width=1920):
	''' tile the named panels in a near-square grid, each labelled. `panels` maps names to
	images, masks (h, w floats), flow fields or latent noise. '''
	h, w = panels["output"].shape[:2]
	cols = math.ceil(math.sqrt(len(names)))
	rows = math.ceil(len(names) / cols)
	scale = min(1.0, max_width / (cols * w))
	cw, ch = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
	sheet = np.zeros((rows * ch, cols * cw, 3), np.uint8)
	for i, name in enumerate(names):
		tile = render(name, panels.get(name), (cw, ch)).copy()
		size = min(0.6, max(0.3, cw / 640))
		for color, weight in (((0, 0, 0), 3), ((255, 255, 255), 1)):
			cv2.putText(tile, name, (6, int(6 + 26 * size)), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)
		r, c = divmod(i, cols)
		sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = tile
	return sheet


class Preview:
	''' `path`: file to keep rewriting (None: don't write). `panels`: comma-separated names
	from PANELS, or "all"; just "output" writes the frame itself. `callback(image)` also gets
	each preview, e.g. to show it in a notebook. '''

	def __init__(self, path=None, panels="output", callback=None):
		names = PANELS if panels == "all" else tuple(p.strip() for p in panels.split(",") if p.strip())
		unknown = set(names) - set(PANELS)
		if unknown or not names:
			raise ValueError(f"unknown preview panels {sorted(unknown)}; choices: {', '.join(PANELS)}, all")
		self.path, self.names, self.callback = path, names, callback

	@property
	def active(self):
		return bool(self.path or self.callback)

	def __call__(self, **panels):
		if not self.active:
			return
		image = panels["output"] if self.names == ("output",) else mosaic(panels, self.names)
		if self.path:
			write_atomic(self.path, image)
		if self.callback:
			self.callback(image)

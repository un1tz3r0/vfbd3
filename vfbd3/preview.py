"""
Live preview: an image rewritten after every frame, either the output alone or a labelled
mosaic of the loop's intermediate images. The file is replaced atomically (written under a
temporary name in the same folder, then renamed over the old one), so an auto-refreshing
viewer never reads a half-written file.

Motion and correspondence can also be shown as a grid of arrows over the two images they
relate, blended half and half: each arrow starts where a point was in the source image and
ends where it is in the destination, so where they agree the arrow heads sit on the matching
features. Arrow colour gives the direction, on the same wheel as the dense `flow` panel.
"""

import math
import os
import pathlib
import secrets
from dataclasses import dataclass

import cv2
import numpy as np

from . import motion

PANELS = ("input", "control", "flow", "flow_vectors", "occlusion", "noise", "feedback", "init", "output",
	"correspondence", "confidence", "attention")

# colours of the attention panel's shares
SHARE_COLORS = {"self": (70, 110, 255), "previous": (255, 120, 40), "input": (60, 220, 90)}


@dataclass
class Vectors:
	''' an arrow field between two images. `start`: (gh, gw, 2) x, y positions in `source`
	pixels, one per cell of a gh x gw grid laid over `dest`; each arrow runs from there to its
	cell's centre. `weight`: (gh, gw) 0..1, how much to trust each arrow (dims the weak ones). '''
	source: np.ndarray
	dest: np.ndarray
	start: np.ndarray
	weight: np.ndarray | None = None

	@classmethod
	def from_flow(cls, source, dest, flow):
		''' `flow` is backward, as motion.warp takes it: dest pixel p came from p + flow(p). '''
		h, w = flow.shape[:2]
		xx, yy = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
		return cls(source, dest, flow + np.stack([xx, yy], -1))


@dataclass
class Shares:
	''' `shares`: (h, w, k) fractions of attention at each location, summing to 1, of the
	`names` (keys of SHARE_COLORS); `background`: image the colours are laid over. '''
	shares: np.ndarray
	names: tuple
	background: np.ndarray | None = None


def _label(image, lines, origin, size):
	x, y = origin
	for text, color in lines:
		for c, weight in (((0, 0, 0), 3), (color, 1)):
			cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, c, weight, cv2.LINE_AA)
		y += int(22 * size) + 4


def draw_vectors(value, size, spacing=24):
	''' source and dest blended 50/50 at `size` (w, h), with arrows at least `spacing` pixels
	apart: one per grid cell when the grid is coarse, sampled when it is dense. '''
	tw, th = size
	fit = lambda im: cv2.resize(im, size, interpolation=cv2.INTER_AREA)
	image = cv2.addWeighted(fit(value.source), 0.5, fit(value.dest), 0.5, 0)
	sh, sw = value.source.shape[:2]
	dh, dw = value.dest.shape[:2]
	gh, gw = value.start.shape[:2]
	stride = max(1, math.ceil(spacing / min(tw / gw, th / gh)))
	rows, cols = np.arange(stride // 2, gh, stride), np.arange(stride // 2, gw, stride)
	start = value.start[np.ix_(rows, cols)].reshape(-1, 2) * (tw / sw, th / sh)
	ey, ex = np.meshgrid((rows + 0.5) * th / gh, (cols + 0.5) * tw / gw, indexing="ij")
	end = np.stack([ex, ey], -1).reshape(-1, 2)
	weight = np.ones(len(end)) if value.weight is None else np.clip(value.weight[np.ix_(rows, cols)].reshape(-1), 0, 1)
	d = (end - start).astype(np.float32)
	hsv = np.zeros((1, len(d), 3), np.uint8)
	hsv[0, :, 0] = (np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)) * 90 / np.pi
	hsv[0, :, 1:] = 255
	colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(float) * (0.55 + 0.45 * weight[:, None])
	for (x0, y0), (x1, y1), color, length in zip(start, end, colors, np.hypot(d[:, 0], d[:, 1])):
		a, b = (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
		for c, t in (((0, 0, 0), 3), (tuple(int(v) for v in color), 1)):
			if length < 1.5:
				cv2.circle(image, b, max(1, t // 2 + 1), c, -1, cv2.LINE_AA)
			else:
				cv2.arrowedLine(image, a, b, c, t, cv2.LINE_AA, tipLength=min(0.4, 5 / length))
	return image


def draw_shares(value, size):
	''' each share's colour, mixed by share, over a dimmed grey copy of the background. '''
	colors = np.array([SHARE_COLORS.get(n, (200, 200, 200)) for n in value.names], np.float32)
	mix = cv2.resize(value.shares.astype(np.float32), size, interpolation=cv2.INTER_NEAREST).reshape(size[1], size[0], -1) @ colors
	if value.background is not None:
		gray = cv2.cvtColor(cv2.resize(value.background, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
		mix = mix * 0.75 + gray[..., None].astype(np.float32) * 0.25
	image = np.clip(mix, 0, 255).astype(np.uint8)
	text = min(0.5, max(0.3, size[0] / 800))
	mean = value.shares.reshape(-1, len(value.names)).mean(0)
	lines = [(f"{n} {m:.0%}", SHARE_COLORS.get(n, (200, 200, 200))) for n, m in zip(value.names, mean)]
	_label(image, lines, (6, size[1] - (int(22 * text) + 4) * len(lines) + int(14 * text)), text)
	return image


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
	elif isinstance(value, Vectors):
		return draw_vectors(value, size)
	elif isinstance(value, Shares):
		return draw_shares(value, size)
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
	images, masks (h, w floats), flow fields, latent noise, Vectors or Shares. '''
	h, w = panels["output"].shape[:2]
	cols = math.ceil(math.sqrt(len(names)))
	rows = math.ceil(len(names) / cols)
	scale = min(1.0, max_width / (cols * w))
	cw, ch = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
	sheet = np.zeros((rows * ch, cols * cw, 3), np.uint8)
	for i, name in enumerate(names):
		tile = render(name, panels.get(name), (cw, ch)).copy()
		size = min(0.6, max(0.3, cw / 640))
		_label(tile, [(name, (255, 255, 255))], (6, int(6 + 26 * size)), size)
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

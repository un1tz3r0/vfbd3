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

PANELS = ("input", "control", "flow", "vectors", "flow_vectors", "occlusion", "noise", "reference", "warped",
	"feedback", "init", "output", "attention", "attended", "correspondence", "confidence", "attention_split")
ATTENTION = ("attention", "attended", "correspondence", "confidence")

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


def legend(image, lo, hi, unit="", gain=1.0):
	''' a small grey ramp in the lower-right corner, labelled with the range the panel was
	stretched over and, for arrows, how much they were magnified (or shrunk) to show it. '''
	h, w = image.shape[:2]
	bw, bh = max(24, w // 6), max(3, h // 60)
	x1, y1 = w - 6, h - 6
	x0, y0 = x1 - bw, y1 - bh
	image[y0:y1, x0:x1] = np.linspace(0, 255, bw).astype(np.uint8)[None, :, None]
	cv2.rectangle(image, (x0 - 1, y0 - 1), (x1, y1), (0, 0, 0), 1)
	text = f"{lo:.3g}..{hi:.3g}{unit}" + (f" x{gain:.2g}" if abs(gain - 1) > 0.01 else "")
	size = min(0.45, max(0.3, w / 900))
	(tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
	_label(image, [(text, (255, 255, 255))], (x1 - tw, y0 - 5), size)
	return image


def stretch(value):
	''' (value min-max normalised to 0..1, min, max); a flat value is only clipped. '''
	lo, hi = float(value.min()), float(value.max())
	return ((value - lo) / (hi - lo) if hi - lo > 1e-6 else np.clip(value, 0, 1)), lo, hi


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
	# magnify small offsets so the longest (99th percentile) reaches most of an arrow spacing
	top = float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 99)) if len(d) else 0.0
	gain = max(1.0, 0.8 * stride * min(tw / gw, th / gh) / top) if top > 1e-3 else 1.0
	d *= gain
	start = end - d
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
	return legend(image, 0, top * sw / tw, "px", gain)


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
def parse(panels):
	names = PANELS if panels == "all" else tuple(p.strip() for p in (panels or "").split(",") if p.strip())
	unknown = set(names) - set(PANELS)
	if unknown or not names:
		raise ValueError(f"unknown preview panels {sorted(unknown)}; choices: {', '.join(PANELS)}, all")
	return names


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


def quiver(background, tails, vectors, size, fit, shrink=True, at_head=False, saturation=None):
	''' arrows from `tails` along `vectors` ((n, 2), pixels of `background`) over the dimmed
	background, drawn at `size` (w, h). Hue follows arrow direction; brightness, width and head
	size follow length relative to the longest (99th percentile), which is scaled to `fit`
	pixels (only ever up, unless `shrink`). `at_head`: scale about the arrow heads instead of
	the tails. `saturation`: (n,) 0..1, low values draw grey and dimmer. '''
	h, w = background.shape[:2]
	k = np.array([size[0] / w, size[1] / h])
	image = (cv2.resize(background, size, interpolation=cv2.INTER_AREA) * 0.4).astype(np.uint8)
	length = np.hypot(vectors[:, 0], vectors[:, 1])
	top = max(float(np.percentile(length, 99)), 1e-3)
	m = np.clip(length / top, 0, 1)
	hue = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360 / 2
	sat = 1.0 if saturation is None else np.clip(saturation, 0, 1)
	hsv = np.stack(np.broadcast_arrays(hue, 255 * sat, (70 + 185 * m) * (0.4 + 0.6 * sat)), -1).astype(np.uint8)
	colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0].tolist()
	gain = fit / top if shrink else max(1.0, fit / top)
	if at_head:
		tails = tails + vectors * (1 - gain)
	tails, heads = tails * k * 16, (tails + vectors * gain) * k * 16   # 4 bits subpixel
	for p, q, mi, color in zip(np.rint(tails).astype(int).tolist(), np.rint(heads).astype(int).tolist(), m, colors):
		if np.hypot(q[0] - p[0], q[1] - p[1]) < 32:
			cv2.circle(image, p, 16, color, -1, cv2.LINE_AA, shift=4)
		else:
			cv2.arrowedLine(image, p, q, color, 1 + round(2 * mi), cv2.LINE_AA, shift=4, tipLength=0.2 + 0.2 * mi)
	return legend(image, 0, top, "px", gain)


def grid_centres(h, w, gh, gw):
	ys, xs = np.mgrid[0:gh, 0:gw]
	return np.stack([(xs + 0.5) * w / gw, (ys + 0.5) * h / gh], -1).reshape(-1, 2)


def flow_vectors(background, flow, size, grid=32):
	''' the motion the image was moved by, averaged over `grid`-pixel cells; arrows point the
	way content travels (the negated backward flow) and are scaled so the longest spans a cell. '''
	h, w = flow.shape[:2]
	gh, gw = max(1, h // grid), max(1, w // grid)
	cells = cv2.resize(flow, (gw, gh), interpolation=cv2.INTER_AREA).reshape(-1, 2)
	return quiver(background, grid_centres(h, w, gh, gw), -cells, size, fit=0.9 * min(h / gh, w / gw))


def attention_vectors(background, correspondence, confidence, size):
	''' one arrow per coarse attention token, from where in `background` it drew from to the
	token itself, shortened and greyed by confidence the way flow remap discounts it (falling back
	to plain flow where attention was diffuse), and magnified when the offsets are small. '''
	h, w = background.shape[:2]
	gh, gw = correspondence.shape[:2]
	conf = confidence.reshape(-1, 1)
	heads = grid_centres(h, w, gh, gw)
	vectors = (heads - correspondence.reshape(-1, 2)) * conf
	return quiver(background, heads - vectors, vectors, size, fit=0.9 * min(h / gh, w / gw),
		shrink=False, at_head=True, saturation=conf[:, 0])


def render(name, value, size):
	''' turn one intermediate into an RGB uint8 image of `size` (w, h). Flow, noise and masks
	are stretched to their own range each frame, shown by a legend in the corner. '''
	scale = None
	if value is None:
		image = np.zeros((size[1], size[0], 3), np.uint8)
	elif isinstance(value, Vectors):
		return draw_vectors(value, size)
	elif isinstance(value, Shares):
		return draw_shares(value, size)
	elif name == "vectors":
		return flow_vectors(*value, size)
	elif name == "attention":
		return attention_vectors(*value, size)
	elif name == "attended":
		# what each output location drew from: the input sampled at the correspondence
		frame, correspondence = value
		corr = cv2.resize(correspondence.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
		image = cv2.remap(frame, corr[..., 0] - 0.5, corr[..., 1] - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	elif name == "flow":
		image = motion.flow_to_image(value)   # brightness is already min-max of the magnitude
		magnitude = np.hypot(value[..., 0], value[..., 1])
		scale = (float(magnitude.min()), float(magnitude.max()), "px")
	elif name == "noise":
		# first three latent channels as colour
		image, lo, hi = stretch(value[:3].transpose(1, 2, 0))
		image, scale = (image * 255).astype(np.uint8), (lo, hi, "sd")
	elif value.ndim == 2:
		image, lo, hi = stretch(value.astype(np.float32))
		image, scale = np.repeat((image * 255).astype(np.uint8)[..., None], 3, axis=2), (lo, hi, "")
	else:
		image = value
	if image.shape[:2] != (size[1], size[0]):
		image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if name == "noise" else cv2.INTER_AREA)
	return image if scale is None else legend(np.ascontiguousarray(image), *scale)


def mosaic(panels, names, max_width=1920):
	''' tile the named panels in a near-square grid, each labelled. `panels` maps names to
	images, masks (h, w floats), flow fields, latent noise, or the tuples
	that `render` takes for vectors, attention and attended, Vectors or Shares. '''
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
		self.path, self.names, self.callback = path, parse(panels), callback

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

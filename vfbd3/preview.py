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

PANELS = ("input", "control", "flow", "vectors", "occlusion", "noise", "reference", "warped", "feedback", "init",
	"output", "attention", "attended", "confidence")
ATTENTION = ("attention", "attended", "confidence")   # these need the attention correspondence recorded


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


def quiver(background, tails, vectors, size, fit=None, saturation=None):
	''' arrows from `tails` along `vectors` ((n, 2), pixels of `background`) over the dimmed
	background, drawn at `size` (w, h). Hue follows arrow direction;
	brightness, width and head size follow length relative to the longest (99th percentile).
	`fit`: rescale so that length is `fit` pixels, else true length. `saturation`: (n,) 0..1,
	low values draw grey and dimmer. '''
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
	tails, heads = tails * k * 16, (tails + vectors * (fit / top if fit else 1.0)) * k * 16   # 4 bits subpixel
	for p, q, mi, color in zip(np.rint(tails).astype(int).tolist(), np.rint(heads).astype(int).tolist(), m, colors):
		if np.hypot(q[0] - p[0], q[1] - p[1]) < 32:
			cv2.circle(image, p, 16, color, -1, cv2.LINE_AA, shift=4)
		else:
			cv2.arrowedLine(image, p, q, color, 1 + round(2 * mi), cv2.LINE_AA, shift=4, tipLength=0.2 + 0.2 * mi)
	label = f"{top:.1f}px" if fit else f"max {top:.0f}px"
	for color, weight in (((0, 0, 0), 3), ((255, 255, 255), 1)):
		cv2.putText(image, label, (6, size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, weight, cv2.LINE_AA)
	return image


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
	to plain flow where attention was diffuse). '''
	h, w = background.shape[:2]
	conf = confidence.reshape(-1, 1)
	heads = grid_centres(h, w, *correspondence.shape[:2])
	vectors = (heads - correspondence.reshape(-1, 2)) * conf
	return quiver(background, heads - vectors, vectors, size, saturation=conf[:, 0])


def render(name, value, size):
	''' turn one intermediate into an RGB uint8 image of `size` (w, h). '''
	if value is None:
		image = np.zeros((size[1], size[0], 3), np.uint8)
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
	images, masks (h, w floats), flow fields, latent noise, or the tuples
	that `render` takes for vectors, attention and attended. '''
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

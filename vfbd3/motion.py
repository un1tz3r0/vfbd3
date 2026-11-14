"""
Dense optical flow and motion compensation.

One convention throughout: flow(a, b)[y, x] = (dx, dy) such that a[y, x] ~ b[y + dy, x + dx].
To carry the previous output forward onto the current frame we therefore estimate
flow(current_input, previous_input) and sample the previous output at the displaced
positions (`warp`).
"""

import cv2
import numpy as np


def _gray(frame):
	return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class DISFlow:
	''' cv2 DIS optical flow: CPU, fast, decent quality. '''

	def __init__(self, preset="medium"):
		presets = {
			"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
			"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
			"medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
		}
		self.dis = cv2.DISOpticalFlow_create(presets[preset])

	def __call__(self, a, b):
		return self.dis.calc(_gray(a), _gray(b), None).astype(np.float32)


class FarnebackFlow:
	def __call__(self, a, b):
		return cv2.calcOpticalFlowFarneback(_gray(a), _gray(b), None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)


class RAFTFlow:
	''' torchvision RAFT (GPU if available). '''

	def __init__(self, small=False, device=None):
		import torch
		from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
		self.torch = torch
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		if small:
			self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
		else:
			self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
		self.model = self.model.to(self.device).eval()

	def _prep(self, frame, size):
		torch = self.torch
		t = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
		return torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)

	def __call__(self, a, b):
		h, w = a.shape[:2]
		# RAFT needs dimensions divisible by 8
		H, W = max(128, -(-h // 8) * 8), max(128, -(-w // 8) * 8)
		with self.torch.no_grad():
			flow = self.model(self._prep(a, (H, W)), self._prep(b, (H, W)))[-1][0]
		flow = flow.permute(1, 2, 0).float().cpu().numpy()
		flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
		flow[..., 0] *= w / W
		flow[..., 1] *= h / H
		return flow


ESTIMATORS = {
	"dis": lambda: DISFlow(),
	"farneback": lambda: FarnebackFlow(),
	"raft": lambda: RAFTFlow(small=False),
	"raft_small": lambda: RAFTFlow(small=True),
}


def create_estimator(name):
	if name not in ESTIMATORS:
		raise ValueError(f"unknown motion estimator {name!r}; choices: {', '.join(sorted(ESTIMATORS))}")
	return ESTIMATORS[name]()


def warp(image, flow, border=cv2.BORDER_REPLICATE):
	''' sample `image` at positions displaced by `flow`: out[y, x] = image[y + dy, x + dx]. '''
	h, w = flow.shape[:2]
	xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
	return cv2.remap(image, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR, borderMode=border)


def occlusion_mask(flow_ab, flow_ba, thresh=1.5):
	''' 1.0 where a pixel of `a` has no consistent match in `b` (forward/backward round trip
	fails). Tolerance grows with flow magnitude (Sundaram et al.) so fast motion alone is not
	flagged; `thresh` is the constant error floor in pixels. '''
	back = warp(flow_ba, flow_ab)
	diff = flow_ab + back
	err2 = (diff * diff).sum(axis=2)
	mag2 = (flow_ab * flow_ab).sum(axis=2) + (back * back).sum(axis=2)
	return (err2 > (0.01 * mag2 + thresh * thresh)).astype(np.float32)


def flow_to_image(flow):
	''' direction -> hue, magnitude -> brightness. '''
	hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = 255
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class MotionCompensator:
	''' Carries an image aligned with the previous input frame onto the current one.

	`alpha` blends each flow field with the previous one (temporal smoothing), `sigma` blurs
	it spatially. With `occlusion`, also returns a mask of pixels the previous frame cannot
	explain (newly revealed areas), so the caller can fill those from somewhere else. '''

	def __init__(self, estimator="dis", alpha=0.0, sigma=0.0, occlusion=False):
		self.estimate = create_estimator(estimator) if isinstance(estimator, str) else estimator
		self.alpha = alpha
		self.sigma = sigma
		self.occlusion = occlusion
		self.flow = None

	def _smooth(self, flow):
		return cv2.GaussianBlur(flow, (0, 0), self.sigma) if self.sigma > 0 else flow

	def __call__(self, prev_input, cur_input, prev_image):
		''' returns (prev_image warped onto cur_input, occlusion mask or None). '''
		flow = self._smooth(self.estimate(cur_input, prev_input))
		occ = None
		if self.occlusion:
			occ = occlusion_mask(flow, self._smooth(self.estimate(prev_input, cur_input)))
		if self.alpha > 0 and self.flow is not None and self.flow.shape == flow.shape:
			keep = self.alpha if occ is None else self.alpha * (1.0 - occ)[..., None]
			flow = self.flow * keep + flow * (1.0 - keep)
		self.flow = flow
		return warp(prev_image, flow), occ

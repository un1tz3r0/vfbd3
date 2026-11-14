import cv2
import numpy as np
import pytest

from vfbd3 import control, motion, video
from vfbd3.vace import iter_chunks, padded_length


def test_probe_and_read(clip):
	info = video.probe(clip)
	assert (info.width, info.height) == (64, 48)
	assert info.fps == pytest.approx(12)
	assert info.has_audio
	frames = list(video.read_frames(clip, (32, 32)))
	assert len(frames) == 24
	assert frames[0].shape == (32, 32, 3) and frames[0].dtype == np.uint8
	assert len(list(video.read_frames(clip, (32, 32), fps=6, start=0.5, duration=1))) == 6


def test_fit_size():
	assert video.fit_size(1920, 1080, short_side=720) == (1280, 720)
	assert video.fit_size(1080, 1920, short_side=720) == (720, 1280)
	w, h = video.fit_size(1920, 1080, max_area=832 * 480)
	assert w % 16 == 0 and h % 16 == 0 and w * h <= 832 * 480
	assert w / h == pytest.approx(16 / 9, rel=0.05)


def test_write_video_keeps_audio(clip, tmp_path):
	out = tmp_path / "out.mp4"
	frames = video.read_frames(clip, (64, 48))
	assert video.write_video(out, frames, 12, audio_from=clip) == 24
	info = video.probe(out)
	assert (info.width, info.height) == (64, 48)
	assert info.has_audio
	assert info.duration == pytest.approx(2.0, abs=0.2)


def test_workdir_resume(tmp_path):
	out = tmp_path / "out.mp4"
	work = video.Workdir(out, "sig")
	frame = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
	work.save(0, frame)
	work.save(1, frame)       # written but never committed: must not count
	work.commit(1)
	again = video.Workdir(out, "sig")
	assert again.done == 1
	assert np.array_equal(again.load(0), frame)
	with pytest.raises(RuntimeError):
		video.Workdir(out, "other settings")
	assert video.Workdir(out, "other settings", restart=True).done == 0


def test_iter_chunks():
	chunks = list(iter_chunks(range(30), chunk_frames=13, overlap=5))
	assert [(start, len(new), prev) for start, new, prev in chunks] == [(0, 13, None), (13, 8, 12), (21, 8, 20), (29, 1, 28)]
	# every frame appears exactly once as a new frame
	assert [f for _, new, _ in chunks for f in new] == list(range(30))
	# resuming after the first chunk yields the same remaining chunks
	assert list(iter_chunks(range(30), 13, 5, skip=13)) == chunks[1:]
	assert list(iter_chunks(range(30), 13, 5, skip=30)) == []


def test_padded_length():
	assert [padded_length(n) for n in (1, 5, 6, 9, 80, 81)] == [5, 5, 9, 9, 81, 81]


def _texture(seed=0, size=(96, 128)):
	rng = np.random.default_rng(seed)
	img = rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
	img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
	return cv2.GaussianBlur(img, (0, 0), 1.5)


@pytest.mark.parametrize("estimator", ["dis", "farneback"])
def test_motion_compensation_follows_a_shift(estimator):
	prev = _texture()
	cur = np.roll(prev, (3, 5), axis=(0, 1))          # content moves 5 px right, 3 px down
	painted = 255 - prev                              # stands in for the previous output
	warped, occluded = motion.MotionCompensator(estimator)(prev, cur, painted)
	assert occluded is None
	expected = np.roll(painted, (3, 5), axis=(0, 1))
	inner = (slice(16, -16), slice(16, -16))
	error = np.abs(warped[inner].astype(int) - expected[inner].astype(int)).mean()
	unwarped = np.abs(painted[inner].astype(int) - expected[inner].astype(int)).mean()
	assert error < unwarped * 0.25


def test_occlusion_mask_flags_inconsistent_flow():
	flow = np.zeros((32, 32, 2), np.float32)
	assert motion.occlusion_mask(flow, flow).sum() == 0
	bad = flow.copy()
	bad[8:16, 8:16] = 6.0                             # backward flow disagrees here
	mask = motion.occlusion_mask(flow, bad)
	assert mask[8:16, 8:16].all() and mask.sum() == 64


def test_controls_and_color_match():
	frame = _texture()
	for name in ("canny", "gray", "none"):
		out = control.create_control(name)([frame, frame])
		assert len(out) == 2 and out[0].shape == frame.shape and out[0].dtype == np.uint8
	tinted = np.clip(frame * np.array([1.0, 0.6, 0.6]), 0, 255).astype(np.uint8)
	fixed = control.match_color(tinted, frame, 1.0)
	assert np.abs(fixed.mean((0, 1)) - frame.mean((0, 1))).max() < np.abs(tinted.mean((0, 1)) - frame.mean((0, 1))).max() / 3
	assert np.array_equal(control.match_color(tinted, frame, 0.0), tinted)

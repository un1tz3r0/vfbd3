import cv2
import numpy as np
import pytest

from vfbd3 import control, frame, motion, vace, video
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


def test_noise_field_moves_with_the_flow_and_stays_white():
	field = motion.NoiseField(64, 96, seed=0)
	before = field.field.copy()
	flow = np.zeros((64, 96, 2), np.float32)
	flow[..., 0] = -5           # content moves 5 px to the right
	field.advance(flow)
	assert np.array_equal(field.field[:, 5:], before[:, :-5])
	assert not np.array_equal(field.field[:, :5], before[:, :5])        # entered the frame: fresh

	# a zoom maps several pixels onto one source; duplicates must be redrawn, not copied
	xx, yy = np.meshgrid(np.arange(96, dtype=np.float32), np.arange(64, dtype=np.float32))
	zoom = np.stack([(xx - 48) * -0.1, (yy - 32) * -0.1], axis=2)
	for _ in range(20):
		field.advance(zoom)
	latent = field.latent()
	assert latent.shape == (4, 8, 12)
	assert abs(field.field.std() - 1) < 0.02 and abs(latent.std() - 1) < 0.15
	rows = field.field[..., 0]
	assert abs(np.corrcoef(rows[:, :-1].ravel(), rows[:, 1:].ravel())[0, 1]) < 0.05


def test_fast_presets_fill_only_unset_fields():
	assert (frame.FrameConfig(prompt="x").steps, frame.FrameConfig(prompt="x").guidance) == (25, 7.5)
	cfg = frame.FrameConfig(prompt="x", fast="dmd2", guidance=2.0, lora=["mine"])
	assert (cfg.steps, cfg.guidance, cfg.lora) == (4, 2.0, [frame.FAST["sdxl"]["dmd2"]["lora"], "mine"])
	cfg = frame.FrameConfig(prompt="x", backend="flux", fast="schnell")
	assert (cfg.steps, cfg.guidance, cfg.model) == (4, 0.0, "black-forest-labs/FLUX.1-schnell") and cfg.transformer.endswith(".gguf")
	assert frame.FrameConfig(prompt="x", backend="sd3").controlnet is None
	with pytest.raises(ValueError):
		frame.FrameConfig(prompt="x", backend="flux", reference=1.0)
	assert frame.FrameConfig(prompt="x", fast="turbo").model == "stabilityai/sdxl-turbo"
	cfg = vace.VaceConfig(prompt="x", fast="selfforcing")
	assert (cfg.steps, cfg.guidance, cfg.flow_shift, len(cfg.lora)) == (4, 1.0, 5.0, 1)
	assert vace.VaceConfig(prompt="x").steps == 30
	with pytest.raises(ValueError):
		vace.VaceConfig(prompt="x", fast="nope")


def test_remap_flow_moves_output_with_its_source():
	forward = np.zeros((40, 80, 2), np.float32)
	forward[:, 40:, 0] = 5                          # the input's right half moves 5 px right
	ys, xs = np.mgrid[0:40, 0:80].astype(np.float32)
	corr = np.dstack([np.minimum(xs + 20, 79), ys])  # the output shows the input 20 px further right
	flow = motion.remap_flow(-forward, forward, corr, np.ones((40, 80), np.float32))
	assert np.allclose(flow[:, 25:35, 0], -5, atol=0.5)     # sourced from x 45..55: moving
	assert np.allclose(flow[:, :15, 0], 0, atol=0.5)        # sourced from x 20..35: still
	none = motion.remap_flow(-forward, forward, corr, np.zeros((40, 80), np.float32))
	assert np.allclose(none[:, 50:, 0], -5, atol=0.5) and np.allclose(none[:, :30, 0], 0, atol=0.5)


def test_colorlock_holds_first_frame_statistics():
	from vfbd3 import filters
	rng = np.random.default_rng(0)
	frame0 = rng.integers(60, 190, (32, 32, 3), dtype=np.uint8)
	chain = filters.parse_video_filter_spec("colorlock amount=1")
	chain(frame0)
	blown = np.clip((frame0.astype(int) - 128) * 2 + 128, 0, 255).astype(np.uint8)
	fixed = chain(blown)
	lab = lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float).std(0)
	assert np.allclose(lab(fixed), lab(frame0), rtol=0.15) and not np.allclose(lab(blown), lab(frame0), rtol=0.15)


def test_preview_writes_atomically_and_tiles(tmp_path):
	from vfbd3 import preview
	out = np.full((48, 64, 3), 200, np.uint8)
	path = tmp_path / "live" / "preview.png"
	shown = []
	show = preview.Preview(str(path), "input,flow,occlusion,noise,output", shown.append)
	show(input=out, flow=np.ones((48, 64, 2), np.float32), occlusion=None, noise=np.zeros((4, 6, 8), np.float32), output=out)
	assert cv2.imread(str(path)).shape == (96, 192, 3) and shown[0].shape == (96, 192, 3)
	assert [p.name for p in path.parent.iterdir()] == ["preview.png"]
	preview.Preview(str(path))(output=out)
	assert cv2.imread(str(path)).shape == (48, 64, 3)
	flow = np.zeros((48, 64, 2), np.float32)
	flow[..., 0] = -3
	corr = np.stack(np.meshgrid(np.arange(4) * 16 + 8.0, np.arange(3) * 16 + 8.0), -1) + 5
	sheet = preview.mosaic(dict(output=out, vectors=(out, flow), attention=(out, corr, np.ones((3, 4))), attended=(out, corr)),
		("vectors", "attention", "attended", "output"))
	assert sheet.shape == (96, 128, 3) and sheet[:48, :64].std() > 0
	with pytest.raises(ValueError):
		preview.Preview(str(path), "output,bogus")

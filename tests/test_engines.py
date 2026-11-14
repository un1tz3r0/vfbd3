import numpy as np
import pytest

from vfbd3 import control, frame, vace, video


class RecordingBackend:
	''' stands in for the diffusion model: returns the init image, slightly brightened, and
	records what the frame loop handed it. '''

	def __init__(self):
		self.calls = []

	def generate(self, init, control_image, denoise, params, reference=None):
		self.calls.append(dict(denoise=denoise, has_feedback=reference is not None, shape=init.shape))
		return np.clip(init.astype(int) + 10, 0, 255).astype(np.uint8)


def test_frame_engine_runs_and_resumes(clip, tmp_path):
	out = tmp_path / "out.mp4"
	cfg = frame.FrameConfig(prompt="x", control="canny", short_side=48, max_area=64 * 48, duration=1.0,
		occlusion=True, color_fix=0.5, output_filter="brightness amount=1.1")
	backend = RecordingBackend()
	frame.run(clip, out, cfg, backend=backend, on_frame=lambda i, t: {"denoise": 0.3} if i == 5 else None)

	assert len(backend.calls) == 12
	assert backend.calls[0] == dict(denoise=1.0, has_feedback=False, shape=(48, 64, 3))
	assert all(c["has_feedback"] and c["denoise"] == 0.5 for i, c in enumerate(backend.calls) if i not in (0, 5))
	assert backend.calls[5]["denoise"] == 0.3
	info = video.probe(out)
	assert (info.width, info.height) == (64, 48) and info.has_audio
	assert len(list(video.read_frames(out, (64, 48)))) == 12

	# same output name, longer duration: the first 12 frames are reused, the rest generated
	more = RecordingBackend()
	cfg.duration = 2.0
	frame.run(clip, out, cfg, backend=more)
	assert len(more.calls) == 12 and all(c["has_feedback"] for c in more.calls)
	assert len(list(video.read_frames(out, (64, 48)))) == 24


def tiny_vace_pipeline():
	''' a randomly initialised Wan VACE pipeline small enough for CPU (same shapes as the
	diffusers test suite uses). Output is noise, but every code path is the real one. '''
	import torch
	from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline, WanVACETransformer3DModel
	from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

	torch.manual_seed(0)
	vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1, temperal_downsample=[False, True, True])
	text_encoder = T5EncoderModel(AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")).eval()
	tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
	transformer = WanVACETransformer3DModel(
		patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
		text_dim=32, freq_dim=256, ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
		rope_max_seq_len=32, vace_layers=[0, 2], vace_in_channels=96)
	return WanVACEPipeline(transformer=transformer, vae=vae, scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0),
		text_encoder=text_encoder, tokenizer=tokenizer)


class SpyPipe:
	''' forwards to the real pipeline, keeping the conditioning each chunk was given. '''

	def __init__(self, pipe):
		self.pipe, self.calls = pipe, []

	def __call__(self, **kwargs):
		self.calls.append(kwargs)
		return self.pipe(**kwargs)


@pytest.mark.slow
def test_vace_engine_chunks_overlap_and_resume(clip, tmp_path):
	out = tmp_path / "out.mp4"
	cfg = vace.VaceConfig(prompt="a mural", control="gray", max_area=32 * 32, fps=12, chunk_frames=13, overlap=5,
		steps=2, seed=1, duration=1.5, warp_frames=2)
	pipe = tiny_vace_pipeline()
	embeds = vace.encode_prompts(pipe, cfg, "cpu")
	assert pipe.text_encoder is None
	spy = SpyPipe(pipe)
	vace.run(clip, out, cfg, pipe=spy, embeds=embeds, control_fn=control.create_control("gray"))

	# 18 frames: 13, then 5 known + 5 new (padded from 10 to 13)
	assert [c["num_frames"] for c in spy.calls] == [13, 13]
	first, second = spy.calls
	assert first["reference_images"] is None and len(second["reference_images"]) == 1
	assert all(np.asarray(m).min() == 255 for m in first["mask"])
	masks = [np.asarray(m) for m in second["mask"]]
	assert all(m.max() == 0 for m in masks[:5])               # known overlap: keep
	assert all(m.min() == 255 for m in masks[7:])             # generated
	work = video.Workdir(out, f"vace {clip} 32x32@12")
	assert work.done == 18
	# the known head of chunk 2 is exactly the saved tail of chunk 1
	assert all(np.array_equal(np.asarray(second["video"][i]), work.load(8 + i)) for i in range(5))
	assert len(list(video.read_frames(out, (32, 32)))) == 18

	# extend the same run: only the new frames are generated, continuing from frame 18
	cfg.duration = 2.0
	spy.calls.clear()
	vace.run(clip, out, cfg, pipe=spy, embeds=embeds, control_fn=control.create_control("gray"))
	assert len(spy.calls) == 1
	assert np.array_equal(np.asarray(spy.calls[0]["video"][4]), work.load(17))
	assert len(list(video.read_frames(out, (32, 32)))) == 24


@pytest.mark.slow
def test_upscaler_tiles_match_scale():
	pytest.importorskip("spandrel")
	from vfbd3.upscale import Upscaler
	up = Upscaler(tile=48, tile_overlap=8, device="cpu")
	image = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
	assert up.enlarge(image).shape == (60 * up.scale, 80 * up.scale, 3)
	assert up(image, short_side=90).shape == (90, 120, 3)

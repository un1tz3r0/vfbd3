import numpy as np
import pytest

from vfbd3 import backends, control, frame, vace, video


class RecordingBackend:
	''' stands in for the diffusion model: returns the init image, slightly brightened, and
	records what the frame loop handed it. '''

	def __init__(self):
		self.calls, self.noise = [], []

	def generate(self, init, control_image, denoise, params, images=None, noise=None, regen=None):
		self.calls.append(dict(denoise=denoise, has_feedback=images["feedback"] is not None, shape=init.shape))
		self.noise.append(noise)
		assert regen is None or regen.shape == init.shape[:2]
		return np.clip(init.astype(int) + 10, 0, 255).astype(np.uint8)


def test_frame_engine_runs_and_resumes(clip, tmp_path):
	out = tmp_path / "out.mp4"
	cfg = frame.FrameConfig(prompt="x", control="canny", short_side=48, max_area=64 * 48, duration=1.0,
		occlusion=True, color_fix=0.5, output_filter="brightness amount=1.1",
		preview=str(tmp_path / "preview.jpg"), preview_panels="all")
	backend = RecordingBackend()
	previews = []
	frame.run(clip, out, cfg, backend=backend, on_frame=lambda i, t: {"denoise": 0.3} if i == 5 else None, on_preview=previews.append)
	assert len(previews) == 12 and previews[-1].shape == (192, 256, 3) and (tmp_path / "preview.jpg").exists()

	assert len(backend.calls) == 12
	assert backend.calls[0] == dict(denoise=1.0, has_feedback=False, shape=(48, 64, 3))
	assert all(c["has_feedback"] and c["denoise"] == 0.5 for i, c in enumerate(backend.calls) if i not in (0, 5))
	assert backend.calls[5]["denoise"] == 0.3
	# the noise field is latent sized and moves with the picture rather than being redrawn
	assert backend.noise[0].shape == (4, 6, 8) and not np.array_equal(backend.noise[0], backend.noise[-1])
	info = video.probe(out)
	assert (info.width, info.height) == (64, 48) and info.has_audio
	assert len(list(video.read_frames(out, (64, 48)))) == 12

	# same output name, longer duration: the first 12 frames are reused, the rest generated
	more = RecordingBackend()
	cfg.duration = 2.0
	frame.run(clip, out, cfg, backend=more)
	assert len(more.calls) == 12 and all(c["has_feedback"] for c in more.calls)
	assert len(list(video.read_frames(out, (64, 48)))) == 24


def tiny_sdxl_pipeline():
	''' a randomly initialised SDXL + ControlNet-Union img2img pipeline small enough for CPU. '''
	import torch
	from diffusers import (AutoencoderKL, ControlNetUnionModel, EulerDiscreteScheduler,
		StableDiffusionXLControlNetUnionImg2ImgPipeline, UNet2DConditionModel)
	from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

	torch.manual_seed(0)
	shared = dict(block_out_channels=(32, 64), layers_per_block=2, in_channels=4, attention_head_dim=(2, 4),
		use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
		transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80, cross_attention_dim=64)
	unet = UNet2DConditionModel(sample_size=32, out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
		up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), **shared)
	controlnet = ControlNetUnionModel(down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
		conditioning_embedding_out_channels=(16, 16, 32, 32), num_trans_channel=32, num_trans_head=2, num_proj_channel=32, **shared)
	vae = AutoencoderKL(block_out_channels=[32, 32, 64, 64], in_channels=3, out_channels=3, latent_channels=4,
		down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4)
	text_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
		num_attention_heads=4, num_hidden_layers=5, pad_token_id=1, vocab_size=1000, hidden_act="gelu", projection_dim=32)
	tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
	scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, steps_offset=1, beta_schedule="scaled_linear")
	return StableDiffusionXLControlNetUnionImg2ImgPipeline(unet=unet, controlnet=controlnet, vae=vae, scheduler=scheduler,
		text_encoder=CLIPTextModel(text_config), tokenizer=tokenizer,
		text_encoder_2=CLIPTextModelWithProjection(text_config), tokenizer_2=tokenizer)


@pytest.mark.slow
@pytest.mark.parametrize("fast", [None, "lcm", "lightning", "turbo"])
def test_sdxl_backend_holds_unmasked_pixels(fast, monkeypatch):
	from vfbd3 import common
	monkeypatch.setattr(common, "load_loras", lambda pipe, specs: None)
	cfg = frame.FrameConfig(prompt="a mural", control="canny", fast=fast, steps=8, seed=3)
	backend = backends.SDXLBackend(cfg, device="cpu", pipe=tiny_sdxl_pipeline())
	assert backend.pipe.text_encoder is None and type(backend.pipe.scheduler).__name__ == (
		frame.FAST["sdxl"][fast]["scheduler"][0] if fast else "EulerDiscreteScheduler")

	rng = np.random.default_rng(0)
	init = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
	noise = rng.standard_normal((4, 6, 8), dtype=np.float32)
	seen = []
	unet = backend.pipe.unet.forward
	monkeypatch.setattr(backend.pipe.unet, "forward", lambda sample, *a, **k: seen.append(sample[-1:].clone()) or unet(sample, *a, **k))

	# no mask: plain img2img, half of the 8 steps
	assert backend.generate(init, init, 0.5, {}, noise=noise).shape == (48, 64, 3)
	assert len(seen) == 4
	plain_start = seen[0]

	# left half masked: all 8 steps run, and the unmasked half enters the fifth step exactly as
	# plain img2img at denoise 0.5 would have started it
	seen.clear()
	regen = np.zeros((48, 64), np.float32)
	regen[:, :32] = 1
	backend.generate(init, init, 0.5, {}, noise=noise, regen=regen)
	assert len(seen) == 8
	assert np.allclose(seen[4][..., 4:], plain_start[..., 4:], atol=1e-4)
	assert not np.allclose(seen[4][..., :4], plain_start[..., :4], atol=1e-2)


def tiny_flow_pipeline(kind):
	''' a randomly initialised Flux or SD3 ControlNet pipeline small enough for CPU (the
	diffusers test-suite shapes). '''
	import torch
	from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlNetModel, FluxControlNetPipeline,
		FluxTransformer2DModel, SD3ControlNetModel, SD3Transformer2DModel, StableDiffusion3ControlNetPipeline)
	from transformers import (AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection,
		CLIPTokenizer, T5EncoderModel)

	clip = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
		num_attention_heads=4, num_hidden_layers=5, pad_token_id=1, vocab_size=1000, hidden_act="gelu", projection_dim=32)
	t5 = lambda: T5EncoderModel(AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")).eval()
	tokenizer, t5_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip"), AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
	torch.manual_seed(0)
	if kind == "flux":
		shape = dict(patch_size=1, in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
			joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=[4, 4, 8])
		vae = AutoencoderKL(block_out_channels=(4, 4, 4, 4), in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
			norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False, shift_factor=0.0609, scaling_factor=1.5035,
			down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4)
		return FluxControlNetPipeline(transformer=FluxTransformer2DModel(**shape), controlnet=FluxControlNetModel(**shape), vae=vae,
			scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True), text_encoder=CLIPTextModel(clip), tokenizer=tokenizer,
			text_encoder_2=t5(), tokenizer_2=t5_tokenizer)
	shape = dict(sample_size=32, patch_size=1, in_channels=8, attention_head_dim=8, num_attention_heads=4, joint_attention_dim=32,
		caption_projection_dim=32, pooled_projection_dim=64, out_channels=8, qk_norm="rms_norm")
	vae = AutoencoderKL(block_out_channels=(4, 4, 4, 4), in_channels=3, out_channels=3, latent_channels=8, layers_per_block=1,
		norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False, shift_factor=0.0609, scaling_factor=1.5035,
		down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4)
	return StableDiffusion3ControlNetPipeline(transformer=SD3Transformer2DModel(num_layers=4, **shape),
		controlnet=SD3ControlNetModel(num_layers=3, **shape), vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
		text_encoder=CLIPTextModelWithProjection(clip), tokenizer=tokenizer, text_encoder_2=CLIPTextModelWithProjection(clip),
		tokenizer_2=tokenizer, text_encoder_3=t5(), tokenizer_3=t5_tokenizer)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["flux", "sd3"])
def test_flow_backend_holds_unmasked_pixels(kind, monkeypatch):
	pipe = tiny_flow_pipeline(kind)
	cfg = frame.FrameConfig(prompt="a mural", backend=kind, control="canny", steps=8, guidance=3.0, seed=3,
		controlnet="tiny" if kind == "sd3" else None)
	backend = backends.create(cfg, device="cpu", pipe=pipe)
	channels = pipe.vae.config.latent_channels
	assert pipe.text_encoder is None and backend.latent_scale == 8
	rng = np.random.default_rng(0)
	init = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
	noise = rng.standard_normal((channels, 8, 8), dtype=np.float32)
	seen = []
	forward = pipe.transformer.forward
	monkeypatch.setattr(pipe.transformer, "forward", lambda *a, **k: seen.append((a[0] if a else k["hidden_states"])[-1:].clone()) or forward(*a, **k))
	unpack = (lambda x: pipe._unpack_latents(x, 64, 64, pipe.vae_scale_factor)) if kind == "flux" else (lambda x: x)

	assert backend.generate(init, init, 0.5, {}, noise=noise).shape == (64, 64, 3)
	assert len(seen) == 4
	plain_start = unpack(seen[0])
	assert np.allclose(plain_start, backend.mix(backend.encode(init), torch_tensor(noise), backend.pipe.scheduler.sigmas[0]), atol=1e-5)

	seen.clear()
	regen = np.zeros((64, 64), np.float32)
	regen[:, :32] = 1
	backend.generate(init, init, 0.5, {}, noise=noise, regen=regen)
	assert len(seen) == 8
	held = unpack(seen[4])
	assert np.allclose(held[..., 4:], plain_start[..., 4:], atol=1e-4)
	assert not np.allclose(held[..., :4], plain_start[..., :4], atol=1e-2)


def torch_tensor(a):
	import torch
	return torch.from_numpy(a)[None]


@pytest.mark.slow
def test_sdxl_shared_attention_and_correspondence(monkeypatch):
	from vfbd3 import common
	monkeypatch.setattr(common, "load_loras", lambda pipe, specs: None)
	rng = np.random.default_rng(0)
	frame_in = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
	previous = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
	noise = rng.standard_normal((4, 8, 8), dtype=np.float32)
	images = dict(input=frame_in, feedback=previous, previous=previous)

	def output(**settings):
		cfg = frame.FrameConfig(prompt="a mural", control="canny", steps=4, guidance=1.0, seed=3, **settings)
		backend = backends.SDXLBackend(cfg, device="cpu", pipe=tiny_sdxl_pipeline())
		return backend, backend.generate(frame_in, frame_in, 0.75, {}, noise=noise, images=images)

	_, plain = output()
	_, shared = output(reference=1.0, reference_input=0.5, reference_adain=0.5)
	assert not np.array_equal(plain, shared)
	# a strength of 0 removes the reference from attention entirely
	_, off = output(reference_input=1e-30)
	assert np.allclose(off.astype(int), plain.astype(int), atol=1)
	backend, _ = output(flow_remap=1.0)
	corr, spread = backend.correspondence
	assert corr.shape[2] == 2 and corr.shape[:2] == spread.shape and corr.min() >= 0 and corr.max() <= 8


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

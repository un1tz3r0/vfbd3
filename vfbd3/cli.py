"""
Command line: `vfbd3 vace`, `vfbd3 frame`, `vfbd3 upscale`, `vfbd3 filters`.
"""

import click


def shared_options(fn):
	for option in reversed([
		click.argument("input_video", type=click.Path(exists=True, dir_okay=False)),
		click.argument("output_video", type=click.Path(dir_okay=False)),
		click.option("--prompt", required=True, help="what the output should look like"),
		click.option("--negative-prompt", default=None, help="what to steer away from"),
		click.option("--control", default="depth", show_default=True, help="structure signal taken from the input: depth, canny (vace also: gray, none)"),
		click.option("--control-scale", type=float, default=1.0, show_default=True, help="how strongly the control signal is followed"),
		click.option("--fast", default=None, help="few-step preset, which sets steps and guidance unless given. vace: causvid, selfforcing. frame: lcm, lightning, dmd2, turbo"),
		click.option("--steps", type=int, default=None, help="denoising steps"),
		click.option("--guidance", type=float, default=None, help="prompt guidance scale"),
		click.option("--seed", type=int, default=None, help="seed for the starting noise"),
		click.option("--color-fix", type=float, default=0.0, show_default=True, help="0..1, pull output colours back toward the first output to stop drift"),
		click.option("--lora", multiple=True, help="LoRA to load, as repo_or_path[:weight_file][@scale]; repeatable"),
		click.option("--start", type=float, default=None, help="start time in seconds"),
		click.option("--duration", type=float, default=None, help="seconds to process"),
		click.option("--upscale", type=int, default=None, help="short side of the final video in pixels (e.g. 720, 1080), reached with a super-resolution model"),
		click.option("--upscale-model", default=None, help="super-resolution checkpoint path or URL (default: Real-ESRGAN x2plus)"),
		click.option("--restart", is_flag=True, help="discard frames from an earlier interrupted run instead of resuming"),
	]):
		fn = option(fn)
	return fn


def _config(cls, options):
	''' build a config dataclass from CLI options, leaving unset ones at the dataclass default. '''
	return cls(**{k: v for k, v in options.items() if v is not None and v != ()})


@click.group()
def main():
	''' Restyle videos with diffusion models. Interrupted runs resume from <output>.work/. '''


@main.command()
@shared_options
@click.option("--model", default=None, help="Wan VACE checkpoint (default: Wan-AI/Wan2.1-VACE-1.3B-diffusers)")
@click.option("--max-area", type=int, default=None, help="pixels per frame the model generates at (default 832*480)")
@click.option("--fps", type=float, default=None, help="frame rate to process at (default 16, Wan's native rate)")
@click.option("--chunk-frames", type=int, default=None, help="frames per generation window, 4k+1 (default 81)")
@click.option("--overlap", type=int, default=None, help="frames carried from one window into the next (default 9)")
@click.option("--flow-shift", type=float, default=None, help="scheduler shift: 3 for 480p, 5 for 720p")
@click.option("--anchor", default=None, help="'first' (default: first output frame is a reference for every window), 'none', or an image path")
@click.option("--warp-frames", type=int, default=None, help="experimental: pre-fill this many frames after the overlap by flow-warping the last known frame")
@click.option("--motion-estimator", default=None, help="for --warp-frames: dis, farneback, raft, raft_small")
def vace(input_video, output_video, **options):
	''' Chunked video-to-video with the Wan VACE video model. '''
	from . import vace as engine
	options["lora"] = list(options["lora"])
	engine.run(input_video, output_video, _config(engine.VaceConfig, options))


@main.command()
@shared_options
@click.option("--model", default=None, help="SDXL checkpoint (default: stabilityai/stable-diffusion-xl-base-1.0)")
@click.option("--controlnet", default=None, help="SDXL ControlNet-Union checkpoint")
@click.option("--short-side", type=int, default=None, help="short side of the frames generated (default 720)")
@click.option("--fps", type=float, default=None, help="frame rate to process at (default: the input's)")
@click.option("--denoise", type=float, default=None, help="0..1, how much of each frame is re-generated (default 0.5)")
@click.option("--first-denoise", type=float, default=None, help="same, for the first frame (default 1.0)")
@click.option("--feedthrough", type=float, default=None, help="0..1, share of the raw input in each init image; the rest is the motion-compensated previous output (default 0)")
@click.option("--motion-estimator", default=None, help="dis (default), farneback, raft, raft_small")
@click.option("--motion-alpha", type=float, default=None, help="temporal smoothing of the motion field, 0..1")
@click.option("--motion-sigma", type=float, default=None, help="spatial smoothing of the motion field, in pixels")
@click.option("--occlusion/--no-occlusion", default=None, help="re-generate newly revealed areas instead of smearing the previous output into them (default on)")
@click.option("--occlusion-denoise", type=float, default=None, help="0..1, denoise used inside newly revealed areas (default 1.0); the rest of the frame keeps --denoise")
@click.option("--noise", default=None, help="starting noise: warp (default, one noise field carried along the motion), fixed (same every frame), random (new every frame)")
@click.option("--tiny-vae", is_flag=True, default=None, help="use the tiny SDXL autoencoder: much faster encode and decode, slightly softer detail")
@click.option("--ip-adapter", type=float, default=None, help="> 0: also condition on the warped previous output as an image prompt, at this scale")
@click.option("--input-filter", default=None, help="filter chain applied to each input frame (see `vfbd3 filters`)")
@click.option("--feedback-filter", default=None, help="filter chain applied to the motion-compensated previous output")
@click.option("--control-filter", default=None, help="filter chain applied to the control image")
@click.option("--output-filter", default=None, help="filter chain applied to each output frame (and so to what is fed back)")
def frame(input_video, output_video, **options):
	''' Per-frame img2img with motion-compensated feedback (the controlnetvideo technique). '''
	from . import frame as engine
	options["lora"] = list(options["lora"])
	engine.run(input_video, output_video, _config(engine.FrameConfig, options))


@main.command()
@click.argument("input_video", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_video", type=click.Path(dir_okay=False))
@click.option("--short-side", type=int, default=1080, show_default=True)
@click.option("--model", default=None, help="super-resolution checkpoint path or URL")
def upscale(input_video, output_video, short_side, model):
	''' Upscale a finished video with a super-resolution model. '''
	from . import video
	from .upscale import Upscaler
	info = video.probe(input_video)
	upscaler = Upscaler(model) if model else Upscaler()
	frames = video.read_frames(input_video, (info.width, info.height))
	count = video.write_video(output_video, (upscaler(f, short_side=short_side) for f in frames), info.fps,
		audio_from=input_video if info.has_audio else None)
	click.echo(f"wrote {count} frames to {output_video}")


@main.command(name="filters")
def list_filters():
	''' List the filters usable in --*-filter chains. '''
	from . import filters
	click.echo(filters.list_filters())


if __name__ == "__main__":
	main()

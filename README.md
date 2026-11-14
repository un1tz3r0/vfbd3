# vfbd3

Restyle video with diffusion models. Successor to
[controlnetvideo](https://github.com/un1tz3r0/controlnetvideo), built to run on a Colab T4.

Two engines, same inputs and outputs:

| | `vace` | `frame` |
|---|---|---|
| Model | Wan 2.1 VACE (video diffusion) | SDXL + ControlNet-Union (image diffusion) |
| Temporal consistency | from the model, across overlapping windows | motion-compensated feedback of the previous output |
| Look | stable, coherent motion | trails, drift, flow smearing: the feedback-loop look |
| Generates at | ~480p, 16 fps, then upscaled | 720p directly, input frame rate |

## Colab

Open `notebooks/vfbd3_colab.ipynb` in Colab with a T4 (or better) runtime. Keep the output on
Drive: frames are saved to `<output>.work/` as they finish, and re-running a cell after a
disconnect continues from the last finished frame.

## Command line

```sh
pip install "vfbd3[upscale] @ git+https://github.com/un1tz3r0/vfbd3"

vfbd3 vace  in.mp4 out.mp4 --prompt "a colorful murakami mural" --upscale 720
vfbd3 frame in.mp4 out.mp4 --prompt "murakami mural" --denoise 0.5
vfbd3 upscale out.mp4 out_1080.mp4 --short-side 1080
vfbd3 filters          # list filters for the --*-filter options
```

`vfbd3 vace --help` and `vfbd3 frame --help` list every option. ffmpeg must be on the path.

## How each engine works

**vace** splits the clip into windows of `--chunk-frames` (81). Each frame handed to the model
is either *known* (kept as is) or *generated* from a control signal extracted from the input
(depth by default). The last `--overlap` output frames of one window are the known head of
the next, and `--anchor first` gives every window the first output frame as a reference.
`--warp-frames N` (experimental) additionally pre-fills the N frames after the overlap with the
last known frame warped along the input's optical flow, leaving only disoccluded areas to generate.

**frame** runs, per frame: optical flow between the previous and current input → warp the
previous output along it → blend with the current input (`--feedthrough`) → img2img at
`--denoise` under the control signal → feed the result back. `--occlusion` fills newly revealed
areas from the input instead of smearing; `--ip-adapter` also conditions on the warped
previous output as an image prompt. Filter chains (`--input-filter`, `--feedback-filter`,
`--control-filter`, `--output-filter`) use the controlnetvideo filter syntax.

## Development

```sh
uv venv && uv pip install -e ".[upscale,dev]"
pytest                 # runs on CPU; -m "not slow" skips the tests that download models
```

The tests exercise both engine loops, chunking, resume, motion compensation and the upscaler
on CPU, using a tiny randomly-initialised VACE pipeline. They do not run the full-size models.

## Not here yet

Audio-reactive parameter animation (`frame.run` takes an `on_frame` hook for it), prompt
sequencing, pose/soft-edge controls, frame interpolation back to the input frame rate for
`vace`, and a diffusion-based upscaler.

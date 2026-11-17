"""
Video I/O through the ffmpeg command line (present on Colab and most desktops), plus the
resumable on-disk work directory both engines write their output frames into.
"""

import json
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
	width: int
	height: int
	fps: float
	duration: float
	has_audio: bool


def _fraction(s):
	num, _, den = str(s).partition("/")
	den = float(den) if den else 1.0
	return float(num) / den if den else 0.0


def probe(path):
	''' size (as displayed, i.e. after rotation metadata), fps, duration and audio presence. '''
	out = subprocess.run(
		["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
		check=True, capture_output=True, text=True).stdout
	data = json.loads(out)
	streams = data.get("streams", [])
	video = next((s for s in streams if s.get("codec_type") == "video"), None)
	if video is None:
		raise ValueError(f"no video stream in {path}")
	width, height = int(video["width"]), int(video["height"])
	rotation = 0
	for side in video.get("side_data_list", []):
		rotation = side.get("rotation", rotation)
	rotation = int(float(video.get("tags", {}).get("rotate", rotation)))
	if abs(rotation) % 180 == 90:
		width, height = height, width
	fps = _fraction(video.get("avg_frame_rate", "0")) or _fraction(video.get("r_frame_rate", "0"))
	duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
	return VideoInfo(width, height, fps, duration, any(s.get("codec_type") == "audio" for s in streams))


def fit_size(width, height, short_side=None, max_area=None, multiple=16):
	''' pick a working size with the input's aspect ratio: scale so the short side is
	`short_side` (if given), shrink further to fit `max_area` pixels (if given), then round
	both dimensions to a multiple of `multiple`. '''
	scale = 1.0
	if short_side:
		scale = short_side / min(width, height)
	if max_area and (width * scale) * (height * scale) > max_area:
		scale = (max_area / (width * height)) ** 0.5
	w = max(multiple, int(round(width * scale / multiple)) * multiple)
	h = max(multiple, int(round(height * scale / multiple)) * multiple)
	if max_area:
		# rounding up may have gone over budget; trim the long side until it fits
		while w * h > max_area and max(w, h) > multiple:
			if w >= h:
				w -= multiple
			else:
				h -= multiple
	return w, h


def read_frames(path, size, fps=None, start=None, duration=None):
	''' yield RGB uint8 frames of exactly `size` (w, h): scaled to cover, then centre-cropped.
	`fps` resamples the video to that frame rate. '''
	w, h = size
	cmd = ["ffmpeg", "-v", "error", "-nostdin"]
	if start:
		cmd += ["-ss", str(start)]
	if duration:
		cmd += ["-t", str(duration)]
	cmd += ["-i", str(path)]
	filters = []
	if fps:
		filters.append(f"fps={fps}")
	filters.append(f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos")
	filters.append(f"crop={w}:{h}")
	cmd += ["-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	nbytes = w * h * 3
	try:
		while True:
			buf = proc.stdout.read(nbytes)
			if len(buf) < nbytes:
				break
			yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
	finally:
		proc.stdout.close()
		proc.terminate()
		proc.wait()


def write_video(path, frames, fps, audio_from=None, audio_start=None, crf=16):
	''' encode an iterable of same-sized RGB uint8 frames to h264. If `audio_from` names a
	file with an audio stream, it is muxed in (offset by `audio_start` seconds) and trimmed
	to the video's length. Returns the number of frames written. '''
	frames = iter(frames)
	first = next(frames, None)
	if first is None:
		raise ValueError("no frames to write")
	h, w = first.shape[:2]
	cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
	if audio_from:
		if audio_start:
			cmd += ["-ss", str(audio_start)]
		cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
	# yuv420p needs even dimensions
	cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
		"-crf", str(crf), str(path)]
	pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
	proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
	count = 0
	try:
		frame = first
		while frame is not None:
			if frame.shape[:2] != (h, w):
				raise ValueError(f"frame {count} is {frame.shape[1]}x{frame.shape[0]}, expected {w}x{h}")
			proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
			count += 1
			frame = next(frames, None)
	finally:
		proc.stdin.close()
		code = proc.wait()
	if code != 0:
		raise RuntimeError(f"ffmpeg failed writing {path} (exit code {code})")
	return count


class Workdir:
	''' Output frames on disk next to the output video (`<output>.work/`), so an interrupted
	run (a Colab disconnect, say) picks up where it stopped. `done` counts the frames that
	are final; frames are only counted once `commit` records them. '''

	def __init__(self, output, signature, restart=False):
		self.path = pathlib.Path(str(output) + ".work")
		self.frames_dir = self.path / "frames"
		self.state_file = self.path / "state.json"
		if restart and self.path.exists():
			shutil.rmtree(self.path)
		self.frames_dir.mkdir(parents=True, exist_ok=True)
		self.done = 0
		if self.state_file.exists():
			state = json.loads(self.state_file.read_text())
			if state.get("signature") != signature:
				raise RuntimeError(
					f"{self.path} holds frames from a run with different settings "
					f"({state.get('signature')} vs {signature}). Use restart to discard them, "
					"or choose another output name.")
			self.done = int(state.get("done", 0))
		self.signature = signature

	def frame_path(self, index):
		return self.frames_dir / f"{index:07d}.png"

	def save(self, index, frame):
		cv2.imwrite(str(self.frame_path(index)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
			[cv2.IMWRITE_PNG_COMPRESSION, 3])

	def load(self, index):
		image = cv2.imread(str(self.frame_path(index)), cv2.IMREAD_COLOR)
		if image is None:
			raise FileNotFoundError(self.frame_path(index))
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	def commit(self, done):
		self.done = done
		tmp = self.state_file.with_suffix(".tmp")
		tmp.write_text(json.dumps({"signature": self.signature, "done": done}))
		tmp.replace(self.state_file)

	def frames(self):
		for index in range(self.done):
			yield self.load(index)

	def save_image(self, name, frame):
		cv2.imwrite(str(self.path / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	def load_image(self, name):
		image = cv2.imread(str(self.path / name), cv2.IMREAD_COLOR)
		return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def download(url, folder=".", max_height=1080):
	''' fetch a video (YouTube or anything else yt-dlp supports) as mp4 into `folder`, once;
	returns its path. Needs `pip install yt-dlp`. '''
	import yt_dlp
	options = {
		"format": f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b",
		"merge_output_format": "mp4",
		"outtmpl": str(pathlib.Path(folder) / "%(title).80B [%(id)s].%(ext)s"),
		"noplaylist": True,
		"quiet": True,
		"noprogress": True,
	}
	with yt_dlp.YoutubeDL(options) as ydl:
		info = ydl.extract_info(url, download=True)
		return str(pathlib.Path(ydl.prepare_filename(info)).with_suffix(".mp4"))

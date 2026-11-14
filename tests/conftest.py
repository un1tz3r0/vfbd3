import subprocess

import pytest


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
	''' 2 s, 64x48, 12 fps moving test pattern with a sine-wave soundtrack. '''
	path = tmp_path_factory.mktemp("clip") / "in.mp4"
	subprocess.run([
		"ffmpeg", "-v", "error", "-y",
		"-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=2",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=2",
		"-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
	return str(path)

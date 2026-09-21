import subprocess
from pathlib import Path

from PIL import Image

from h3studio.metadata import _ffmpeg_path
from h3studio.thumbnails import create_thumbnail


def test_thumbnail_is_bounded_and_cached_by_video_revision(tmp_path: Path, monkeypatch):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            _ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=640x360:d=0.1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    cache = Path("thumbs")
    first = create_thumbnail(video, cache, "job-1")
    second = create_thumbnail(video, cache, "job-1")
    assert first == second
    with Image.open(first) as image:
        assert image.format == "JPEG"
        assert image.size == (480, 270)
    assert len(list((tmp_path / cache).glob("job-1-*.jpg"))) == 1

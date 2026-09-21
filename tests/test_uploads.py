from pathlib import Path

import pytest
from PIL import Image

from h3studio.uploads import validate_image


def test_uploaded_image_is_decoded_and_format_checked(tmp_path: Path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), "navy").save(path)
    assert validate_image(path, "PNG") == (64, 48)
    with pytest.raises(ValueError, match="expected JPEG"):
        validate_image(path, "JPEG")


def test_corrupt_uploaded_image_is_rejected(tmp_path: Path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"not an image")
    with pytest.raises(Exception):
        validate_image(path, "PNG")

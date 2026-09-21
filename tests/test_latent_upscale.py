from __future__ import annotations

import pytest
import torch

from h3studio.latent_upscale import (
    LATENTS_MEAN,
    LATENTS_STD,
    denormalize_from_upscaler,
    normalize_for_upscaler,
)


def test_release_channel_normalization_and_inverse() -> None:
    latents = torch.zeros(1, 24, 2, 3, 4, dtype=torch.float32)
    normalized = normalize_for_upscaler(latents)
    expected = -torch.tensor(LATENTS_MEAN) / torch.tensor(LATENTS_STD)
    torch.testing.assert_close(normalized[0, :, 0, 0, 0], expected)
    torch.testing.assert_close(denormalize_from_upscaler(normalized), latents)


def test_release_channel_normalization_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="Bx24xTxHxW"):
        normalize_for_upscaler(torch.zeros(1, 23, 2, 3, 4))

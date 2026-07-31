"""D-QECG single-layer visual token pruning utilities."""

import torch


def _require_indices(indices: torch.Tensor, name: str) -> torch.Tensor:
    if indices is None:
        raise RuntimeError(f"D-QECG is enabled, but `{name}` is missing.")
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor.")
    return indices


def _as_visual_relative_indices(
    indices: torch.Tensor,
    name: str,
    image_token_length: int,
    device: torch.device,
) -> torch.Tensor:
    indices = _require_indices(indices, name).to(device=device, dtype=torch.long).view(-1)
    image_token_length = int(image_token_length)

    if indices.numel() == 0:
        return indices

    valid = (indices >= 0) & (indices < image_token_length)
    if not torch.all(valid):
        bad = indices[~valid][:8].detach().cpu().tolist()
        raise ValueError(
            f"`{name}` must contain visual-token-relative indices in "
            f"[0, {image_token_length}); got invalid values {bad}."
        )

    return torch.unique(indices, sorted=True)


def combine_d_squared_visual_token_indices(
    first_visual_indices: torch.Tensor,
    second_visual_indices: torch.Tensor,
    image_token_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Combine first- and second-stage visual-token-relative indices."""
    first = _as_visual_relative_indices(
        first_visual_indices,
        name="first_visual_indices",
        image_token_length=image_token_length,
        device=device,
    )
    second = _as_visual_relative_indices(
        second_visual_indices,
        name="second_visual_indices",
        image_token_length=image_token_length,
        device=device,
    )

    if first.numel() == 0:
        return second
    if second.numel() == 0:
        return first
    return torch.unique(torch.cat((first, second)), sorted=True)


def build_d_squared_visual_token_indices(
    first_visual_indices: torch.Tensor,
    second_visual_indices: torch.Tensor,
    image_token_start_index: int,
    image_token_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return absolute sequence indices for kept visual tokens."""
    visual_start = int(image_token_start_index)
    visual_indices = combine_d_squared_visual_token_indices(
        first_visual_indices=first_visual_indices,
        second_visual_indices=second_visual_indices,
        image_token_length=image_token_length,
        device=device,
    )
    return visual_indices + visual_start


def build_d_squared_keep_indices(
    first_visual_indices: torch.Tensor,
    second_visual_indices: torch.Tensor,
    image_token_start_index: int,
    image_token_length: int,
    seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build full sequence indices after visual-token pruning."""
    seq_length = int(seq_length)
    visual_start = int(image_token_start_index)
    visual_end = visual_start + int(image_token_length)

    if visual_start < 0 or visual_end > seq_length:
        raise ValueError(
            "D-QECG visual token range is outside the current sequence: "
            f"start={visual_start}, end={visual_end}, seq_length={seq_length}."
        )

    visual_indices = build_d_squared_visual_token_indices(
        first_visual_indices=first_visual_indices,
        second_visual_indices=second_visual_indices,
        image_token_start_index=image_token_start_index,
        image_token_length=image_token_length,
        device=device,
    )

    keep_indices = torch.cat(
        (
            torch.arange(visual_start, device=device),
            visual_indices,
            torch.arange(visual_end, seq_length, device=device),
        )
    )
    return keep_indices.sort().values

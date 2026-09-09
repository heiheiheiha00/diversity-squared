"""D-QECG visual-token pruning utilities independent of Transformers internals."""

import torch


def build_visual_position_keep_indices(
    selected_visual_indices: torch.Tensor,
    *,
    visual_positions: torch.Tensor,
    seq_length: int,
    device: torch.device,
    validate: bool = True,
) -> torch.Tensor:
    """Build sorted sequence keep indices for arbitrary visual-token blocks."""

    seq_length = int(seq_length)
    positions = _require_indices(visual_positions, "visual_positions").to(
        device=device, dtype=torch.long
    ).view(-1)
    if validate and positions.numel():
        valid = (positions >= 0) & (positions < seq_length)
        if not torch.all(valid):
            raise ValueError("Visual positions are outside the current sequence.")
        if positions.numel() > 1 and not torch.all(positions[1:] > positions[:-1]):
            raise ValueError("Visual positions must be strictly increasing.")
    selected = _as_visual_relative_indices(
        selected_visual_indices,
        name="selected_visual_indices",
        image_token_length=positions.numel(),
        device=device,
        validate=validate,
    )
    keep_mask = torch.ones(seq_length, device=device, dtype=torch.bool)
    keep_mask[positions] = False
    if selected.numel():
        keep_mask[positions.index_select(0, selected)] = True
    return torch.where(keep_mask)[0]


def build_visual_prune_keep_indices(
    selected_visual_indices: torch.Tensor,
    *,
    image_token_start_index: int,
    image_token_length: int,
    seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build full-sequence keep indices for one visual pruning boundary."""

    seq_length = int(seq_length)
    visual_start = int(image_token_start_index)
    visual_length = int(image_token_length)
    visual_end = visual_start + visual_length
    if visual_start < 0 or visual_length < 0 or visual_end > seq_length:
        raise ValueError(
            "D-QECG visual token range is outside the current sequence: "
            f"start={visual_start}, end={visual_end}, seq_length={seq_length}."
        )

    selected = _as_visual_relative_indices(
        selected_visual_indices,
        name="selected_visual_indices",
        image_token_length=visual_length,
        device=device,
    )
    return torch.cat(
        (
            torch.arange(visual_start, device=device, dtype=torch.long),
            selected + visual_start,
            torch.arange(visual_end, seq_length, device=device, dtype=torch.long),
        )
    )


def remap_kept_positions(
    positions: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Map kept absolute sequence positions through a sorted keep-index vector."""

    positions = positions.to(device=keep_indices.device, dtype=torch.long).view(-1)
    if positions.numel() == 0:
        return positions
    mapped = torch.searchsorted(keep_indices, positions)
    if validate:
        valid = mapped < keep_indices.numel()
        if not torch.all(valid) or not torch.equal(
            keep_indices.index_select(0, mapped[valid]),
            positions[valid],
        ):
            raise ValueError("All remapped positions must be retained by keep_indices.")
    return mapped


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
    validate: bool = True,
) -> torch.Tensor:
    indices = _require_indices(indices, name).to(device=device, dtype=torch.long).view(-1)
    image_token_length = int(image_token_length)

    if indices.numel() == 0:
        return indices

    if validate:
        valid = (indices >= 0) & (indices < image_token_length)
        if not torch.all(valid):
            bad = indices[~valid][:8].detach().cpu().tolist()
            raise ValueError(
                f"`{name}` must contain visual-token-relative indices in "
                f"[0, {image_token_length}); got invalid values {bad}."
            )
        return torch.unique(indices, sorted=True)
    return indices


def combine_d_squared_visual_token_indices(
    first_visual_indices: torch.Tensor,
    second_visual_indices: torch.Tensor,
    image_token_length: int,
    device: torch.device,
) -> torch.Tensor:
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
    visual_indices = combine_d_squared_visual_token_indices(
        first_visual_indices=first_visual_indices,
        second_visual_indices=second_visual_indices,
        image_token_length=image_token_length,
        device=device,
    )
    return visual_indices + int(image_token_start_index)


def build_d_squared_keep_indices(
    first_visual_indices: torch.Tensor,
    second_visual_indices: torch.Tensor,
    image_token_start_index: int,
    image_token_length: int,
    seq_length: int,
    device: torch.device,
) -> torch.Tensor:
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
        image_token_start_index=visual_start,
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

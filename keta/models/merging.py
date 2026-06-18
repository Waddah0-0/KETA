import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def ties_merge_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    density: float = 0.2,
    scaling_coefficient: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge on a list of adapter state dicts."""
    if not state_dicts:
        raise ValueError("state_dicts list is empty")

    keys = list(state_dicts[0].keys())
    merged = {}
    logger.info(f"TIES-Merging {len(state_dicts)} adapters (density={density}, λ={scaling_coefficient})")

    for key in keys:
        tensors = [sd[key] for sd in state_dicts]

        if not tensors[0].is_floating_point():
            merged[key] = tensors[0].clone()
            continue

        dtype = tensors[0].dtype

        # 1. Trim — keep top density% by magnitude
        trimmed = []
        for t in tensors:
            flat = t.flatten()
            k = int(density * flat.numel())
            if k == 0:
                trimmed.append(torch.zeros_like(t))
                continue
            threshold, _ = torch.kthvalue(flat.abs(), flat.numel() - k + 1)
            mask = (flat.abs() >= threshold).view_as(t)
            trimmed.append(torch.where(mask, t, torch.zeros_like(t)))

        stacked = torch.stack(trimmed, dim=0)

        # 2. Elect Sign — majority vote weighted by magnitude
        elected_sign = torch.sign(stacked.sum(dim=0))

        # 3. Disjoint Merge — average only sign-aligned updates
        task_signs = torch.sign(stacked)
        align_mask = (task_signs == elected_sign.unsqueeze(0)) & (elected_sign.unsqueeze(0) != 0)
        aligned = torch.where(align_mask, stacked, torch.zeros_like(stacked))
        num_aligned = align_mask.to(torch.int32).sum(dim=0)
        mean_aligned = torch.where(
            num_aligned > 0,
            aligned.sum(dim=0) / num_aligned.clamp(min=1).to(dtype),
            torch.zeros_like(aligned.sum(dim=0)),
        )

        # 4. Scale
        merged[key] = scaling_coefficient * mean_aligned

    logger.info("TIES-Merging complete")
    return merged

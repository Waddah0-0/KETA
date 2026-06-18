import torch
from keta.models.merging import ties_merge_state_dicts


def test_ties_merge_state_dicts():
    t1 = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    t2 = torch.tensor([-1.0, 1.5, -0.5, -3.0, 6.0])

    # density=0.6 keeps top 3 by magnitude per task
    # Trimmed t1: [0, 0, 3, -4, 5]    Trimmed t2: [0, 1.5, 0, -3, 6]
    # Sum → [0, 1.5, 3, -7, 11]  →  Sign: [0, +, +, -, +]
    # Aligned averages: [0, 1.5, 3.0, -3.5, 5.5]
    # Scaled by λ=0.5: [0, 0.75, 1.5, -1.75, 2.75]

    merged = ties_merge_state_dicts([{"weight": t1}, {"weight": t2}], density=0.6, scaling_coefficient=0.5)
    expected = torch.tensor([0.0, 0.75, 1.5, -1.75, 2.75])
    assert torch.allclose(merged["weight"], expected, atol=1e-5)

    # Non-float tensors should pass through unchanged
    merged_int = ties_merge_state_dicts([{"step": torch.tensor([10])}, {"step": torch.tensor([10])}])
    assert merged_int["step"].item() == 10


if __name__ == "__main__":
    test_ties_merge_state_dicts()
    print("test_merging.py passed")

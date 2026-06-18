import logging
from typing import List

logger = logging.getLogger(__name__)


def get_keta_peft_model(
    model,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    target_layers: List[int] = None,
    target_modules: List[str] = None,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 3407,
):
    """Applies LW-LoRA via Unsloth, restricting adapters to specific transformer layers."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is not installed.")

    if target_layers is None:
        target_layers = list(range(13, 26))  # layers 14–26 (1-based) for a 32-layer model
    else:
        target_layers = [int(l) for l in target_layers]

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    logger.info(f"LW-LoRA r={r} alpha={lora_alpha} layers={target_layers}")

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        layers_to_transform=target_layers,
    )

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"Trainable: {trainable:,} ({100 * trainable / total:.4f}%)")

    return peft_model

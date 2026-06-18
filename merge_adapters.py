import argparse
import os
import shutil
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("merge_adapters")


def load_adapter_weights(path: str) -> dict:
    """Loads adapter weights from safetensors or bin format."""
    st_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")

    if os.path.exists(st_path):
        try:
            from safetensors.torch import load_file
            logger.info(f"Loading safetensors: {st_path}")
            return load_file(st_path)
        except ImportError:
            logger.warning("safetensors not installed, trying .bin fallback")

    if os.path.exists(bin_path):
        logger.info(f"Loading bin: {bin_path}")
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No adapter_model.safetensors or .bin in {path}")


def save_adapter_weights(weights: dict, path: str):
    os.makedirs(path, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(weights, os.path.join(path, "adapter_model.safetensors"))
        return
    except ImportError:
        pass
    torch.save(weights, os.path.join(path, "adapter_model.bin"))


def main():
    p = argparse.ArgumentParser(description="KETA-Net: TIES-Merge monolingual + translation adapters")
    p.add_argument("--adapter_mono_dir", type=str, default="./outputs/theta_mono")
    p.add_argument("--adapter_trans_dir", type=str, default="./outputs/theta_trans")
    p.add_argument("--output_dir", type=str, default="./outputs/theta_merged")
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--scaling_coefficient", type=float, default=0.3)
    args = p.parse_args()

    try:
        mono = load_adapter_weights(args.adapter_mono_dir)
        trans = load_adapter_weights(args.adapter_trans_dir)
    except FileNotFoundError as e:
        logger.error(f"{e} — run train_mono.py and train_trans.py first")
        return

    from keta.models.merging import ties_merge_state_dicts
    merged = ties_merge_state_dicts([mono, trans], density=args.density, scaling_coefficient=args.scaling_coefficient)
    save_adapter_weights(merged, args.output_dir)

    # Copy config files from mono adapter so PEFT can load the merged folder
    for name in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(args.adapter_mono_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, name))

    logger.info(f"Merged adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()

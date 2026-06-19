import argparse
import os
import torch
import logging
from keta.decoding.metrics import DialectScorer
from keta.decoding.mbr import DialectAwareMBR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_mbr_inference")


def main():
    p = argparse.ArgumentParser(description="KETA-Net: Dialect-Aware MBR Inference")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter_dir", type=str, default="./outputs/theta_merged")
    p.add_argument("--prompt", type=str, default="Can you help me cancel my order and get a refund?")

    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.5,
                    help="Weight for chrF++ in linear utility: U = α·chrF + (1-α)·ADI2")

    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        logger.warning("Unsloth not installed, using vanilla HF")
        use_unsloth = False

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=512,
            load_in_4bit=torch.cuda.is_available(),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if os.path.exists(args.adapter_dir):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_dir)
            FastLanguageModel.for_inference(model)
        else:
            logger.warning(f"No adapter at {args.adapter_dir}, using base model")
            FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if os.path.exists(args.adapter_dir):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_dir)

    scorer = DialectScorer()
    mbr = DialectAwareMBR(model=model, tokenizer=tokenizer, dialect_scorer=scorer)

    formatted_prompt = (
        f"<|im_start|>system\nYou are a helpful bilingual assistant translating English queries into GCC Arabic dialect.<|im_end|>\n"
        f"<|im_start|>user\nTranslate this message into fluent Gulf Arabic: \"{args.prompt}\"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    best, info = mbr.decode(
        prompt=formatted_prompt,
        num_candidates=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        alpha=args.alpha,
    )

    print(f"\n{'='*80}")
    print("KETA-NET MBR DECODING RESULTS")
    print(f"{'='*80}")
    print(f"English: {args.prompt}")
    print(f"{'─'*80}")
    print(f"Best:    {best}")
    print(f"{'='*80}")

    print(f"\n{'Idx':<5} {'chrF++':>8} {'ADI2':>8} {'Utility':>8}   Candidate")
    print("─" * 90)
    for i, c in enumerate(info["candidates"]):
        marker = "→" if i == info["selected_index"] else " "
        print(f"{marker}{i:<4} {info['semantic_scores'][i]:>8.4f} {info['dialect_scores'][i]:>8.4f} {info['utility_scores'][i]:>8.4f}   {c}")


if __name__ == "__main__":
    main()

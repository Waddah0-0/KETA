try:
    import unsloth
except ImportError:
    pass

import argparse
import json
import os
import torch
import logging
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_trans")


def gpu_stats(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"{tag} VRAM: {alloc:.2f}/{total:.2f} GB")


SYSTEM_PROMPT = "أنت موظف خدمة عملاء خليجي ودود ومحترف. تتحدث بلهجة خليجية طبيعية وتساعد العميل بكل احترام."


def format_conversation(sample: dict) -> str:
    """Builds a ChatML string from a multi-turn conversation sample.

    Expected format:
    {
        "context": {"intent": "...", "region": "..."},         # optional
        "conversation_history": [{"role": "...", "content": "..."}, ...],  # optional
        "next_turn": {
            "prompt": "...",
            "chosen": "...",       # the response we train on
            "rejected": "..."     # ignored in SFT, useful for future DPO
        }
    }
    """
    ctx = sample.get("context", {})
    intent = ctx.get("intent", "")
    region = ctx.get("region", "")

    # Build system prompt with context if available
    sys_prompt = SYSTEM_PROMPT
    if intent or region:
        extras = []
        if intent:
            extras.append(f"intent: {intent}")
        if region:
            extras.append(f"region: {region}")
        sys_prompt += "\n[" + ", ".join(extras) + "]"

    parts = [f"<|im_start|>system\n{sys_prompt}<|im_end|>"]

    # Replay conversation history
    for turn in sample.get("conversation_history", []):
        role = turn["role"]
        parts.append(f"<|im_start|>{role}\n{turn['content']}<|im_end|>")

    # Final turn: user prompt + chosen assistant response
    next_turn = sample.get("next_turn", {})
    if next_turn.get("prompt"):
        parts.append(f"<|im_start|>user\n{next_turn['prompt']}<|im_end|>")
    if next_turn.get("chosen"):
        parts.append(f"<|im_start|>assistant\n{next_turn['chosen']}<|im_end|>")

    return "\n".join(parts) + "\n"


def load_conversations(path: str) -> list:
    """Loads .json (list) or .jsonl (one-per-line) conversation files robustly."""
    samples = []
    logger.info(f"Loading conversations from {path}...")
    
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_idx+1}: {e}")
    else:
        # Robust .json parsing (handles concatenated arrays or multiple root objects)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            # Skip whitespace and commas
            while idx < len(content) and (content[idx].isspace() or content[idx] == ","):
                idx += 1
            if idx >= len(content):
                break
                
            try:
                obj, next_idx = decoder.raw_decode(content, idx)
                if isinstance(obj, list):
                    samples.extend([item for item in obj if isinstance(item, dict)])
                elif isinstance(obj, dict):
                    samples.append(obj)
                else:
                    logger.warning(f"Skipped non-dict JSON object at {idx}")
                idx = next_idx
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON at character {idx}: {e}")
                # Try to recover by skipping to the next bracket or brace
                next_bracket = content.find("[", idx + 1)
                next_brace = content.find("{", idx + 1)
                
                valid_nexts = [n for n in (next_bracket, next_brace) if n != -1]
                if not valid_nexts:
                    break
                idx = min(valid_nexts)
                logger.warning(f"Attempting to recover by jumping to character {idx}")

    logger.info(f"Loaded {len(samples)} conversations from {path}")
    return samples


# Dummy data for testing without a real dataset
DUMMY_CONVERSATIONS = [
    {
        "context": {"intent": "shipping_complaint", "region": "Saudi_Najdi"},
        "conversation_history": [
            {"role": "user", "content": "السلام عليكم، طلبي وش صار فيه؟ صار له ٦ أيام وما وصل"},
            {"role": "assistant", "content": "وعليكم السلام، حياك الله. عطني رقم الطلب وأشوف لك وينه الحين 🙏"},
        ],
        "next_turn": {
            "prompt": "طيب وش الحل؟ أنا تعبت من الانتظار صراحة",
            "chosen": "والله معك حق وأعتذر عن التأخير. حجزت لك توصيل سريع بكره الصبح إن شاء الله، ومن طرفنا بنعفيك من رسوم الشحن. أبشر 🙏",
        },
    },
    {
        "context": {"intent": "refund_request", "region": "Kuwait"},
        "conversation_history": [
            {"role": "user", "content": "هلا، أبي أرجع المنتج هذا مو نفس اللي بالصورة"},
        ],
        "next_turn": {
            "prompt": "شلون أرجعه؟",
            "chosen": "أهلاً فيك! تقدر ترجعه من خلال التطبيق، ادخل على طلباتي واختار إرجاع. أو إذا تبي أرسل لك مندوب يمر عليك، شنو أحسن لك؟",
        },
    },
    {
        "context": {"intent": "order_status", "region": "UAE"},
        "conversation_history": [],
        "next_turn": {
            "prompt": "مرحبا شحالكم، وين طلبي؟ رقمه ٧٧٨٣",
            "chosen": "هلا والله! خلني أشيك لك على الطلب الحين. لحظة وحدة طال عمرك ⏳",
        },
    },
    {
        "context": {"intent": "product_inquiry", "region": "Bahrain"},
        "conversation_history": [
            {"role": "user", "content": "عندكم الجهاز هذا بلون ثاني؟"},
            {"role": "assistant", "content": "هلا فيك! أي جهاز بالضبط تقصد؟"},
        ],
        "next_turn": {
            "prompt": "آيفون ١٦ أبيه بالأزرق",
            "chosen": "متوفر عندنا بالأزرق والأسود. تبي أحجز لك وحدة؟ التوصيل يوصلك خلال يومين إن شاء الله",
        },
    },
    {
        "context": {"intent": "greeting", "region": "Qatar"},
        "conversation_history": [],
        "next_turn": {
            "prompt": "السلام عليكم",
            "chosen": "وعليكم السلام ورحمة الله! حياك الله، شلون أقدر أساعدك اليوم؟ 😊",
        },
    },
] * 10


def main():
    p = argparse.ArgumentParser(description="KETA-Net: Customer service instruction adapter (θ_trans)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs/theta_trans")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--target_layers", type=int, nargs="+", default=list(range(13, 26)),
                    help="0-indexed layers to apply LoRA (default: 13–25)")

    args = p.parse_args()
    gpu_stats("Start")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed")
        return

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        device_map={"": 0},
    )
    gpu_stats("Model loaded")

    # LW-LoRA
    from keta.models.peft_utils import get_keta_peft_model
    model = get_keta_peft_model(
        model, r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_layers=args.target_layers, use_gradient_checkpointing="unsloth",
    )
    gpu_stats("LoRA applied")

    # Data
    from datasets import Dataset as HFDataset

    if args.dataset_path and os.path.exists(args.dataset_path):
        conversations = load_conversations(args.dataset_path)
    else:
        logger.warning("No dataset — using dummy customer service conversations")
        conversations = DUMMY_CONVERSATIONS

    formatted = [format_conversation(conv) for conv in conversations]
    formatted = [f for f in formatted if f.strip()]  # drop empties
    logger.info(f"Formatted {len(formatted)} training examples")

    dataset = HFDataset.from_dict({"text": formatted})

    tokenizer.model_max_length = args.max_seq_length

    sft_config = SFTConfig(
        packing=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=args.output_dir,
        save_strategy="no",
        report_to="none",
    )
    # Monkeypatch to avoid init TypeError on older TRL 0.24 versions
    sft_config.max_seq_length = args.max_seq_length
    sft_config.dataset_text_field = "text"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Adapter saved to {args.output_dir}")
    gpu_stats("Done")


if __name__ == "__main__":
    main()

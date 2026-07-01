import argparse
import os
import torch
import logging
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_mono")


def gpu_stats(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"{tag} VRAM: {alloc:.2f}/{total:.2f} GB")


def main():
    p = argparse.ArgumentParser(description="KETA-Net: Monolingual dialectal adapter (θ_mono)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs/theta_mono")
    p.add_argument("--add_vocab_size", type=int, default=0, help="New dialectal tokens to add via PVE")

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
        device_map="auto",
    )
    gpu_stats("Model loaded")

    # PVE
    if args.add_vocab_size > 0 and args.dataset_path:
        from keta.tokenizer.pve import train_dialect_vocabulary, expand_vocabulary_and_embeddings
        new_tokens = train_dialect_vocabulary(args.dataset_path, args.add_vocab_size, tokenizer)
        tokenizer = expand_vocabulary_and_embeddings(model, tokenizer, new_tokens)
        gpu_stats("PVE done")

    # LW-LoRA
    from keta.models.peft_utils import get_keta_peft_model
    model = get_keta_peft_model(
        model, r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_layers=args.target_layers, use_gradient_checkpointing="unsloth",
    )
    gpu_stats("LoRA applied")

    # Data
    from datasets import Dataset as HFDataset
    from keta.utils.data import KETADataset, get_formatting_prompts_fn

    if args.dataset_path and os.path.exists(args.dataset_path):
        raw = KETADataset(args.dataset_path)
        samples = [s["text"] for s in raw.samples if "text" in s]
    else:
        logger.warning("No dataset — using dummy GCC samples")
        samples = [
            "شلونك طال عمرك؟ عساك طيب وبخير يا رب",
            "أبي أسألك الحين إذا تسوي توصيل للرياض أو لا",
            "تكفى شف لي طريقة أبي أرجع الطلب هذا وأسترجع فلوسي وايد تأخرتوا",
            "شنو صار على الشحنة؟ للحين ما وصلت عندي هالحزة",
            "عساكم على القوة يا أهل الكويت، نبي نطلب كيكة عيد ميلاد الحين",
        ] * 10

    fmt = get_formatting_prompts_fn(system_prompt="أنت مساعد ذكي تتحدث بلهجة خليجية عامية وتساعد العميل في طلباته.")
    dataset = HFDataset.from_dict(fmt({"text": samples}))

    tokenizer.model_max_length = args.max_seq_length
    tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
    tokenizer.eos_token = "<|im_end|>"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            max_seq_length=args.max_seq_length,
            eos_token="<|im_end|>",
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
        ),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Adapter saved to {args.output_dir}")
    gpu_stats("Done")


if __name__ == "__main__":
    main()

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import List
import logging

logger = logging.getLogger(__name__)


def train_dialect_vocabulary(
    corpus_path: str,
    vocab_size_to_add: int,
    base_tokenizer: PreTrainedTokenizerFast,
) -> List[str]:
    """Trains BPE on dialectal text to discover new subwords not in the base tokenizer."""
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        raise ImportError("Install the tokenizers library: pip install tokenizers")

    logger.info(f"Training dialectal vocab on {corpus_path} ({vocab_size_to_add} new tokens)...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    target_vocab_size = vocab_size_to_add * 2 + 1000
    trainer = trainers.BpeTrainer(special_tokens=[], vocab_size=target_vocab_size)
    tokenizer.train([corpus_path], trainer)

    dialect_vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(dialect_vocab.items(), key=lambda x: x[1])

    base_vocab = base_tokenizer.get_vocab()
    new_tokens = []
    for token, _ in sorted_tokens:
        tok = token.strip()
        if not tok or len(tok) <= 1:
            continue
        if tok not in base_vocab and token not in base_vocab:
            new_tokens.append(tok)
            if len(new_tokens) >= vocab_size_to_add:
                break

    logger.info(f"Found {len(new_tokens)} new dialectal tokens")
    return new_tokens


def expand_vocabulary_and_embeddings(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    new_tokens: List[str],
) -> PreTrainedTokenizerFast:
    """
    Adds new tokens to the tokenizer, resizes embeddings, and initializes new
    embeddings by averaging their constituent subword fragments in the original space.
    """
    if not new_tokens:
        logger.warning("No new tokens provided")
        return tokenizer

    num_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Added {num_added} tokens (vocab size: {len(tokenizer)})")

    original_vocab_size = len(tokenizer) - num_added
    model.resize_token_embeddings(len(tokenizer))

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    in_data = input_emb.weight.data
    has_output = output_emb is not None
    out_data = output_emb.weight.data if has_output else None

    # Load a clean copy of the base tokenizer for fragmenting new tokens
    try:
        ancestral_tok = AutoTokenizer.from_pretrained(tokenizer.name_or_path, trust_remote_code=True)
    except Exception:
        ancestral_tok = None

    initialized = 0
    for token in new_tokens:
        new_id = tokenizer.convert_tokens_to_ids(token)

        if ancestral_tok is not None:
            subword_ids = ancestral_tok.encode(token, add_special_tokens=False)
        else:
            base_vocab = tokenizer.get_vocab()
            subword_ids = [tokenizer.convert_tokens_to_ids(c) for c in token if c in base_vocab]

        # Only use IDs from the original embedding range
        subword_ids = [sid for sid in subword_ids if sid is not None and sid < original_vocab_size]

        if not subword_ids:
            in_data[new_id] = in_data[:original_vocab_size].mean(dim=0)
            if has_output:
                out_data[new_id] = out_data[:original_vocab_size].mean(dim=0)
            continue

        in_data[new_id] = in_data[subword_ids].mean(dim=0)
        if has_output:
            out_data[new_id] = out_data[subword_ids].mean(dim=0)
        initialized += 1

    logger.info(f"Initialized {initialized} new token embeddings via subword averaging")
    return tokenizer

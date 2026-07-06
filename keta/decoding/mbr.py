import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from typing import List, Dict, Tuple, Any
import numpy as np
import logging
from .metrics import DialectScorer

logger = logging.getLogger(__name__)


def chrf_score_simple(hyp: str, ref: str, char_n: int = 6, word_n: int = 2, beta: float = 2.0) -> float:
    """chrF++ score: character n-gram F-score plus word n-grams. Uses NLTK if available."""
    try:
        from nltk.translate.chrf_score import sentence_chrf
        return sentence_chrf(ref, hyp, min_len=1, max_len=char_n, beta=beta)
    except ImportError:
        pass

    if not hyp or not ref:
        return 0.0

    def char_ngrams(text: str, size: int) -> Dict[str, int]:
        grams = {}
        for i in range(len(text) - size + 1):
            g = text[i : i + size]
            grams[g] = grams.get(g, 0) + 1
        return grams

    def word_ngrams(text: str, size: int) -> Dict[str, int]:
        tokens = text.split()
        grams = {}
        for i in range(len(tokens) - size + 1):
            g = " ".join(tokens[i : i + size])
            grams[g] = grams.get(g, 0) + 1
        return grams

    def _fscore(h_ng: Dict[str, int], r_ng: Dict[str, int]) -> float:
        if not h_ng or not r_ng:
            return 0.0
        matches = sum(min(freq, r_ng.get(g, 0)) for g, freq in h_ng.items())
        p = matches / sum(h_ng.values()) if h_ng else 0.0
        r = matches / sum(r_ng.values()) if r_ng else 0.0
        denom = beta**2 * p + r
        return (1 + beta**2) * p * r / denom if denom > 0 else 0.0

    # Character n-gram F-scores (chrF)
    f_scores = []
    for i in range(1, char_n + 1):
        f_scores.append(_fscore(char_ngrams(hyp, i), char_ngrams(ref, i)))

    # Word n-gram F-scores (the ++ in chrF++)
    for i in range(1, word_n + 1):
        f_scores.append(_fscore(word_ngrams(hyp, i), word_ngrams(ref, i)))

    return sum(f_scores) / len(f_scores) if f_scores else 0.0


class DialectAwareMBR:
    """MBR decoder that generates N candidates and picks the one with best
    joint semantic similarity (chrF++) × dialectal fidelity (ADI2)."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, dialect_scorer: DialectScorer):
        self.model = model
        self.tokenizer = tokenizer
        self.dialect_scorer = dialect_scorer

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 20,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """Stochastically generates N candidate sequences."""
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        eos_token_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_tokens = [t for t in [eos_token_id, im_end_id] if t is not None]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_candidates,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=stop_tokens,
            )

        candidates = list({
            self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
            for i in range(num_candidates)
        })
        logger.info(f"Generated {num_candidates} candidates ({len(candidates)} unique)")
        return candidates if candidates else [""]

    def decode(
        self,
        prompt: str,
        num_candidates: int = 20,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        alpha: float = 0.5,
    ) -> Tuple[str, Dict[str, Any]]:
        """Runs MBR: generate → score pairwise chrF++ → score dialectness → pick best.
        Utility is a linear combination: U(y) = α·chrF++(y) + (1-α)·ADI2(y).
        """
        candidates = self.generate_candidates(prompt, num_candidates, max_new_tokens, temperature, top_p)

        if len(candidates) == 1:
            return candidates[0], {"candidates": candidates, "scores": [1.0]}

        N = len(candidates)

        # Pairwise chrF++ mean per candidate
        semantic_scores = np.zeros(N)
        for i in range(N):
            pairwise = sum(chrf_score_simple(candidates[i], candidates[j]) for j in range(N) if i != j)
            semantic_scores[i] = pairwise / (N - 1)

        dialect_scores = np.array([self.dialect_scorer.score(c) for c in candidates])

        # Joint utility: linear combination as per paper
        joint = alpha * semantic_scores + (1 - alpha) * dialect_scores

        best_idx = int(np.argmax(joint))
        logger.info(f"Best candidate #{best_idx} (utility={joint[best_idx]:.4f})")

        return candidates[best_idx], {
            "candidates": candidates,
            "semantic_scores": semantic_scores.tolist(),
            "dialect_scores": dialect_scores.tolist(),
            "utility_scores": joint.tolist(),
            "selected_index": best_idx,
        }

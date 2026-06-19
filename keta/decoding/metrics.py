import re
import logging

logger = logging.getLogger(__name__)

# ASCII-only punctuation pattern — won't touch Arabic characters
_PUNCT_RE = re.compile(r"[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]")


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


class DialectScorer:
    """Scores text for GCC dialectness using lexicon matching and optional neural classifier."""

    def __init__(self, classifier_model_name: str = None, target_dialect_label: str = "glf"):
        self.target_dialect_label = target_dialect_label
        self.pipeline = None

        if classifier_model_name:
            try:
                from transformers import pipeline
                self.pipeline = pipeline("text-classification", model=classifier_model_name, top_k=None)
            except Exception as e:
                logger.warning(f"Could not load classifier pipeline: {e}")

        self.gulf_lexicon = {
            r"\bشلون\b", r"\bوش\b", r"\bشسويت\b", r"\bتكفى\b", r"\bتكفون\b",
            r"\bياخوي\b", r"\bالحين\b", r"\bأبي\b", r"\bنبي\b", r"\bشنو\b",
            r"\bشخبار\b", r"\bعساك\b", r"\bسيد سيدي\b", r"\bزين\b", r"\bوايد\b",
            r"\bمرّة\b", r"\bهالحزة\b", r"\bعيل\b", r"\bشكو\b", r"\bطال عمرك\b",
            r"\bشصار\b", r"\bبسك\b", r"\bشحقة\b", r"\bبخص\b", r"\bميب\b",
        }

        self.msa_lexicon = {
            r"\bسوف\b", r"\bلماذا\b", r"\bكيف\b", r"\bماذا\b", r"\bهكذا\b",
            r"\bالآن\b", r"\bأريد\b", r"\bكثيراً\b", r"\bجيد\b", r"\bفقط\b",
            r"\bلكن\b", r"\bالذي\b", r"\bالتي\b", r"\bهؤلاء\b", r"\bهذا\b",
        }

    def compute_aldi_score(self, text: str) -> float:
        """Density of Gulf dialect markers over the total word count. Returns 0.0–1.0."""
        words = text.split()
        if not words:
            return 0.0

        gulf_matches = 0

        for word in words:
            clean = _PUNCT_RE.sub("", word)
            if not clean:
                continue

            for pattern in self.gulf_lexicon:
                if re.search(pattern, clean):
                    gulf_matches += 1
                    break

        if gulf_matches == 0:
            return 0.0

        return float(_clamp(gulf_matches / len(words), 0.0, 1.0))

    def compute_nadi_score(self, text: str) -> float:
        """Posterior probability that text is target dialect."""
        if self.pipeline is not None:
            try:
                outputs = self.pipeline(text)
                for pred in outputs[0]:
                    label = pred["label"].lower()
                    if self.target_dialect_label in label or label in self.target_dialect_label:
                        return float(pred["score"])
                return 0.05
            except Exception as e:
                logger.error(f"Classifier error: {e}")
                return 0.2

        # Rule-based fallback
        gulf_density = sum(1 for w in text.split() if any(re.search(p, w) for p in self.gulf_lexicon))
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        return min(1.0, gulf_density / total_words)

    def score(self, text: str) -> float:
        """ADI2 = ALDi(y) * NADI(y)_C"""
        return self.compute_aldi_score(text) * self.compute_nadi_score(text)

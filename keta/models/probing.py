import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from typing import List, Dict, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LayerWiseProbe:
    """Probes each transformer layer with a logistic regression classifier to find
    which layers encode dialectal style. Used to select target layers for LW-LoRA."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def collect_hidden_states(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (num_layers, num_samples, hidden_dim) and (num_samples,) arrays."""
        self.model.eval()
        device = next(self.model.parameters()).device

        all_layer_states = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                # Skip embedding layer (index 0), use last non-padding token per sample
                mask = inputs.attention_mask
                batch_layer_states = []
                for layer_idx, state in enumerate(outputs.hidden_states):
                    if layer_idx == 0:
                        continue
                    last_indices = mask.sum(dim=1) - 1
                    samples = [state[s, last_indices[s]].cpu().numpy() for s in range(state.shape[0])]
                    batch_layer_states.append(np.stack(samples))

                all_layer_states.append(np.stack(batch_layer_states))
                all_labels.extend(batch_labels)

        return np.concatenate(all_layer_states, axis=1), np.array(all_labels)

    def probe_layers(
        self,
        texts: List[str],
        labels: List[int],
        threshold: float = 0.75,
        test_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Trains a probe per layer and returns scores + target layers above threshold."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("Install scikit-learn: pip install scikit-learn")

        hidden_states, labels_arr = self.collect_hidden_states(texts, labels)
        num_layers, num_samples, hidden_dim = hidden_states.shape
        logger.info(f"Probing {num_layers} layers, {num_samples} samples, dim={hidden_dim}")

        scores = {}
        target_layers = []

        for layer_idx in range(num_layers):
            X_train, X_test, y_train, y_test = train_test_split(
                hidden_states[layer_idx], labels_arr,
                test_size=test_split, random_state=42, stratify=labels_arr,
            )
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
            clf.fit(X_train, y_train)

            macro_f1 = f1_score(y_test, clf.predict(X_test), average="macro")
            layer_num = layer_idx + 1  # 1-indexed
            scores[layer_num] = macro_f1

            if macro_f1 >= threshold:
                target_layers.append(layer_num)
            logger.info(f"Layer {layer_num:2d} F1: {macro_f1:.4f}")

        # Fallback: top 30% if nothing crossed the threshold
        if not target_layers:
            logger.warning("No layers crossed threshold, picking top 30%")
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_k = max(1, int(num_layers * 0.3))
            target_layers = [l for l, _ in ranked[:top_k]]

        target_layers = sorted(target_layers)
        logger.info(f"Selected layers: {target_layers}")
        return {"probing_scores": scores, "target_layers": target_layers}

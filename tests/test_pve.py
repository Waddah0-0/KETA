import torch
from unittest.mock import MagicMock
from keta.tokenizer.pve import expand_vocabulary_and_embeddings
import keta.tokenizer.pve as pve_module


class DummyEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(vocab_size * dim, dtype=torch.float32).view(vocab_size, dim)
        )


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size, dim=4):
        super().__init__()
        self.embeddings = DummyEmbedding(vocab_size, dim)
        self.lm_head = DummyEmbedding(vocab_size, dim)

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        dim = self.embeddings.weight.shape[1]
        for emb in [self.embeddings, self.lm_head]:
            new_w = torch.nn.Parameter(torch.zeros(new_size, dim))
            new_w.data[:emb.weight.shape[0]] = emb.weight.data
            emb.weight = new_w


def test_expand_vocabulary_and_embeddings():
    model = DummyModel(vocab_size=5, dim=3)
    # E[0]=[0,1,2] E[1]=[3,4,5] E[2]=[6,7,8] E[3]=[9,10,11] E[4]=[12,13,14]

    tokenizer = MagicMock()
    tokenizer.name_or_path = "dummy-base"
    tokenizer.add_tokens = MagicMock(return_value=1)
    tokenizer.get_vocab = MagicMock(return_value={"t0": 0, "t1": 1, "t2": 2, "t3": 3, "t4": 4})
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=5)
    tokenizer.__len__ = MagicMock(return_value=6)

    # Ancestral tokenizer splits "new_tok" into subword IDs [1, 3]
    # Expected: mean(E[1], E[3]) = mean([3,4,5], [9,10,11]) = [6, 7, 8]
    mock_ancestral = MagicMock()
    mock_ancestral.encode = MagicMock(return_value=[1, 3])

    original_auto = pve_module.AutoTokenizer
    pve_module.AutoTokenizer = MagicMock()
    pve_module.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_ancestral)

    try:
        expand_vocabulary_and_embeddings(model, tokenizer, ["new_tok"])
        weights = model.get_input_embeddings().weight.data
        assert weights.shape[0] == 6
        assert torch.allclose(weights[5], torch.tensor([6.0, 7.0, 8.0]))
    finally:
        pve_module.AutoTokenizer = original_auto


if __name__ == "__main__":
    test_expand_vocabulary_and_embeddings()
    print("test_pve.py passed")

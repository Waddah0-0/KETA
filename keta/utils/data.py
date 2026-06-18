import json
import logging
from typing import Dict, Any
from torch.utils.data import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


class KETADataset(Dataset):
    """Loads JSON, JSONL, CSV, Parquet, or plain text files."""

    def __init__(self, data_path: str, text_column: str = "text", target_column: str = None):
        self.data_path = data_path
        self.text_column = text_column
        self.target_column = target_column
        self.samples: list = []
        self._load()

    def _load(self):
        path = self.data_path
        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    self.samples = [json.loads(line) for line in f if line.strip()]

            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.samples = data if isinstance(data, list) else [data]

            elif path.endswith(".csv"):
                self.samples = pd.read_csv(path).to_dict(orient="records")

            elif path.endswith(".parquet"):
                try:
                    self.samples = pd.read_parquet(path).to_dict(orient="records")
                except ImportError:
                    logger.error("pyarrow or fastparquet needed for .parquet files")
                    self.samples = []

            else:
                with open(path, "r", encoding="utf-8") as f:
                    self.samples = [{self.text_column: line.strip()} for line in f if line.strip()]

            logger.info(f"Loaded {len(self.samples)} samples from {path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def get_formatting_prompts_fn(
    text_column: str = "text",
    system_prompt: str = "You are a helpful assistant speaking fluent GCC Arabic dialect.",
):
    """Returns a function that wraps text samples in the ChatML instruction template."""

    def fmt(examples):
        texts = examples[text_column]
        if isinstance(texts, str):
            texts = [texts]
        return {
            "text": [
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n"
                for t in texts
            ]
        }

    return fmt

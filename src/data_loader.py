import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import yaml
import os


class DataPreprocessor:
    def __init__(self, data_path, max_length=128):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    def preprocess(self):
        df = pd.read_csv(self.data_path)

        df["review"] = df["review"].apply(self.clean_text)

        texts = df["review"].tolist()
        labels = df["sentiment"].tolist()

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }

        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42
        )

        os.makedirs("data/processed", exist_ok=True)

        with open("data/processed/train.pkl", "wb") as f:
            pickle.dump(train_data, f)

        with open("data/processed/test.pkl", "wb") as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    processor = DataPreprocessor(
        config["data"]["dataset_path"],
        config["train"]["max_length"]
    )
    processor.preprocess()

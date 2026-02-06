import pickle
import mlflow
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch


class SentimentModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

    def train(self, train_data, val_data, config):
        training_args = TrainingArguments(
            output_dir="models/trained",
            evaluation_strategy="epoch",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            logging_dir="./logs",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer
        )

        mlflow.start_run()
        mlflow.log_params(config)

        trainer.train()
        eval_results = trainer.evaluate()

        mlflow.log_metrics(eval_results)

        self.model.save_pretrained("models/trained/model")
        self.tokenizer.save_pretrained("models/trained/model")

        mlflow.end_run()


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    with open("data/processed/train.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open("data/processed/test.pkl", "rb") as f:
        test_data = pickle.load(f)

    model = SentimentModel(params["train"]["model_name"])
    model.train(train_data, test_data, params["train"])

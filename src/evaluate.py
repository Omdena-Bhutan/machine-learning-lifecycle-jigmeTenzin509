import json
import pickle
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    with open("eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_metrics(metrics)
    return metrics

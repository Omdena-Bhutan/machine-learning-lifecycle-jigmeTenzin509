import mlflow.transformers

# Load best model
model = mlflow.transformers.load_model(
    "runs:/RUN_ID/model",
    return_type="huggingface"
)
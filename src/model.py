class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        # TODO: Load tokenizer from HuggingFace
        # TODO: Load pre-trained model
        # TODO: Freeze/unfreeze layers as needed
        pass
    
    def train(self, train_data, val_data, epochs=3):
        # TODO: Create Trainer with training arguments
        # TODO: Log hyperparameters to MLflow
        # TODO: Train and validate
        # TODO: Save model and log to MLflow
        pass
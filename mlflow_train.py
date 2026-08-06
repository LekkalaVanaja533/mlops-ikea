import mlflow
import mlflow.transformers
from transformers import pipeline

# Set experiment
mlflow.set_experiment("Sentiment Analysis Demo")

# Load the model from your local folder
classifier = pipeline(
    "sentiment-analysis",
    model="model",
    tokenizer="model"
)

with mlflow.start_run():

    # Test prediction
    text = "I love learning MLOps with Hugging Face."
    result = classifier(text)

    print(result)

    # Log parameter
    mlflow.log_param("model_name", "distilbert-base-uncased-finetuned-sst-2-english")

    # Log metric
    score = result[0]["score"]
    mlflow.log_metric("confidence_score", score)

    # Log the Hugging Face pipeline
    mlflow.transformers.log_model(
        transformers_model=classifier,
        artifact_path="sentiment_model"
    )

print("✅ Model logged successfully to MLflow")
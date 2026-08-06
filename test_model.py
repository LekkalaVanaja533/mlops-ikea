from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="model",
    tokenizer="model"
)

text = "I love learning MLOps."

result = classifier(text)

print(result)
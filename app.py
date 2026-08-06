from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="./model"
)

text = "I love MLOps"

result = classifier(text)

print(result)
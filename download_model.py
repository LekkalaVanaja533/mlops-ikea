from transformers import AutoTokenizer, AutoModel

# Hugging Face model name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Downloading {model_name}...")

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save locally
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Model downloaded successfully!")
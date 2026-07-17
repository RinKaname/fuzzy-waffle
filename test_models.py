from transformers import pipeline

models = [
    "cross-encoder/nli-deberta-v3-base",
    "typeform/distilbert-base-uncased-mnli",
    "facebook/bart-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
]

for model in models:
    try:
        classifier = pipeline("zero-shot-classification", model=model, device=-1)
        print(f"Model {model} loaded successfully.")
        break
    except Exception as e:
        print(f"Model {model} failed: {e}")

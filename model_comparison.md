# Model Comparison for Phase 1: Zero-Shot Intent Classification

This document outlines the performance characteristics and observations made when comparing different models for interpreting user intent (capabilities) from natural language probes in the PolicyProbe environment.

## The Challenge
The pipeline must accurately map a user request (e.g., "Find the product of 5.7(8) mentally using the Distributive Property") to a set of defined capabilities (e.g., "Conceptual Physics", "Formal Logic", "Marketing", "Prehistory") dynamically.

## 1. SentenceTransformers Embeddings (Cosine Similarity)
**Model:** `BAAI/bge-base-en-v1.5`
**Accuracy (Candidate 0):** ~75.44%
**Mechanism:** Computes the cosine similarity between the embedded request and the embedded capability labels.
**Strengths:**
- Very fast on CPU.
- Decent baseline without training.
**Weaknesses:**
- Struggles with nuanced semantic mapping. It relies heavily on lexical overlap or generic semantic closeness in the embedding space. A raw math equation may have an embedding that sits nowhere near the embedded phrase "Formal Logic", resulting in a low score and falling back to a misclassification.

## 2. Generic Zero-Shot Classification (Cross-Encoder / NLI)
**Model:** `cross-encoder/nli-deberta-v3-base`
**Accuracy (Candidate 0):** ~68.42%
**Mechanism:** Uses Natural Language Inference (NLI) to predict whether the premise (the probe) entails the hypothesis (e.g., "This text is about {capability}").
**Strengths:**
- Generally superior for zero-shot text classification tasks compared to raw embeddings because it explicitly models the relationship between the two strings.
**Weaknesses:**
- In this specific domain, generic NLI models perform poorly. The pre-training corpora (like MNLI or FEVER) do not prepare the model to logically link an abstract math question to the label "Formal Logic" or "Conceptual Physics" without explicit context. The lack of domain knowledge leads to poor confidence scores and frequent misclassifications.

## 3. Domain Fine-Tuned Zero-Shot Classification
**Model:** `RinKana/policyprobe-nli-model`
**Accuracy (Candidate 0):** ~84.21%
**Mechanism:** Similar to the generic NLI model, but fine-tuned specifically on the domain of the PolicyProbe competition.
**Strengths:**
- Significantly higher accuracy. The fine-tuning process allows the model to map the idiosyncratic vocabulary of the probes (e.g., math problems, specific scenarios) to the required capabilities effectively.
- Correctly identifies intents where generic models fail completely.
**Weaknesses:**
- Still misses a few edge cases (approx. 15% error rate on Candidate 0), but this is by far the most robust approach for the constraints provided (10 CPU cores, no GPU).

## Conclusion
For Phase 1 of the PolicyProbe architecture, deploying a domain-specific fine-tuned NLI classifier (like `RinKana/policyprobe-nli-model`) is the optimal strategy. It strictly adheres to the computational constraints while providing the highest accuracy for deterministic action matrix synthesis.

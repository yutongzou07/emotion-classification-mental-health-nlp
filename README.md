# emotion-classification-mental-health-nlp
# Emotion Classification for Mental-Health Support Text

**Deep learning pipeline** — LSTM and Transformer-based emotion classifiers with an LLM-driven supportive response generator, built for short free-form text common in digital mental-health platforms.

Model development (EmotionRNN, DistilBERT fine-tuning, LLM response integration) led by Yutong Zou and Sannidhi Patel.

[Open in Colab](#) *(update this link once your notebook is pushed)*

## Business Problem

Online mental-health platforms typically rely on basic positive/negative sentiment classification, which can't capture the nuance needed for effective user triage. This project builds a **six-class emotion classifier** (anger, fear, joy, love, sadness, surprise) for short text messages, with a downstream module that generates emotion-appropriate supportive responses — aimed at improving triage efficiency and response quality in digital therapy or support contexts.

## Approach

**1. Data preparation**
Built a unified vocabulary (76,134 unique tokens) from ~422K emotion-labeled sentences (Kaggle Sentiment and Emotion Analysis Dataset) plus a smaller 3,309-sample positive/negative sentiment dataset. Cleaned, tokenized, integer-encoded, and padded sequences (178 tokens for emotion, 70 for sentiment), then split 80/10/10 into train/validation/test sets.

**2. Model development — three classifiers, one generator**
- **EmotionRNN & SentimentRNN**: custom 2-layer LSTM architectures (PyTorch, hidden dim 256), built and trained from scratch — embedding → LSTM → final hidden state → linear classification head
- **DistilBERT**: fine-tuned as a pretrained Transformer baseline on a stratified subset, to benchmark custom LSTMs against modern pretrained architectures
- **Llama-3.2-1B (quantized)**: prompted with the predicted emotion label to generate a short, emotion-appropriate supportive reply

**3. Safety layer**
Built a multi-layer safety module for the deployed pipeline: incoming text is screened for self-harm-related language before any classification/generation occurs, generated output is re-screened before being shown to the user, and low-confidence predictions (<0.55) trigger a clarifying question instead of a generated response — all designed to keep the system conservative in high-risk or ambiguous cases.

## Key Findings

- **EmotionRNN reached ~94% test accuracy** on the six-way emotion task, closely matching validation performance (93.8–94.3%) — indicating strong generalization despite a large vocabulary and long sequences. Misclassifications mostly occurred between semantically adjacent emotions (e.g. fear vs. sadness), reflecting genuine ambiguity in language rather than model weakness.
- **SentimentRNN significantly underperformed (~51.7% accuracy)**, close to the majority-class baseline (~50.2%) — diagnosed as a data quality issue (noisy/inconsistent labels), not an architecture problem, since the same architecture performed strongly on the higher-quality emotion dataset.
- **DistilBERT reached ~78% validation accuracy** on a reduced 4,000-sample subset — a useful upper-bound reference, but didn't surpass the full-dataset LSTM, which had ~100x more training data. This highlighted a real-world tradeoff: pretrained Transformers can be more sample-efficient, but a well-tuned LSTM trained on a large, clean dataset can still win on both accuracy and deployment efficiency.
- The final deployed classifier is **EmotionRNN, not DistilBERT** — chosen deliberately for its full-dataset training, top accuracy, and low-latency inference, which mattered more for a real-time interactive use case than DistilBERT's marginally richer contextual representations.

## Recommendation / Business Impact

The modular design (lightweight classifier → safety screening → LLM-based generation) allows each component to be upgraded independently as better models become available, without re-architecting the system. For deployment in a real mental-health or customer-support context, this balances three competing priorities: **speed** (LSTM inference is cheap), **safety** (conservative fallback behavior under uncertainty or risk signals), and **quality** (LLM-generated responses feel more natural than templated replies).

## Tools

Python, PyTorch (custom LSTM architectures, training loops), HuggingFace Transformers (DistilBERT fine-tuning, Trainer API), Llama-3.2-1B (quantized, for response generation), pandas

## Notes

This was a team project for a graduate deep learning course. Dataset is the publicly available Kaggle Sentiment and Emotion Analysis Dataset; raw data files are not included in this repository.

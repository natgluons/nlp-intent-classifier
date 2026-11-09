# Customer Intent Classification for AI-Powered Support

A small, reproducible **data science case study** exploring whether traditional machine-learning models can classify customer support messages into intents that can inform support workflows.

> **Scope:** Educational proof of concept using an independent public dataset. This is **not** a Reylo AI / Euromedica integration or an evaluation on their real customer data.

## Business problem

Incoming customer messages have different purposes: complaints, refund requests, payment issues, and requests to talk to a person. Predicting the intent could help an AI support platform **suggest** routing decisions. Sensitive or uncertain cases should still be reviewed by a human.

## Dataset

[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), a hybrid-synthetic **English-language** dataset (~26.9k rows, 27 intents). This case study uses six existing labels:

| Intent | Meaning |
| --- | --- |
| `complaint` | Customer complaint |
| `contact_human_agent` | Request for a human agent |
| `payment_issue` | Problem with payment |
| `get_refund` | Request a refund |
| `cancel_order` | Cancel an order |
| `check_payment_methods` | Ask about accepted payment methods |

Only the `instruction` field is used as a model feature and `intent` as the label. `response` is **not** used; it represents an answer and could leak information unavailable when the customer message first arrives. The underlying dataset remains with its publisher and is not bundled in this repository. Review the [dataset's license](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) before redistribution or commercial use.

## Method

1. Load the dataset from Hugging Face with `datasets.load_dataset` and filter to six intents.
2. Check missing data, label balance, exact duplicate texts and sample messages.
3. Deduplicate normalized customer texts **before** splitting; discard texts with conflicting labels.
4. Split into stratified train / validation / test sets (70% / 15% / 15%).
5. Compare **TF-IDF + Multinomial Naive Bayes** and **TF-IDF + Logistic Regression** using validation macro-F1.
6. Evaluate the selected model **once** on test data; inspect per-class metrics, confusion matrix, and errors.
7. Show illustrative predictions and describe operational considerations.

Both models use scikit-learn `Pipeline`, so TF-IDF is fitted **only on the training set**. The test set is not used for model selection. The split is reproducible with a fixed random seed.

## Results

Run on 2026-11-09 with seed 42 (5,309 unique messages after cleaning; 3,716 / 796 / 797 train/validation/test).

| Model | Validation macro-F1 | Validation accuracy |
| --- | ---: | ---: |
| Multinomial Naive Bayes | 0.9989 | 0.9987 |
| Logistic Regression | 0.9989 | 0.9987 |
| Dummy (most frequent) | 0.0526 | 0.1872 |

Selected model: **TF-IDF + Multinomial Naive Bayes** (tied with Logistic Regression at four decimals; the notebook keeps the first row after sorting by validation macro-F1).

Final test accuracy: **0.9962** · Final test macro-F1: **0.9959**

**Insights:** Only 3 of 797 test messages were misclassified. The confusions were `contact_human_agent → complaint` (1), `payment_issue → cancel_order` (1), and `payment_issue → check_payment_methods` (1). These are short, underspecified messages such as "payment help" and "contactingassistant". Operationally, low-information messages and requests for a human should be routed conservatively rather than auto-answered. The near-perfect score reflects the synthetic, template-generated source and should not be read as production performance.

The notebook saves plots to `figures/` (`intent_distribution.png` and `confusion_matrix.png`); that folder is generated locally and ignored by git. A plain-text summary is in `report.txt`.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook intent_classification.ipynb
```

Run all notebook cells from top to bottom. **Internet access is required on first run** to download the public dataset from Hugging Face via `datasets.load_dataset`; later runs use the local Hugging Face cache. To execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace intent_classification.ipynb
```

## Possible application and limitations

- The output is an **intent suggestion**, not an autonomous action or a diagnosis. Messages about medical symptoms and adverse treatment reactions should have separate safety rules and human review.
- The source is synthetic English customer support text, not real Indonesian/WhatsApp clinic conversations. High held-out performance would **not** establish production performance in the Euromedica setting.
- Confidence scores from `predict_proba` are **not necessarily calibrated**. Do not use a confidence threshold as a production handoff policy without calibration and domain validation.
- Only six known classes are modeled. Other intents, code-switching and ambiguous/multiple-intent messages are **out of scope**; the model will otherwise assign one of the six labels.
- Random splitting of a template-generated dataset can overestimate generalization even after exact deduplication. A separate real-world, domain-specific evaluation set would be needed.

## Next steps

Obtain permitted, de-identified Indonesian clinic messages; define business-specific intent taxonomy (including booking and appointment changes); annotate examples with clear quality checks; evaluate multilingual and out-of-scope behavior; then test human-in-the-loop routing with operational metrics.

## Repository contents

```text
customer-intent-classification/
├── README.md
├── intent_classification.ipynb
├── requirements.txt
├── report.txt
├── .gitignore
└── figures/                 # generated locally, ignored by git
    ├── confusion_matrix.png
    └── intent_distribution.png
```

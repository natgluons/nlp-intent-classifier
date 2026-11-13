# Customer Intent Classification with NLP

Open-source data science project. This project tests whether simple, interpretable NLP models can classify customer support messages into common intents and what their mistakes reveal about message routing.

## Scope

This is a proof of concept using the public Bitext Customer Support dataset from Hugging Face. The dataset inspired this work and may be similar to what Euromedika needs, but this is not a Reylo AI or Euromedika integration and uses no private customer messages.

## Dataset and method

The source is English customer support text. Only the incoming message is used as a feature; the example response is not used. The six existing intents are complaint, contact_human_agent, payment_issue, get_refund, cancel_order, and check_payment_methods.

We deduplicate before splitting, use a 70/15/15 stratified split, compare TF-IDF + Multinomial Naive Bayes with TF-IDF + Logistic Regression on validation macro-F1, and evaluate the selected model once on the test set. Error analysis reviews the misclassified messages and their routing impact.

## Results

The selected model reached 0.9962 test accuracy and 0.9959 test macro-F1. Only 3 of 797 test messages were misclassified. The near-perfect score reflects synthetic, template-like data, not production performance.

## Business insights and action plan

Intent prediction should be a routing suggestion, not an autonomous reply. Complaints and requests for a human agent should be escalated, while payment, refund, and cancellation messages can go to dedicated queues.

Action plan: collect permitted, de-identified messages; define clinic-specific labels such as booking, appointment changes, pricing, billing, complaint, and symptom concern; build a held-out evaluation set; calibrate scores and add out-of-scope detection; run a human-in-the-loop pilot; monitor drift and retrain on reviewed examples.

## Run

Open intent_classification.ipynb in Jupyter and run all cells. Internet access is required on the first run to download the public dataset.

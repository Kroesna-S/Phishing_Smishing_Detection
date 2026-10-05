# Phishing and Scam Message Detection — SMS Smishing Classifier

A comparative analysis of Naive Bayes, SVM, and Random Forest for detecting smishing (SMS phishing) messages, with a risk-scoring layer, keyword flagging, and a small Flask app for real-time predictions. Built for ITM-360 (Artificial Intelligence), School of Digital Technologies, American University of Phnom Penh.
Advisor: Kuntha Pin
Group members: Sam Chankroesna (Team Leader), Chhoeung Narirath

**Notebook:** `Phishing_Smishing_Detection.ipynb`
**Repo:** `https://github.com/Kroesna-S/Phishing_Smishing_Detection`

## Tech stack

- Python 3 (Google Colab)
- pandas + NumPy (data loading and cleaning)
- NLTK (tokenization, stopwords, Porter stemming)
- scikit-learn (TF-IDF, Naive Bayes, SVM, Random Forest, GridSearchCV, metrics)
- imbalanced-learn (SMOTE inside a cross-validation pipeline)
- Matplotlib + Seaborn (visualization)
- Flask (real-time prediction interface and JSON API)

## Dataset

[SMS Smishing Collection](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set) on Kaggle (SMS Smish Collection v.1, Almeida & Gomez Hidalgo, 2011). Each line of the file is `label<TAB>message`.

| Property                | Value                          |
|-------------------------|--------------------------------|
| Messages (raw)          | 5,574                          |
| Exact duplicates removed | 403                           |
| Messages used           | 5,171 (4,518 ham / 653 smish)  |
| Class balance           | about 87% ham / 13% smish      |
| Source                  | UK / Singapore SMS, 2011       |

The dataset isn't included in this repo. Download it from the link above and keep the file named `SMSSmishCollection.txt`.

## Getting started

**Google Colab (recommended)**

1. Open `Phishing_Smishing_Detection.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `SMSSmishCollection.txt` to the Colab session.
3. Run the cells top to bottom. NLTK data is downloaded automatically on the first run.

**Locally**

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk matplotlib seaborn flask jupyter
jupyter notebook Phishing_Smishing_Detection.ipynb
```

**Running the web app**

After the notebook has run, it writes `app.py` and a `model_artifacts/` folder. From a terminal in the same folder:

```bash
python app.py     # then open http://127.0.0.1:5000
```

## How to use the app

**Web form** — paste an SMS message into the text box and click "Check message". The page shows the predicted label (HAM or SMISH), a 0–100 risk score with its category, and any flagged keywords.

**JSON API** — send a POST request to `/predict`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE entry! Txt WIN to 80088 now, cash prizes up to 5000 pounds"}'
```

```json
{
  "label": "smish",
  "risk_score": 90,
  "risk_category": "High",
  "flagged_keywords": ["free", "prize", "txt", "win"]
}
```

Risk categories: **High** (score 70+), **Medium** (30–69), **Low** (under 30).

## How the notebook works

**Step 0 — Setup** — imports, NLTK downloads, and a fixed random seed (42) so results are reproducible.

**Step 1 — Data loading & EDA** — loads the TSV with `quoting=3` so stray quote characters don't silently merge rows, checks class balance, removes exact duplicates (to prevent train/test leakage), and compares message length and raw word frequency by class. Smish messages are noticeably longer (median 148 vs. 53 characters).

**Step 2 — Text preprocessing** — lowercase, strip symbols (digits are kept since phone numbers and short codes are informative), tokenize, remove stopwords, and apply Porter stemming.

**Steps 3 & 4 — Feature engineering and splitting** — a stratified 80/10/10 train/validation/test split is done first, then a TF-IDF vectorizer (3,000 features) is fit on the training text only. SMOTE is applied only to training data, inside the cross-validation pipeline, so validation and test sets keep their real class distribution.

**Step 5 — Training & tuning** — three models, each wrapped in a `SMOTE → classifier` pipeline and tuned with 3-fold stratified cross-validation scored on smish-class F1. Learning curves are used as the loss-curve equivalent for these non-gradient-based models.

| Model                  | Tuned parameter                      | Best value   | Best CV F1 |
|------------------------|--------------------------------------|--------------|------------|
| Multinomial Naive Bayes | `alpha`                             | 0.5          | 0.8850     |
| SVM (linear kernel)    | `C`                                  | 1            | 0.9308     |
| Random Forest          | `n_estimators`, `max_depth`          | 100, None    | 0.9309     |

**Step 6 — Evaluation** — final metrics on the untouched test set, plus confusion matrices, ROC curves, and a comparison against the numeric targets set in the project proposal.

**Step 7 — Risk scoring & lexical analysis** — converts the smish probability into a 0–100 score, extracts the top-20 smish indicator words from each model (log-probability differences for Naive Bayes, coefficients for SVM, feature importances for Random Forest), and builds the end-to-end `predict_sms()` function.

**Step 8 — Flask interface** — saves the vectorizer, model, and keyword list with joblib and writes the standalone `app.py`.

## Results

Test set (518 messages, about 66 smish):

| Model          | Accuracy | Precision | Recall | F1     | AUC-ROC |
|----------------|----------|-----------|--------|--------|---------|
| Naive Bayes    | 0.9710   | 0.8493    | 0.9394 | 0.8921 | 0.9916  |
| SVM (linear)   | 0.9768   | 0.8857    | 0.9394 | 0.9118 | 0.9947  |
| Random Forest  | 0.9846   | 0.9677    | 0.9091 | 0.9375 | 0.9981  |

Random Forest had the highest test F1, so it is the deployed model.

Compared with the proposal's targets:

| Model          | Accuracy target | Recall target | F1 target | Met?                      |
|----------------|-----------------|---------------|-----------|---------------------------|
| Naive Bayes    | 0.96            | 0.88          | 0.90      | Accuracy, recall (F1 0.892 just under) |
| SVM (linear)   | 0.97            | 0.92          | 0.93      | Accuracy, recall (F1 0.912 under)      |
| Random Forest  | 0.96            | 0.90          | 0.91      | All three                 |

All three models exceeded the AUC-ROC target of 0.97.

## Key findings

1. **Random Forest is the strongest overall** — best accuracy, precision, F1, and AUC on the test set, with the trade-off of slightly lower recall than Naive Bayes and SVM.
2. **Recall vs. precision depends on the use case** — Naive Bayes and SVM catch more smish messages (recall 0.939) but flag more legitimate messages; a deployment that prioritizes missing nothing might prefer them.
3. **The three models agree on scam vocabulary** — `claim`, `prize`, `txt`, `mobil`, `www`, `150p`, `repli`, and `servic` appear as top smish indicators across all three extraction methods.
4. **Random Forest overfits the most, but still generalizes best** — train F1 0.999 vs. CV F1 0.918, the largest gap of the three, while its CV score is still the highest.

## Known limitations

- **Dated, region-specific data** — the dataset is English-language UK/Singapore SMS from 2011, built around premium-rate callbacks, ringtones, and text-to-win prizes. It won't transfer directly to Khmer-language or Telegram/Messenger scams.
- **Modern phishing is missed** — in the demo, "Your account has been suspended. Verify now at bit.ly/..." was classified as ham (risk 39, Medium). Shortened-link account-lockout phishing isn't represented in the training data.
- **Small test set** — with only 518 test messages (about 66 smish), a single misclassification moves recall by about 1.5 points, so the model rankings are not statistically firm.
- **Model chosen on the test set** — the deployed model is picked by test-set F1, so the reported test score is slightly optimistic. A cleaner approach would select on the validation set.
- **Word-level features only** — TF-IDF over words misses obfuscated spelling such as "Fr33". Character n-grams were left as future work.
- **SMOTE creates synthetic vectors, not sentences** — it interpolates TF-IDF vectors, so the extra minority samples don't correspond to readable text.
- **Notebook conclusion is slightly off** — the closing text says all three models cleared the proposal targets, but Naive Bayes and SVM missed their F1 targets (see the table above).

## Future work

- Collect a small labeled sample of real Khmer-language or Telegram scam messages to test generalization.
- Add character-level n-grams to catch obfuscation.
- Select the deployed model on validation data and add modern link-based phishing examples.

## References

1. T. A. Almeida, J. M. Gomez Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," *Proc. ACM Symp. Document Engineering (DocEng)*, 2011.
2. S. J. Delany, M. Buckley, and D. Greene, "SMS spam filtering: Methods and data," *Expert Systems with Applications*, vol. 39, no. 10, 2012.
3. M. Khonji, Y. Iraqi, and A. Jones, "Phishing detection: A literature survey," *IEEE Communications Surveys & Tutorials*, vol. 15, no. 4, 2013.

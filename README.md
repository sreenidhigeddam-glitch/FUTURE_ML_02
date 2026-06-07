# FUTURE_ML_02
# 🎫 Support Ticket Classification & Prioritization System
**Future Interns — ML Task 2 (2026)**

---

## 📁 Project Structure

```
support_ticket_classifier/
│
├── ticket_classifier.py      ← Main ML code (models, training, evaluation)
├── notebook.ipynb            ← Jupyter Notebook version
├── requirements.txt          ← All dependencies
│
├── data/
│   └── support_tickets.csv   ← Your dataset goes here (auto-generated if missing)
│
├── models/
│   ├── category_model.pkl    ← Saved category classifier
│   └── priority_model.pkl    ← Saved priority classifier
│
└── outputs/
    ├── confusion_matrices.png
    └── data_distribution.png
```

---

## ⚙️ How It Works

| Step | What Happens |
|------|-------------|
| 1 | Load support ticket dataset (CSV or synthetic) |
| 2 | Clean text — lowercase, remove stopwords, lemmatize |
| 3 | Convert text to TF-IDF features |
| 4 | Train Logistic Regression for Category prediction |
| 5 | Train Random Forest for Priority prediction |
| 6 | Evaluate with accuracy, precision, recall, F1 |
| 7 | Visualize confusion matrices |
| 8 | Save models → predict on new tickets |

---

## 🚀 Step-by-Step Execution

### OPTION A — Run as Python Script (Recommended for beginners)

```bash
# Step 1: Open a terminal / command prompt

# Step 2: Navigate to the project folder
cd support_ticket_classifier

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the classifier
python ticket_classifier.py
```

### OPTION B — Run in Jupyter Notebook

```bash
# Step 1: Install Jupyter (if not installed)
pip install jupyter

# Step 2: Launch Jupyter
jupyter notebook

# Step 3: Open notebook.ipynb in your browser
# Step 4: Run cells one by one (Shift + Enter)
```

### OPTION C — Run in Google Colab (No installation needed!)
1. Go to https://colab.research.google.com
2. Click **File → Upload Notebook** → upload `notebook.ipynb`
3. Upload `ticket_classifier.py` in the Files panel (left sidebar)
4. Run all cells

---

## 📂 Using a Real Dataset (Kaggle)

1. Download from: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
2. Rename or place the CSV as `data/support_tickets.csv`
3. The script auto-detects columns named `description`, `category`, `priority`
4. Run the script — it will use your real data automatically

---

## 📊 Output Example

```
📝 Ticket  : I was charged twice this month and need a refund!
   Category : Billing
   Priority : High
   Confidence: 94.2% → Billing
```

---

## 🧠 Models Used

| Task | Algorithm | Why |
|------|-----------|-----|
| Category | Logistic Regression (TF-IDF) | Fast, interpretable, great for text |
| Priority | Random Forest (TF-IDF) | Handles imbalanced classes well |

---

## 📈 Metrics Evaluated
- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted positives, how many were right
- **Recall** — Of actual positives, how many were found
- **F1 Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown per class

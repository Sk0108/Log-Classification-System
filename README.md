# 📝 Log Classification Pipeline

This project is a full pipeline for **log message classification**, combining unsupervised clustering, rule-based regex classification, and supervised machine learning. The goal is to automate the categorization of unstructured log data into meaningful classes for analysis and monitoring.

---

## 🚀 Features

- ✨ Text Embedding with `SentenceTransformer (all-MiniLM-L6-v2)`
- 🔍 Unsupervised log clustering using **DBSCAN**
- 🧩 Regex rule-based classification
- 🤖 Supervised classification using **Logistic Regression**
- 💾 Model persistence with **joblib**
- 📝 Detailed metrics & classification report

---

## 📂 Project Structure

```plaintext
.
├── dataset/
│   └── synthetic_logs.csv        # Input dataset containing synthetic logs
├── models/
│   └── log_classifier.joblib     # Saved Logistic Regression model
├── log_classification_pipeline.py # Main pipeline script
└── README.md                     # Project documentation (this file)
⚙️ Tech Stack
Python 3.7+

pandas - Data processing

scikit-learn - ML models (DBSCAN & Logistic Regression)

sentence-transformers - Text embeddings

regex - Rule-based classification

joblib - Model export

🧩 Pipeline Breakdown
1. Load Dataset
Read the logs from synthetic_logs.csv located in the /dataset folder.

Logs contain fields like log_message and other metadata.

2. Embedding with Sentence Transformer
Convert raw log messages into dense numerical vectors using all-MiniLM-L6-v2, a fast transformer-based model from the sentence-transformers library.

3. Unsupervised Clustering (DBSCAN)
Use DBSCAN clustering algorithm with cosine distance metric to group similar logs into clusters.

Helps to identify legacy logs or frequently recurring patterns.

4. Regex Classification
Apply rule-based regex patterns to quickly classify legacy or highly repetitive logs.

Example:

regex
Copy
Edit
r"User User\d+ logged (in|out)." -> "User Action"
r"Backup (started|ended) at .*" -> "System Notification"
r"Disk cleanup completed successfully." -> "System Notification"
5. Supervised Learning (Logistic Regression)
For logs not captured by regex rules, train a Logistic Regression classifier to classify them based on embeddings.

Split into training and test sets, evaluate using classification report.

6. Model Export
Save the trained Logistic Regression model to the /models directory using joblib.

🏃‍♂️ How to Run
🛠️ Prerequisites
Python 3.7 or higher

Recommended: Create and activate a virtual environment

📦 Install Dependencies
bash
Copy
Edit
pip install pandas scikit-learn sentence-transformers joblib
▶️ Execute Pipeline
bash
Copy
Edit
python log_classification_pipeline.py
This will:

Process the logs

Perform clustering & regex classification

Train Logistic Regression on remaining logs

Save the model to /models/log_classifier.joblib

📊 Output
Displays cluster statistics and classification reports

Saves a trained model (log_classifier.joblib)

Prints the performance of both regex and ML classifications

🚀 Future Improvements
🔄 Hyperparameter tuning for DBSCAN (e.g., eps, min_samples)

🏗️ Replace Logistic Regression with Random Forest, XGBoost, or Neural Networks

🛡️ Integrate FastAPI/Flask for API deployment

📈 Add anomaly detection for unusual log patterns
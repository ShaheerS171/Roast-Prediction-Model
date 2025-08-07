# 🔥 Roast Text Classifier

A machine learning-powered web app that classifies text as a **roast** or **normal** using NLP techniques and a trained SVM model. Built with **Python**, **scikit-learn**, and **Streamlit**, this app lets you paste any sentence and instantly get a classification along with prediction probabilities.

---

## 🚀 Features

- 🧠 Trained NLP pipeline with:
  - Lowercasing, punctuation removal
  - Stopword filtering
  - Emoji, number, and URL stripping
- 🗂️ TF-IDF Vectorization for feature extraction
- ✅ Multiple model training (Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost)
- 💯 Cross-validation accuracy scores
- 📊 Final model: Support Vector Classifier with high accuracy
- 🔮 Real-time prediction via a professional Streamlit interface
- 📈 Prediction probability chart for model transparency
- 💾 Model persistence using `joblib`

---

## 🧪 Sample Inputs

### 🔥 Roast:
> *"You're not stupid — you just have bad luck thinking."*
### ✅ Normal:
> *"Could you please help me with this task?"*

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| **Python** | Core language |
| **pandas, numpy** | Data manipulation |
| **scikit-learn** | Model training and evaluation |
| **XGBoost** | Advanced boosting model |
| **NLTK** | Text preprocessing |
| **Streamlit** | Web app frontend |
| **joblib** | Model serialization |

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/roast-classifier.git
cd roast-classifier
```

### 2. Set Up Virtual Environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate  # or env\Scripts\activate for Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt`, use:
```bash
pip install streamlit pandas numpy scikit-learn xgboost nltk joblib
```

### 4. Download NLTK Data (once)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
roast-classifier/
│
├── app.py                      # Streamlit app
├── svc_model.pkl               # Trained SVM model
├── tf_vectorizer.pkl           # TF-IDF vectorizer
├── label_encoder.pkl           # Label encoder
├── roast.csv                   # Raw dataset
├── README.md                   # This file
```

---

## 💡 Future Improvements

- 🎯 Add more labels (sarcasm, compliment, question, etc.)
- 📱 Mobile-responsive design for Streamlit
- 💬 Add a feedback loop to improve model over time
- 📁 Upload CSV for batch predictions
- 🧠 Use deep learning (BERT or RoBERTa) for better contextual understanding
- 🌐 Deploy on cloud (Streamlit Cloud, Hugging Face, or Heroku)

---

## 📜 License

This project is licensed under the MIT License.

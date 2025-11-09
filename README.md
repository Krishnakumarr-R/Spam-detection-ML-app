# 🛡️ Email Spam Classifier

An AI-powered email spam detection application built with **Streamlit** and **Machine Learning**. This application uses TF-IDF vectorization and Logistic Regression to classify emails as spam or legitimate (ham) with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎯 **High Accuracy** spam detection using Machine Learning
- 🎨 **Beautiful UI** with modern design and smooth animations
- 📧 **8 Pre-loaded Examples** (4 spam, 4 legitimate) for quick testing
- 📊 **Confidence Scores** and probability visualization
- 🚀 **Real-time Classification** with instant results
- 💡 **Educational Sidebar** with spam indicators and tips

## 🖼️ Screenshots

*Add screenshots of your app here*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
https://github.com/Krishnakumarr-R/Spam-detection-ML-app.git
cd Spam-detection-ML-app
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
The app will automatically open at `http://localhost:8501`

## 📦 Project Structure

```
spam_classifier_app/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── verify_model.py                 # Model verification script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── spam_classifier_model.joblib    # Trained model
├── tfidf_vectorizer.joblib        # TF-IDF vectorizer
├── spam_classifier_model.pkl      # Model backup (pickle)
└── tfidf_vectorizer.pkl           # Vectorizer backup (pickle)
```

## 🧠 How It Works

1. **Text Preprocessing**: Email content is cleaned and preprocessed
2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical features
3. **Classification**: Logistic Regression model predicts spam/ham
4. **Probability Calculation**: Returns confidence scores for both categories

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine Learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

## 📊 Model Performance

- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorization
- **Training Accuracy**: ~96-98%
- **Test Accuracy**: ~95-97%

## 🎮 Usage

1. **Manual Input**: Type or paste email content in the text area
2. **Use Examples**: Click any example button to auto-load sample emails
3. **Classify**: Click "Classify Email" to get results
4. **View Results**: See classification, confidence score, and probability distribution

## 🔄 Retraining the Model

If you want to retrain the model with your own dataset:

1. Place your `mail_data.csv` in the project folder (should have 'Message' and 'Category' columns)
2. Run the training script:
```bash
python complete_training_script.py
```
3. The script will create new model files automatically

## 📝 Dataset Format

Your CSV file should have this format:

| Category | Message |
|----------|---------|
| spam     | WINNER!! You won $1000... |
| ham      | Hi, let's meet for dinner... |

- **Category**: 'spam' or 'ham'
- **Message**: Email content text

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Inspired by email security and spam filtering research

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ **If you found this project helpful, please give it a star!** ⭐

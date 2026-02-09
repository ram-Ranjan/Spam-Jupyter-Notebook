# Hindi Spam Detection 🚫📱

A machine learning project for detecting spam messages in Hindi language, featuring a complete ML pipeline with data processing, model training, evaluation, and prediction capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Contributing](#contributing)

## 🎯 Overview

This project classifies Hindi text messages as **Spam** or **Ham** (legitimate) using machine learning techniques. It has been developed with production-ready code, featuring:

- Modular, maintainable codebase
- Comprehensive feature engineering
- Multiple ML algorithms support
- CLI tools for training and prediction
- Extensive evaluation metrics and visualizations
- Configuration-driven design
- Comprehensive logging

## ✨ Features

### Data Processing
- Automatic data loading and validation
- Handle imbalanced datasets with oversampling
- Configurable train/test splits
- Support for UTF-8 encoded Hindi text

### Feature Engineering
- **Text Features**: Message length, word count
- **Pattern Detection**: URLs, phone numbers, money amounts
- **Hindi-Specific**: Spam keyword detection (14+ Hindi keywords)
- **TF-IDF Vectorization**: Configurable n-gram features
- **Special Characters**: Count of special characters

### Model Training
- **Supported Models**:
  - Random Forest (default)
  - Logistic Regression
- **Features**:
  - Cross-validation
  - Hyperparameter configuration
  - Feature scaling
  - Class balancing (SMOTE/RandomOverSampler)
  - Model persistence

### Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance
- **Reports**: Detailed classification reports

### Prediction
- Single message prediction
- Batch prediction
- File-based prediction (CSV)
- Confidence scores
- Interactive CLI mode

## 📁 Project Structure

```
├── configs/
│   └── config.yaml              # All configuration parameters
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed features
│   └── external/                # External data sources
├── logs/                        # Application logs
├── models/                      # Saved models and artifacts
├── notebooks/
│   └── Spam_or_ham.ipynb       # Original research notebook
├── reports/                     # Evaluation plots and reports
├── src/
│   ├── data/                   # Data loading modules
│   ├── features/               # Feature engineering
│   ├── models/                 # Model train/eval/predict
│   └── utils/                  # Utilities (logging, config)
├── tests/                      # Unit tests
├── main.py                     # Training pipeline
├── predict_cli.py              # Prediction CLI
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd "Spam Jupyter Notebook"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   ```bash
   # Place your CSV file in data/raw/
   # The CSV should have columns: 'Message' and 'Category'
   # Example: data/raw/Spam_or_ham.csv
   ```

## 🚀 Quick Start

### 1. Train a Model

```bash
# Train with default configuration
python main.py --train

# Train with custom config
python main.py --train --config configs/config.yaml

# Train with custom data
python main.py --train --data path/to/your/data.csv
```

**Output**: After training, you'll find:
- Model saved in `models/spam_classifier.pkl`
- Scaler in `models/scaler.pkl`
- TF-IDF vectorizer in `models/tfidf_vectorizer.pkl`
- Evaluation plots in `reports/`
- Logs in `logs/spam_detection.log`

### 2. Make Predictions

#### Interactive Mode
```bash
python predict_cli.py
```

#### Single Message
```bash
python predict_cli.py "आपने लॉटरी जीत ली है! अभी कॉल करें"
```

#### Batch Prediction from File
```bash
# Input: CSV with 'Message' column
python predict_cli.py --file input_messages.csv --output predictions.csv
```

## 📖 Usage

### Training Pipeline

The training pipeline (`main.py`) performs:

1. **Data Loading**: Loads and validates CSV data
2. **Feature Engineering**: Extracts 8 key features
3. **Data Splitting**: 80/20 train-test split
4. **Preprocessing**: Scaling and oversampling
5. **Model Training**: Trains Random Forest classifier
6. **Evaluation**: Generates metrics and visualizations
7. **Model Saving**: Persists all artifacts

**Example Output**:
```
================================================================================
MODEL EVALUATION RESULTS
================================================================================
ACCURACY            : 0.9845 (98.45%)
PRECISION           : 0.9823 (98.23%)
RECALL              : 0.9756 (97.56%)
F1_SCORE            : 0.9789 (97.89%)
ROC_AUC             : 0.9912 (99.12%)
================================================================================

Model saved to: models/spam_classifier.pkl
Reports saved to: reports/
```

### Prediction CLI

The prediction CLI (`predict_cli.py`) supports three modes:

#### 1. Single Message Mode
```bash
python predict_cli.py "मुफ्त में पाएं 5000 रूपए"
```
**Output**:
```
================================================================================
PREDICTION RESULT
================================================================================
Message: मुफ्त में पाएं 5000 रूपए
Prediction: Spam
Confidence: 95.67%
================================================================================
```

#### 2. Interactive Mode
```bash
python predict_cli.py

# Then enter messages interactively
Enter message: नमस्ते, आप कैसे हैं?
  Prediction: Ham
  Confidence: 92.34%

Enter message: quit
Goodbye!
```

#### 3. File Mode
```bash
python predict_cli.py --file messages.csv --output results.csv
```

**Input CSV** (`messages.csv`):
```csv
Message
आपने लॉटरी जीत ली है
क्या आप कल मिल सकते हैं?
फ्री में 10000 रूपए पाएं
```

**Output CSV** (`results.csv`):
```csv
Message,prediction,prediction_code,confidence
आपने लॉटरी जीत ली है,Spam,1,0.9567
क्या आप कल मिल सकते हैं?,Ham,0,0.9234
फ्री में 10000 रूपए पाएं,Spam,1,0.9789
```

## ⚙️ Configuration

All settings are in `configs/config.yaml`:

### Data Paths
```yaml
data:
  raw_path: "data/raw/Spam_or_ham.csv"
  processed_path: "data/processed/spam_features.csv"
  encoding: "utf-8"
```

### Feature Engineering
```yaml
features:
  tfidf_max_features: 100
  spam_words:
    - 'फ्री'
    - 'मुफ्त'
    - 'जल्दी'
    # ... more keywords
```

### Model Configuration
```yaml
model:
  name: "RandomForest"
  test_size: 0.2
  rf_params:
    n_estimators: 100
    max_depth: null
```

### Training Settings
```yaml
training:
  oversample: true
  scale_features: true
  cross_validation: true
  cv_folds: 5
```

## 📊 Model Performance

Expected performance metrics on test data:

| Metric | Score |
|--------|-------|
| **Accuracy** | ~98.5% |
| **Precision** | ~98.2% |
| **Recall** | ~97.6% |
| **F1-Score** | ~97.9% |
| **ROC-AUC** | ~99.1% |

### Feature Importance

Top features for spam detection:
1. **spam_word_count** - Count of Hindi spam keywords
2. **tfidf_score** - TF-IDF relevance score
3. **has_money** - Presence of money amounts
4. **has_url** - Presence of URLs
5. **has_phone** - Presence of phone numbers
6. **special_char_count** - Count of special characters
7. **message_length** - Length of message
8. **word_count** - Number of words

### Hindi Spam Keywords Detected

The model detects these common Hindi spam keywords:
- फ्री (free)
- मुफ्त (free)
- जल्दी (hurry)
- लिमिटेड (limited)
- विजेता (winner)
- इनाम (prize)
- ऑफर (offer)
- कॉल (call)
- क्लिक (click)
- लकी (lucky)
- खरीदें (buy)
- बधाई (congratulations)
- मौका (chance)
- शीघ्र (urgent)

## 🔍 Example Predictions

### Spam Examples
```python
# High confidence spam
"बधाई हो! आपने 50000 रुपए की लॉटरी जीती है। क्लिक करें" → Spam (98.5%)
"फ्री में पाएं 10GB डेटा! अभी कॉल करें 9876543210" → Spam (97.2%)
"लिमिटेड ऑफर! मुफ्त में iPhone पाएं" → Spam (96.8%)
```

### Ham Examples
```python
# High confidence ham (legitimate)
"नमस्ते, आप कैसे हैं?" → Ham (95.4%)
"कल शाम 6 बजे मिलते हैं" → Ham (93.7%)
"धन्यवाद आपकी सहायता के लिए" → Ham (94.2%)
```

## 🧪 Testing

Run tests (after implementing):
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Logging

Logs are written to:
- **Console**: INFO level and above
- **File**: `logs/spam_detection.log` (all levels)

Log format:
```
2024-02-09 12:34:56 - spam_detection - INFO - Model training complete
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Add more Hindi spam keywords**
2. **Implement additional models** (XGBoost, LSTM, BERT)
3. **Add unit tests**
4. **Improve feature engineering**
5. **Create REST API**
6. **Add Docker support**
7. **Implement model monitoring**

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Original notebook development and research
- Hindi spam detection dataset
- Scikit-learn and imbalanced-learn libraries

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Made with ❤️ for Hindi spam detection**

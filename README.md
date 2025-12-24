# Customer Voice Matters: Big Data Analysis of Amazon Product Reviews

A comprehensive sentiment analysis project that analyzes Amazon product reviews using machine learning techniques to classify customer feedback as positive or negative.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

This project performs sentiment analysis on Amazon product reviews to understand customer satisfaction. The analysis uses machine learning techniques to classify reviews as positive or negative, helping businesses gain insights from customer feedback at scale.

**Course**: AIT 622 (Deep Learning 1)  
**Group**: Group 3

## ✨ Features

- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **TF-IDF Vectorization**: Converts text data into numerical features
- **Logistic Regression Model**: Binary classification for sentiment analysis
- **Model Evaluation**: 
  - Classification report (precision, recall, F1-score)
  - Confusion matrix visualization
  - ROC curve and AUC score
- **Big Data Handling**: Efficient sampling and processing of large datasets

## 📁 Project Structure

```
Customer-Voice-Matters-Big-Data-Analysis-of-Amazon-Product-Reviews-main/
│
├── AIT 622(DL 1) Group 3 Code.html    # Jupyter notebook (exported as HTML)
├── AIT 622(DL1)-Group 3- Presentation.pptx  # Project presentation
├── README.md                           # This file
├── training.csv                        # Training dataset (required)
└── test.csv                           # Test dataset (required)
```

## 📦 Requirements

The project requires the following Python libraries:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
  - `TfidfVectorizer` - Text vectorization
  - `LogisticRegression` - Classification model
  - `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve` - Evaluation metrics
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `re` - Regular expressions (built-in)
- `nltk` - Natural Language Toolkit (if used for advanced text processing)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Customer-Voice-Matters-Big-Data-Analysis-of-Amazon-Product-Reviews-main
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk
   ```

3. **Prepare your datasets**:
   - Ensure you have `training.csv` and `test.csv` files in the project directory
   - Both files should have the following structure (no header):
     - Column 1: `label` (sentiment label: 1 for negative, 2 for positive)
     - Column 2: `title` (review title)
     - Column 3: `review` (review text)

## 💻 Usage

### Running the Analysis

1. **Open the notebook**:
   - If you have the original `.ipynb` file, open it in Jupyter Notebook or JupyterLab
   - Alternatively, view the exported HTML file: `AIT 622(DL 1) Group 3 Code.html`

2. **Execute the cells in order**:
   - The notebook will:
     - Load and preprocess the data
     - Clean the text (remove special characters, normalize, etc.)
     - Sample the datasets (50,000 training samples, 10,000 test samples)
     - Vectorize text using TF-IDF
     - Train a Logistic Regression model
     - Evaluate the model and display results

### Expected Output

The analysis will generate:
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Visual representation of prediction accuracy
- **ROC Curve**: Receiver Operating Characteristic curve with AUC score

## 🔬 Methodology

### Data Preprocessing

1. **Text Cleaning**:
   - Remove special characters and HTML tags
   - Convert to lowercase
   - Remove extra whitespace
   - Combine title and review text

2. **Sampling**:
   - Training set: Up to 50,000 samples
   - Test set: Up to 10,000 samples
   - Random sampling with fixed seed (42) for reproducibility

### Feature Engineering

- **TF-IDF Vectorization**:
  - Removes English stop words
  - Extracts top 5,000 features
  - Converts text to numerical vectors

### Model Training

- **Algorithm**: Logistic Regression
- **Task**: Binary classification (positive/negative sentiment)
- **Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC

## 📊 Results

The model achieves classification of Amazon product reviews with evaluation metrics including:
- Accuracy
- Precision and Recall for each class
- F1-score
- ROC-AUC score

*Note: Specific results may vary based on the dataset used. Refer to the notebook output for detailed metrics.*

## 👥 Contributors

- **Course**: AIT 622 (Deep Learning 1)
- **Group**: Group 3

## 📄 License

This project is created for educational purposes as part of the AIT 622 course. Please refer to your institution's academic policies regarding code sharing and usage.

## 📝 Notes

- The datasets (`training.csv` and `test.csv`) are not included in this repository. You will need to provide your own datasets following the specified format.
- The HTML file contains the complete notebook output and can be viewed in any web browser.
- For best results, ensure your datasets are properly formatted and contain sufficient samples for training and testing.

## 🔗 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [TF-IDF Vectorization Guide](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

**Disclaimer**: This project is for educational purposes. Ensure you have proper permissions and comply with data usage policies when working with Amazon review data.

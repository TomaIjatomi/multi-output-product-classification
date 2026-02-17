# Deep Learning Framework for Multi-Output Product Categorization

Multi-output LSTM-based text classification for large-scale e-commerce product categorization under class imbalance.

---

## Project Overview

In the modern e-commerce landscape, accurate product categorization is essential for searchability, SEO, and user experience. This project develops a machine learning framework to automate the classification of products into multiple categories simultaneously.

Using a dataset of over 229,000 unique products, the system predicts four distinct targets from a single text input:

- Top Category (15 classes)  
- Bottom Category (2,609 classes)  
- Primary Color  
- Secondary Color  

The study compares classical machine learning baselines with deep neural architectures under severe class imbalance and high-dimensional textual inputs.

---

## Dataset

The dataset is proprietary data provided by Etsy, a global e-commerce marketplace for unique and creative goods. The dataset contains over 229,000 unique products with 26 features per product including:

- Product Titles and Descriptions
- Product Images encoded in Bytes
- Metadata: Tags, materials, occasion, and more.

The dataset exhibits significant class imbalance, particularly in the bottom category (2,609 classes) and color labels, with a pronounced long-tail distribution. Product titles are generally short (most under 100 characters), increasing the difficulty of fine-grained classification.

Due to its proprietary nature, the dataset cannot be publicly shared.

---

## Technical Architecture

The framework employs a combination of traditional machine learning and advanced deep learning techniques:

### Data Preprocessing

- Text Cleaning: Lowercasing, removal of HTML entities, URLs, punctuation, and stopwords. 
- NLP Techniques: Tokenization and Lemmatization using NLTK. 
- Feature Engineering: Implementation of TF-IDF for baseline models and GloVe Word Embeddings for deep learning. 

### Models Explored

- Baseline Models: Naïve Bayes, Support Vector Machines (SVM), and Random Forests. 
- Deep Learning: An LSTM (Long Short-Term Memory) model incorporated into a MultiOutput framework to handle simultaneous classification tasks. 
- Image Processing: MobileNetV3Small for visual feature analysis. 

### Feature Reduction

- PCA (Principal Component Analysis): Used for visualization and revealing connections between content features and product categories.

## Tech Stack

##### Python | TensorFlow | Keras | Scikit-learn | NLTK | Pandas | Matplotlib
---

## Results

The models were evaluated using weighted F1-scores across the four targets. While baseline models provided initial insights, the deep learning approach (LSTM with GloVe) offered enhanced performance in capturing the nuances of e-commerce text data. Fine-grained category prediction remains constrained by extreme class imbalance.


### Final Test F1 Scores:
- Top Category: 83%  
- Bottom Category: 47%  
- Primary Color: 46%  
- Secondary Color: 27%  

---

## Future Work
 
- Transformer Models: Replace the current LSTM architecture with Transformer-based models like BERT or RoBERTa for better context understanding.
- Multimodal Integration: integrate visual features from product images (extracted via MobileNetV3Small) with textual features into a single Late Fusion model to improve categorization accuracy, especially for color identification.  

---

## Repository Contents

- `Multi-Output Product Categorization Report.pdf` – Full technical report  
- `Multi-Output Product Categorization Notebook.ipynb` – Implementation notebook  
- `README.md`
---

## Academic Context

This work was completed as part of my MSc. in Artificial Intelligence program at Dublin City University (DCU).
Developed for the CA684 Machine Learning module (Spring 2024)

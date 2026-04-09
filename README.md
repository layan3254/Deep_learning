
# Amazon Fine Food Sentiment Analysis 
## 📌 Project Overview
This project builds a robust sentiment analysis system that predicts whether a customer review is Positive or Negative. Unlike traditional models that only look at text, this project implements a Hybrid Approach, combining Natural Language Processing (NLP) with User/Product behavioral statistics.

## 🛠️ Technical Stack
**Language**: Python

**Data Manipulation**: Pandas, NumPy

**NLP**: Scikit-learn (TF-IDF Vectorizer)

**Modeling**: TensorFlow/Keras (Artificial Neural Networks), Scikit-learn (Logistic Regression)

**Visualization**: Matplotlib

## 🚀 Key Features & Implementation Details
**1. Data Engineering**
The real power of this model comes from Feature Engineering. We didn't just use text; we extracted:

**Behavioral Metadata**: Calculated user_avg_score and product_avg_score. This helps the model understand if a user is naturally "strict" or if a product is "consistently good".

**Text Metrics**: Included text_len as a feature to capture the depth of the review.

**2. NLP Pipeline**
**Vectorization**: Used TF-IDF with a limit of 1,000 features.

**N-grams**: Implemented Unigrams and Bigrams (ngram_range=1,2) to capture context like "not good" instead of just "good".

**3. Model Architecture (ANN)**
We developed a Deep Multi-Layer Perceptron (MLP) with the following layers to ensure stability:

**Input Layer**: 1003 features (1000 TF-IDF + 3 Metadata).

**Regularization**: Used L2 Regularization and Dropout (0.3) to prevent overfitting.

**Normalization**: Integrated BatchNormalization layers to speed up training and ensure weight stability.

**Output**: Sigmoid activation for Binary Classification.

**4. Training Optimization**
To reach peak performance, we implemented:

**EarlyStopping**: Monitored val_loss with a patience of 7 epochs to restore the best weights.

**Learning Rate Scheduler**: Used ReduceLROnPlateau to decrease the learning rate when the model hits a plateau.

## 📊 Experimental Results
We compared a Baseline model with our Deep Learning model:

**Baseline (Logistic Regression)**: Achieved an impressive ~96% Accuracy due to the strong linear relationship of our engineered features.

**Deep Learning (ANN)**: Achieved ~94-95% Accuracy, providing a sophisticated alternative for non-linear patterns.

## 🔮 Inference Example
The project includes a dedicated pipeline for Real-world testing. You can input any raw text, and the model will:

Transform text via TF-IDF.

Fuse it with metadata.

Predict the sentiment probability.


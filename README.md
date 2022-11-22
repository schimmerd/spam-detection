# SMS spam detection model

# !! Work in progress !!

This repository contains a spam detector program in Python which classifies given SMS as spam or ham using the Naive Bayes approach.
It works with both Bag-of-words and TFIDF features.

# Requirement

```shell
# used for stemming and stopword removal
pip install nltk
```

# Training data 
SMSSpamCollection.csv needs to be stored at /data to ensure that the main code is working properly.
Source: UCI Machine Learning Repository
# Usage 

```python
from SpamClassifier import NaiveBayes
from utils import preprocess, test_and_train_split, classification_report


path_to_data = "data/SMSSpamCollection.csv"
x_label, y_label = "sms", "label"
method = "tfidf"  # or bow

sms_data = preprocess(path_to_data, header=["spam", "label"])

X_train, X_test, y_test = test_and_train_split(sms_data, x_label=x_label, y_label=y_label)

spam_classifier = NaiveBayes(method=method, verbose=False)
spam_classifier.fit(X_train, x_label=x_label, y_label=y_label)

y_predictions = spam_classifier.predict(X_test)

classification_report(y_predictions, y_test, method, labels=["spam", "ham"], verbose=True)
```
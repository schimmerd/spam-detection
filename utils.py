from typing import List, Dict, Union

import string
import random
import csv
import math

from itertools import chain, islice
from functools import reduce
from collections import Counter

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


ps = PorterStemmer()
sw = stopwords.words('english')


def test_and_train_split(data: List, x_label: str, y_label: str, train_size: float = 0.8, test_size: float = 0.2):
    """ Split data into train and test sets
    :param data: data set
    :param train_size: size of the training set
    :param test_size: size of the test set
    :return: X_train, X_test, y_test
    """

    # shuffle
    shuffled = random.sample(data, len(data))

    train_and_test = [
        list(
            islice(iter(shuffled), elm)) for elm in [
            round((len(data) * train_size)),
            round((len(data) * test_size))
        ]
    ]
    x_train = train_and_test[0]
    x_test = list(map(lambda x: x.get(x_label), train_and_test[1]))
    y_test_label = list(map(lambda x: x.get(y_label), train_and_test[1]))

    return x_train, x_test, y_test_label


def clean_text(text: str, stop_words: Union[List, None], stem_words: bool) -> List:
    processed_text = text.replace(r'\W+', '').replace(r'\s+', '').strip()
    processed_text = processed_text.lower()
    processed_text = processed_text.split()  # tokenization

    if stop_words:
        processed_text = [
            token for token in processed_text if token not in stop_words
        ]
    if stem_words:
        processed_text = [
            ps.stem(token) for token in processed_text
        ]
    processed_text = [
        token.translate(str.maketrans('', '', string.punctuation)) for token in processed_text
    ]
    return [
        token for token in processed_text if token != ''
    ]


def preprocess(file_path: str, header: List = None, stop_words: Union[List, None] = sw, stem_words: bool = True) -> List:
    """ Load CSV data
    :param file_path: Path to csv file
    :param header: None or List of names
    :param stop_words: word to be removed from text
    :param stem_words: True or False if stemming should be used
    :return: List of rows
    """

    if header is None:
        header = []

    csv_data = list()
    with open(file_path, "r") as csvfile:
        csv_reader = csv.DictReader(csvfile, fieldnames=header, dialect="excel-tab")
        for row in csv_reader.reader:
            # replace label with 0 (ham) or 1 (spam)
            y_data = 1 if row[0].lower() == "spam" else 0
            x_data = row[1]
            preprocessed_data = clean_text(x_data, stop_words, stem_words)
            if len(preprocessed_data) != 0:
                csv_data.append(dict(label=y_data, sms=preprocessed_data))
    return csv_data


def _print_matrix(matrix: List[List], labels: List[str]):
    """ Print Confusion matrix
    :param matrix: Predictions in form [[10, 20],[2,23],...]
    :param labels: Classification output labels
    """

    print("Confusion Matrix:")
    l = max(
        reduce(lambda n, x: len("%s" % x) if n < len("%s" % x) else n, [0] + list(chain(matrix))),
        reduce(lambda n, x: len("%s" % x) if n < len("%s" % x) else n, [0] + list(chain(matrix)))
    )

    print("\t", eval("\"%%%is\"%%\"%s\"" % (l, "")), end=" ")

    for column in labels:
        print(eval("\"%%%is\"%%\"%s\"" % (l, column)), end=" ")
    print()

    i = -0
    for row in matrix:
        print("\t", eval("\t\"%%%is\"%%\"%s\"" % (l, labels[i] if i >= 0 else "")), end=" ")
        i += 1
        for column in row:
            print(eval("\"%%%is\"%%%s" % (l, column)), end=" ")
        print()


def classification_report(predictions: Dict, y_test: List, method: str, labels: List, verbose: bool = True):
    """ Print model classification report
    :param labels: Classification output labels
    :param y_test: test output label
    :param method: model method
    :param predictions: y_prediction
    :param verbose: True/False print logs
    """

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(y_test)):
        true_pos += int(y_test[i] == 1 and predictions[i] == 1)
        true_neg += int(y_test[i] == 0 and predictions[i] == 0)
        false_pos += int(y_test[i] == 0 and predictions[i] == 1)
        false_neg += int(y_test[i] == 1 and predictions[i] == 0)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fall_out = false_pos / (false_pos + true_neg)
    mcc = (true_pos * true_neg - false_pos * false_neg) / math.sqrt(
        (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg))
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    counter = Counter(y_test)

    if verbose:
        print()
        print("Classification report for:", method.upper())
        print()
        print("Total test messages:", len(y_test))
        print()
        _print_matrix([[true_pos, false_neg], [false_pos, true_neg]], labels=labels)
        print()
        print("Precision:\t", round(precision, 2))
        print("Recall:\t\t", round(recall, 2))
        print("Fall-Out:\t", round(fall_out, 2))
        print("MCC:\t\t", round(mcc, 2))
        print("Accuracy:\t", round(accuracy, 2))

        print()
        print("Ham messages in test (%):", (counter.get(0) * 100) / len(y_test))
        print("Spam messages in test (%):", (counter.get(1) * 100) / len(y_test))
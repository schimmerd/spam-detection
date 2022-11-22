from typing import List, Dict
from itertools import chain
from collections import defaultdict

import math


class NaiveBayes:

    def __init__(self, method: str, smoothing_num: int = 1, smoothing_den: int = 2, verbose: bool = False):
        self.smoothing_num = smoothing_num  # Laplace smoothing£
        self.smoothing_den = smoothing_den
        self.method = method
        self.verbose = verbose

        self._vector_list_spam: List = None
        self._vector_list_ham: List = None
        self._index_spam: Dict = None
        self._index_ham: Dict = None
        self._sum_of_spam: float = None
        self._sum_of_ham: float = None
        self._p_spam: float = None
        self._p_ham: float = None

    @staticmethod
    def _create_index(vocab: List) -> Dict:
        index_dict = dict()
        i = 0
        for word in vocab:
            index_dict[word] = i
            i += 1
        return index_dict

    @staticmethod
    def _count_frequency(sentences: List, vocab: List) -> Dict:
        count_dict = dict()
        for word in vocab:
            count_dict[word] = 0
            for sent in sentences:
                if word in sent:
                    count_dict[word] += 1
        return count_dict

    @staticmethod
    def _bow_vector(sentences: List, index_dict: Dict, vocab: List) -> List:
        bow_list = list()
        for sent in sentences:
            count_dict = defaultdict(int)
            vec = [float(0)] * len(vocab)
            for word in sent:
                count_dict[word] += 1.0
            for word, count in count_dict.items():
                vec[index_dict[word]] = count
            bow_list.append(vec)
        return bow_list

    def _tfidf_vector(self, sentences: List, vocab: List, count_dict: Dict, index_dict: Dict) -> List:
        vector_list = list()

        for sentence in sentences:
            vector: List = self._compute_tfidf(vocab, sentence, count_dict, index_dict, len(sentences))
            vector_list.append(vector)

        return vector_list

    @staticmethod
    def _compute_tf(sentence: List, word: str) -> float:
        N = len(sentence)
        occ = len([token for token in sentence if token == word])
        return occ / N

    @staticmethod
    def _compute_idf(count_dict: Dict, word: str, no_of_sentences: int) -> float:
        try:
            word_occ = count_dict[word] + 1
        except KeyError:
            word_occ = 1
        return math.log(no_of_sentences / word_occ)

    def _compute_tfidf(self, vocab: List, sentence: List, count_dict: Dict, index_dict: Dict,
                       no_of_sentences: int) -> List:
        tfidf_vec: List = [float(0)] * len(vocab)
        for word in sentence:
            tf = self._compute_tf(sentence, word)
            idf = self._compute_idf(count_dict, word, no_of_sentences)
            value = tf * idf
            tfidf_vec[index_dict[word]] = value
        return tfidf_vec

    @staticmethod
    def _vocabulary(spam, ham):
        return set(list(chain.from_iterable(spam))), set(list(chain.from_iterable(ham)))

    def fit(self, x_train: List, x_label: str, y_label: str):
        """ Fit the classification model
        :param x_train: training set
        :param x_label: data label name
        :param y_label: output label name
        """

        X_train_ham = list(
            map(lambda y: y.get(x_label),
                filter(lambda x: x.get(y_label) == 0, x_train))
        )
        X_train_spam = list(
            map(lambda y: y.get(x_label),
                filter(lambda x: x.get(y_label) == 1, x_train))
        )

        vocab_spam, vocab_ham = self._vocabulary(X_train_spam, X_train_ham)

        if self.verbose:
            print("Total number of Ham messages in train:", len(X_train_ham))
            print("Total number of Spam messages in train:", len(X_train_spam))
            print("Size of ham vocab:", len(vocab_ham))
            print("Size of spam vocab:", len(vocab_spam))

        self._index_spam = self._create_index(vocab_spam)
        self._index_ham = self._create_index(vocab_ham)

        if self.method == "tfidf":
            count_spam = self._count_frequency(X_train_spam, vocab_spam)
            count_ham = self._count_frequency(X_train_ham, vocab_ham)

            self._vector_list_spam = self._tfidf_vector(X_train_spam, vocab_spam, count_spam, self._index_spam)
            self._vector_list_ham = self._tfidf_vector(X_train_ham, vocab_ham, count_ham, self._index_ham)

            self._sum_of_spam = sum(list(chain.from_iterable(self._vector_list_spam)))
            self._sum_of_ham = sum(list(chain.from_iterable(self._vector_list_ham)))

        if self.method == "bow":
            self._vector_list_spam = self._bow_vector(X_train_spam, self._index_spam, vocab_spam)
            self._vector_list_ham = self._bow_vector(X_train_ham, self._index_ham, vocab_ham)

        self._p_spam = len(X_train_spam) / (len(X_train_spam) + (len(X_train_ham)))
        self._p_ham = len(X_train_ham) / (len(X_train_ham) + (len(X_train_spam)))

    def classifier(self, message: List) -> bool:
        """ Classifies a given message either as spam (1) or ham (0)
        :param message: List of strings
        :return: 0 (ham) or 1 (spam)
        """

        p_w_spam, p_w_ham = 0.0, 0.0
        for word in message:

            if self.method.lower() == "tfidf":

                try:
                    p_w_spam += math.log((sum(list(vector[self._index_spam[word]]
                                                   for vector in self._vector_list_spam)
                                              ) + self.smoothing_num) / self._sum_of_spam + self.smoothing_den)
                except KeyError:
                    p_w_spam += math.log(1 / (len(self._vector_list_spam) + self.smoothing_den))

                try:
                    p_w_ham += math.log((sum(list(vector[self._index_ham[word]]
                                                  for vector in self._vector_list_ham)
                                             ) + self.smoothing_num) / self._sum_of_ham + self.smoothing_den)
                except KeyError:
                    p_w_ham += math.log(1 / (len(self._vector_list_ham) + self.smoothing_den))

            if self.method.lower() == "bow":
                try:
                    p_w_spam += math.log((sum(list(vector[self._index_spam[word]]
                                                   for vector in self._vector_list_spam)
                                              ) + self.smoothing_num))
                except KeyError:
                    p_w_spam += math.log(1 / (len(self._vector_list_spam) + self.smoothing_den))

                try:
                    p_w_ham += math.log((sum(list(vector[self._index_ham[word]]
                                                  for vector in self._vector_list_ham)
                                             ) + self.smoothing_num))
                except KeyError:
                    p_w_ham += math.log(1 / (len(self._vector_list_ham) + self.smoothing_den))

            p_w_spam += math.log(self._p_spam)
            p_w_spam += math.log(self._p_ham)

        if p_w_spam >= p_w_ham:
            return 1
        return 0

    def predict(self, x_test: List) -> Dict:
        result = dict()
        for i, message in enumerate(x_test):
            result[i] = self.classifier(message)

        if self.verbose:
            print("Model: {0}, predictions: {1}".format(self.method.upper(), str(result)))

        return result

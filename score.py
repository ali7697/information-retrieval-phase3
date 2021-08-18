import math

from Dictionary import *


class score:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def cal_idf(self):
        # N: total number of terms in the dictionary
        num_total_words = len(self.dictionary.dictionary.keys())
        # nt: length of a words key
        for word in self.dictionary.dictionary.keys():
            nt = math.log10(num_total_words / len(self.dictionary.dictionary[word]))
            self.dictionary.dictionary[word].insert(0, nt)

    def cal_tf(self):
        # for implementation of this I need to save the repetitions of each
        # word in a document
        for word in self.dictionary.dictionary.keys():
            for doc_count_pair in self.dictionary.dictionary[word]:
                # turning doc_count_pair pair to doc_weight_pair
                doc_count_pair[1] = 1 + math.log10(doc_count_pair[1])
                                    # * self.dictionary.dictionary[word][0]  # tfidf_weight
    def cal_tfidf(self):
        for word in self.dictionary.dictionary.keys():
            for i in range(1, len(self.dictionary.dictionary[word])):
                # turning doc_count_pair pair to doc_weight_pair
                self.dictionary.dictionary[word][i][1] *= self.dictionary.dictionary[word][0]  # tfidf_weight

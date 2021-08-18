from Dictionary import *


class Cluster:
    def __init__(self, seed, dictionary, num_clusters):
        self.seed = seed
        self.doc_vector_dict = dict()
        self.length = dict()
        self.dictionary = dictionary
        self.num_clusters = num_clusters

    def cal_length(self):
        if len(self.doc_vector_dict.keys()) == self.num_clusters:
            for doc_id in self.doc_vector_dict.keys():
                self.length[doc_id] = math.sqrt(sum([j ** 2 for j in self.doc_vector_dict[doc_id].values()]))
        else:
            for doc_id in self.doc_vector_dict.keys():
                self.length[doc_id] = self.dictionary.doc_vectors_lengths[doc_id]

import heapq
import math
import pickle
from copy import deepcopy


class Dictionary:

    def __init__(self):
        self.dictionary = dict()
        self.words_total_count = dict()
        self.id_to_url_dict = dict()
        self.doc_vectors = dict()
        self.doc_vectors_lengths = dict()
        self.champions_list_dict = dict()

        self.doc_cluster_number = dict()
        self.doc_centers = dict()

    def sort_dict(self):
        # create the sorted final dictionary
        words = sorted(self.dictionary.keys())
        final_dictionary = dict()
        for word in words:
            final_dictionary[word] = self.dictionary[word]
        self.dictionary = final_dictionary

    def save_dict(self):
        filename = 'dict'
        outfile = open(filename, 'wb')
        pickle.dump(self.dictionary, outfile)
        outfile.close()
        filename = 'dict_url'
        outfile = open(filename, 'wb')
        pickle.dump(self.id_to_url_dict, outfile)
        outfile.close()
        filename = 'docs_lengths'
        outfile = open(filename, 'wb')
        pickle.dump(self.doc_vectors, outfile)
        outfile.close()
        filename = 'champions_list_dict'
        outfile = open(filename, 'wb')
        pickle.dump(self.champions_list_dict, outfile)
        outfile.close()
        filename = 'doc_vector_lengths'
        outfile = open(filename, 'wb')
        pickle.dump(self.doc_vectors_lengths, outfile)
        outfile.close()
        filename = 'doc_cluster_number'
        outfile = open(filename, 'wb')
        pickle.dump(self.doc_cluster_number, outfile)
        outfile.close()
        filename = 'doc_centers'
        outfile = open(filename, 'wb')
        pickle.dump(self.doc_centers, outfile)
        outfile.close()

    def read_dict(self):
        infile = open('dict', 'rb')
        s = pickle.load(infile)
        infile.close()
        infile = open('dict_url', 'rb')
        ss = pickle.load(infile)
        infile.close()
        infile = open('docs_lengths', 'rb')
        vecs = pickle.load(infile)
        infile.close()
        infile = open('champions_list_dict', 'rb')
        champs = pickle.load(infile)
        infile.close()
        infile = open('doc_vector_lengths', 'rb')
        dls = pickle.load(infile)
        infile.close()
        infile = open('doc_cluster_number', 'rb')
        doc_cluster = pickle.load(infile)
        infile.close()
        infile = open('doc_centers', 'rb')
        doc_centers = pickle.load(infile)
        infile.close()
        return s, ss, vecs, champs, dls, doc_cluster, doc_centers

    def test_dict(self, input_word):
        if input_word in self.dictionary.keys():
            print(self.dictionary[input_word])
        else:
            print("This word is not in the dictionary")

    def get_frequency_based_dict(self):
        freq_based_dict = sorted(self.words_total_count.items(), key=lambda x: x[1], reverse=True)
        # freq_based_dict = dict(freq_based_dict)
        return freq_based_dict

    def remove_k_frequent_words(self, k):
        words = self.get_frequency_based_dict()[0:k]
        for word in words:
            del self.dictionary[word[0]]

    def create_doc_vectors(self):
        for ID in self.id_to_url_dict.keys():
            self.doc_vectors[ID] = dict()
            self.doc_vectors[ID]['cluster'] = 0
        for word in self.dictionary.keys():
            for i in range(1, len(self.dictionary[word])):
                self.doc_vectors[self.dictionary[word][i][0]][word] = self.dictionary[word][i][1]

    def cal_doc_vectors_lengths(self):
        for x in self.doc_vectors.keys():
            self.doc_vectors_lengths[x] = math.sqrt(sum([j ** 2 for j in self.doc_vectors[x].values()]))


    def create_champions_list(self):
        tmp_dict = deepcopy(self.dictionary)
        for word in self.dictionary.keys():
            tmp_dict[word].remove(tmp_dict[word][0])
            largest = heapq.nlargest(40, tmp_dict[word], key=lambda x: x[1])
            largest = sorted(largest, key=lambda x: x[0])
            largest.insert(0, self.dictionary[word][0])
            self.champions_list_dict[word] = largest
        return self.champions_list_dict

from Retriver import *
from score import *

dictionary = Dictionary()
dictionary.dictionary, dictionary.id_to_url_dict, dictionary.doc_vectors, dictionary.champions_list_dict,\
    dictionary.doc_vectors_lengths, dictionary.doc_cluster_number, dictionary.doc_centers = dictionary.read_dict()
print(len(dictionary.dictionary.keys()))
ret = Retriever(dictionary)
while ret.get_query() != '-1':
    ret.retrieve('saved')


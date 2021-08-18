import math
import random
from copy import deepcopy
import heapq
from Equalizer import *
from score import *
from clusters import *


class Retriever:
    query: str

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.equalizer = Equalizer(self.dictionary)
        self.initial_doc_vectors = deepcopy(self.dictionary.doc_vectors)

    def get_dict(self):
        self.dictionary.dictionary, self.dictionary.id_to_url_dict, self.dictionary.doc_vectors, \
        self.dictionary.champions_list_dict, self.dictionary.doc_vectors_lengths, self.dictionary.doc_cluster_number, self.dictionary.doc_centers = self.dictionary.read_dict()

    def get_query(self):
        self.query = input('Please enter the query: ')

    def get_equalized_query(self):
        alphabet = 'آاأبپتثجچحخدذرزژسشصضطظعغفقکكگلم‌نوؤهیيئء'  # نیم فاصله داره
        r = re.compile(f'[{alphabet}]+')
        words = r.findall(self.query)
        words = self.equalizer.equalize_query(words)
        return words

    def retrieve(self, saved_or_from_scratch):
        words = self.get_equalized_query()
        b = 4
        # create query vector
        query_vector = dict()
        for word in words:
            query_vector[word] = 1 + math.log10(words.count(word))
        if saved_or_from_scratch == 'saved':
            seed_clusters = Cluster(-5, self.dictionary, 0)
            seed_clusters.doc_vector_dict = self.dictionary.doc_centers
            seed_clusters.cal_length()
        else:
            all_clusters = self.k_means()
            seed_clusters = all_clusters[0]
        dist_from_seeds = self.cal_cosine_similarity(query_vector, seed_clusters)
        dist_from_seeds = dict(sorted(dist_from_seeds.items(), key=lambda item: item[1], reverse=True))
        cluster_numbers = list(dist_from_seeds.keys())[0:b]
        big_cluster_vectors = dict()
        if saved_or_from_scratch == 'saved':
            for doc_id in self.dictionary.doc_cluster_number.keys():
                if self.dictionary.doc_cluster_number[doc_id] in cluster_numbers:
                    big_cluster_vectors[doc_id] = self.dictionary.doc_vectors[doc_id]
        else:
            for cluster_num in cluster_numbers:
                big_cluster_vectors = {**big_cluster_vectors, **all_clusters[cluster_num].doc_vector_dict}
        big_cluster = Cluster(-1, self.dictionary, 0)
        big_cluster.doc_vector_dict = big_cluster_vectors
        big_cluster.cal_length()
        dist_from_docs_in_cluster = self.cal_cosine_similarity(query_vector, big_cluster)
        # dist_from_docs_in_cluster = dict(
        #     sorted(dist_from_docs_in_cluster.items(), key=lambda item: item[1], reverse=True))
        # closest_docs = list(dist_from_docs_in_cluster.keys())[0:30]
        closest_docs = heapq.nlargest(30, list(dist_from_docs_in_cluster.items()), key=lambda x: x[1])
        closest_docs = sorted(closest_docs, key=lambda x: x[1], reverse=True)
        output_printed = dict()
        print('Num of docs: ' + str(len(closest_docs)))
        # for doc_id in closest_docs:
        #     print(str(doc_id) + '\t' + self.dictionary.id_to_url_dict[doc_id])
        for doc_id, val in closest_docs:
            print(str(doc_id) + '\t' + self.dictionary.id_to_url_dict[doc_id])
        return output_printed

    def cal_rss(self, clusters):
        rss = 0
        for i in range(1, len(clusters)):
            # calculate the sum of cosine similarities between each cluster center and its docs
            rss += sum(list(self.cal_cosine_similarity(clusters[0].doc_vector_dict[i], clusters[i]).values()))
        return rss

    def cal_cosine_similarity(self, main_doc, cluster):
        doc_similarity_dict = dict()
        main_doc_vector_length = math.sqrt(sum([j ** 2 for j in main_doc.values()]))
        for doc_id in cluster.doc_vector_dict.keys():
            if cluster.length[doc_id] == 0:
                continue
            tmp_sum = 0
            for term in main_doc.keys():
                if term in cluster.doc_vector_dict[doc_id].keys() and term != 'cluster':
                    tmp_sum += main_doc[term] * cluster.doc_vector_dict[doc_id][term]
            tmp_sum /= (cluster.length[doc_id] * main_doc_vector_length)
            doc_similarity_dict[doc_id] = tmp_sum
        return doc_similarity_dict

    def cal_one_clustering_iteration(self, k):
        seeds = random.sample(list(self.initial_doc_vectors.values()), k)
        clusters = dict()
        clusters[0] = Cluster(0, self.dictionary, k)
        for i in range(1, k + 1):
            clusters[i] = Cluster(seeds[i - 1], self.dictionary, k)
            clusters[0].doc_vector_dict[i] = seeds[i - 1]
        clusters[0].cal_length()
        for iteration in range(10):
            for doc_id in self.dictionary.doc_vectors.keys():
                if self.dictionary.doc_vectors_lengths[doc_id] == 0:
                    continue
                # calculate the similarity of the doc to each of the cluster seeds
                sim_dict = self.cal_cosine_similarity(self.dictionary.doc_vectors[doc_id], clusters[0])
                sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
                cluster_number = list(sim_dict.keys())[0]
                if self.dictionary.doc_vectors[doc_id]['cluster'] != cluster_number:
                    clusters[cluster_number].doc_vector_dict[doc_id] = self.dictionary.doc_vectors[doc_id]
                    if self.dictionary.doc_vectors[doc_id]['cluster'] != 0:
                        del clusters[self.dictionary.doc_vectors[doc_id]['cluster']].doc_vector_dict[doc_id]
                    self.dictionary.doc_vectors[doc_id]['cluster'] = cluster_number
            for i in range(1, k + 1):
                clusters[i].cal_length()
            # update seeds
            for i in range(1, k + 1):
                new_cluster_seed = dict()
                for doc_id in clusters[i].doc_vector_dict.keys():
                    for term in clusters[i].doc_vector_dict[doc_id]:
                        if term != 'cluster':
                            if term in new_cluster_seed.keys():
                                new_cluster_seed[term] += (clusters[i].doc_vector_dict[doc_id][term]) \
                                                          / len(clusters[i].doc_vector_dict.keys())
                            else:
                                new_cluster_seed[term] = clusters[i].doc_vector_dict[doc_id][term] \
                                                         / len(clusters[i].doc_vector_dict.keys())
                clusters[0].doc_vector_dict[i] = new_cluster_seed
            clusters[0].cal_length()
        return clusters

    def reset_doc_clusters(self):
        for doc_id in self.dictionary.doc_vectors.keys():
            self.dictionary.doc_vectors[doc_id]['cluster'] = 0

    def k_means(self):
        k = 20
        num_reps_of_clustering = 4
        best_clusters = self.cal_one_clustering_iteration(k)
        print('one_done')
        best_rss = self.cal_rss(best_clusters)
        # self.dictionary.doc_vectors = deepcopy(self.initial_doc_vectors)
        self.reset_doc_clusters()
        for cluster_count in range(num_reps_of_clustering):
            tmp_clusters = self.cal_one_clustering_iteration(k)
            print('one_done')
            tmp_rss = self.cal_rss(tmp_clusters)
            if tmp_rss > best_rss:
                best_rss = tmp_rss
                best_clusters = tmp_clusters
            self.reset_doc_clusters()
        for cluster_id in range(1, len(best_clusters)):
            for doc_id in best_clusters[cluster_id].doc_vector_dict.keys():
                self.dictionary.doc_cluster_number[doc_id] = cluster_id
        for cluster_center_id in best_clusters[0].doc_vector_dict.keys():
            self.dictionary.doc_centers[cluster_center_id] = best_clusters[0].doc_vector_dict[cluster_center_id]
        return best_clusters

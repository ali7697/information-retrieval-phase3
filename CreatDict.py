from News import *
from Retriver import *
from score import score

All_News = []
equalizer = Equalizer(dictionary)
# read the News file
id_counter = 0
with open('17k.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        # if id_counter == 2000:
        #     break
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer)
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

with open('20k.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        # if line == 300:
        #     break
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer)
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

with open('IR00_3_11k News.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        # if line == 300:
        #     break
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer)
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

dictionary.remove_k_frequent_words(50)
equalizer.equalize_dict()
dictionary1 = equalizer.ret_dict()
dictionary.sort_dict()
score_giver = score(dictionary)
score_giver.cal_tf()
score_giver.cal_idf()
score_giver.cal_tfidf()
dictionary.create_doc_vectors()
dictionary.create_champions_list()
dictionary.cal_doc_vectors_lengths()


ret = Retriever(dictionary)
ret.get_query()
ret.retrieve('from_scratch')
dictionary.save_dict()


import re
import csv
from copy import deepcopy


def get_2D_list(filename):
    first = []
    second = []
    alphabet = 'آاأبپتثجچحخدذرزژسشصضطظعغفقکكگلم‌نوؤهیيئء'  # نیم فاصله داره
    r = re.compile(f'[{alphabet}]+')
    with open(filename, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row[0] = r.findall(row[0])[0]
            row[1] = r.findall(row[1])[0]
            second.append(row[0])
            first.append(row[1])
    return first, second


plurals, singles = get_2D_list('plurals3.csv')
present, past = get_2D_list('verbs.csv')


def cal_list(lists):
    i = 0
    prim_lists = deepcopy(lists)

    while i < len(lists) - 1:
        tmp = []
        # if lists == s and i == 60:
        #     print('jj')
        if lists[i][0] == lists[i + 1][0]:
            tmp = [lists[i][0], lists[i][1] + lists[i + 1][1]]
            lists.remove(lists[i])
            lists.remove(lists[i])
            lists.insert(i, tmp)
        else:
            i += 1

    for i in range(len(lists) - 1):
        if lists[i][0] == lists[i + 1][0]:
            print('jj')
    return lists


class Equalizer:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def prefix_equalizer(self, word, used_set, v_or_n, q_or_d):
        the_keys = self.dictionary.dictionary.keys()

        for pre in used_set:
            equalized_word = word[len(pre):]
            if word.startswith(pre):
                if q_or_d == 'd' and word in the_keys and len(equalized_word) > 1:
                    if equalized_word not in the_keys:  # added
                        self.dictionary.dictionary[equalized_word] = []  # added
                    # self.dictionary.dictionary[equalized_word] = list(set(sorted(
                    #     self.dictionary.dictionary[equalized_word] + self.dictionary.dictionary[word])))
                    self.dictionary.dictionary[equalized_word] = cal_list(sorted(
                        self.dictionary.dictionary[equalized_word] + self.dictionary.dictionary[word], key=lambda x: x[0]))
                    del self.dictionary.dictionary[word]
                    word = equalized_word
                    word = self.prefix_equalizer(word, used_set, v_or_n, q_or_d)
                elif q_or_d == 'q' and len(equalized_word) > 1:
                    word = equalized_word
                    word = self.prefix_equalizer(word, used_set, v_or_n, q_or_d)
        return word

    def postfix_equalizer(self, word, used_set, q_or_d):
        the_keys = self.dictionary.dictionary.keys()
        for post in used_set:
            equalized_word = word[:-len(post)]
            if word.endswith(post):
                if word in the_keys and len(equalized_word) > 1 and q_or_d == 'd':
                    if equalized_word not in the_keys:
                        self.dictionary.dictionary[equalized_word] = []
                    # self.dictionary.dictionary[equalized_word] = list(set(sorted(
                    #     self.dictionary.dictionary[equalized_word] + self.dictionary.dictionary[word])))
                    self.dictionary.dictionary[equalized_word] = cal_list(sorted(
                        self.dictionary.dictionary[equalized_word] + self.dictionary.dictionary[word], key=lambda x: x[0]))

                    del self.dictionary.dictionary[word]
                    word = equalized_word
                    word = self.postfix_equalizer(word, used_set, q_or_d)
                elif q_or_d == 'q' and len(equalized_word) > 1:
                    word = equalized_word
                    word = self.postfix_equalizer(word, used_set, q_or_d)
        return word

    def verb_list_equalizer(self, word, q_or_d):
        starting = 'no_starting'
        for st in self.set_of_pres_present:
            if word.startswith(st):
                starting = st
                break
        first_starting = starting
        for verb in past:
            if verb in word:

                if word.startswith(verb) or starting != 'no_starting':
                    word = self.prefix_equalizer(word, self.set_of_pres_past, 'v', q_or_d)
                    word = self.postfix_equalizer(word, self.set_of_posts_verbs_past, q_or_d)

                return word

        for verb in present:
            if (word.startswith('ب') or word.startswith('ن')) and word.endswith(verb) and word != verb:
                if verb in word:
                    word = word[len(starting):]
            if first_starting == 'no_starting' and word.startswith(verb):
                starting = verb
            for ending in self.set_of_posts_verbs_present:
                if word.endswith(verb + ending) and starting != 'no_starting' \
                        and (word == starting + verb + ending or word == verb + ending):
                    word = word[len(starting):]
                    word = self.postfix_equalizer(word, self.set_of_posts_verbs_present, q_or_d)

                    return word
            if first_starting == 'no_starting':
                starting = 'no_starting'
        return word

    def plurals_dict(self, word):
        for i in range(len(plurals)):
            if word == plurals[i]:
                word = singles[i]
                if singles[i] in self.dictionary.dictionary.keys():
                    # self.dictionary.dictionary[singles[i]] = list(set(sorted(
                    #     self.dictionary.dictionary[singles[i]] + self.dictionary.dictionary[plurals[i]])))
                    self.dictionary.dictionary[singles[i]] = cal_list(sorted(
                        self.dictionary.dictionary[singles[i]] + self.dictionary.dictionary[plurals[i]], key=lambda x: x[0]))
                else:
                    self.dictionary.dictionary[singles[i]] = self.dictionary.dictionary[plurals[i]]
                del self.dictionary.dictionary[plurals[i]]
                return word
        return word

    def plurals_query(self, word):
        for i in range(len(plurals)):
            if word == plurals[i]:
                word = singles[i]
                return word
        return word

    def char_equalizer(self, word, q_or_d):
        chars_1 = ['', 'أ', 'و', 'ی', 'ک', 'ا']
        chars_2 = ['ء', 'ئ', 'ؤ', 'ي', 'ك', 'آ']
        for i in range(len(chars_2)):
            if chars_2[i] in word:
                word = re.sub(chars_2[i], chars_1[i], word)
        return word

    def equalize_dict(self):
        the_keys = self.dictionary.dictionary.keys()
        the_keys = list(the_keys)
        for word in self.set_of_pres_present:
            if word in the_keys:
                del self.dictionary.dictionary[word]
        for word in self.set_of_posts_verbs_present:
            if word in the_keys:
                del self.dictionary.dictionary[word]
        for word in self.set_of_pres_non_verb:
            if word in the_keys:
                del self.dictionary.dictionary[word]
        for word in self.set_of_posts_non_verb:
            if word in the_keys and word != 'ی' and word != 'م':
                del self.dictionary.dictionary[word]
        for word in the_keys:
            if word in self.dictionary.dictionary.keys():
                kk = word

                word = self.verb_list_equalizer(word, 'd')

                word = self.prefix_equalizer(word, self.set_of_pres_non_verb, 'n', 'd')

                word = self.postfix_equalizer(word, self.set_of_posts_non_verb, 'd')

                word = self.plurals_dict(word)

    def equalize_query(self, words):
        final_words = []
        all_words = self.dictionary.dictionary.keys()
        for word in words:
            if word in self.set_of_pres_present \
                    or word in self.set_of_posts_verbs_present or word in self.set_of_pres_non_verb \
                    or word in self.set_of_posts_non_verb:
                continue
            # omitting \u200c
            if '\u200c' in word:
                word = re.sub('\u200c', '', word)
            word = self.verb_list_equalizer(word, 'q')
            word = self.prefix_equalizer(word, self.set_of_pres_non_verb, 'n', 'q')
            word = self.postfix_equalizer(word, self.set_of_posts_non_verb, 'q')
            word = self.plurals_query(word)
            word = self.char_equalizer(word, 'q')
            final_words.append(word)
            if word not in all_words:
                continue
        return final_words

    def ret_dict(self):
        return self.dictionary

    set_of_pres_present = ['می', 'نمی', 'ب', 'ن']
    set_of_pres_past = ['می', 'نمی', 'ن']
    set_of_pres_non_verb = ['ابر', 'با', 'بی', 'پس', 'پسا', 'پیش', 'تک', 'نا']
    set_of_posts_verbs_past = ['ام', 'ای', 'اند', 'ایم', 'اید', 'یم', 'ید', 'ند', 'ی', 'م', 'ه', 'ن']
    set_of_posts_verbs_present = ['م', 'ی', 'یم', 'ید', 'ند', 'د']
    set_of_posts_non_verb = ['مان', 'بان', 'ستان', 'تان', 'دان', 'شان', 'انه', 'ان',
                             'تر', 'ترین', 'بد', 'زار',
                             'سرا', 'کده',
                             'گاه',
                             'ناک',
                             'وار', 'واره', 'واری', 'ی', 'ها', 'ت']

### Data summary

import os
import nltk
from scipy import stats
from numpy import std, mean, sqrt, arange
import json
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import operator
import csv


class DataAnalysis:
    def __init__(self):
        self.pos_list = ["RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"]
        self.unqiue_words = {}

    def readApp(self):
        ### Detect number of opinion words
        gp_app_dir = os.listdir("../dataset/android/")
        ios_app_dir = os.listdir("../dataset/ios/")
        for dir in gp_app_dir:
            dir_n = "../dataset/android/" + dir
            if os.path.isdir(dir_n):
                print("Come to dir ", dir)
                data_fn = os.path.join("../dataset/android/" + dir+"/total_info.txt")
                data_fr = open(data_fn)
                lines = data_fr.readlines()
                for idx, line in enumerate(lines):
                    review = line.split("******")[1]
                    tokens = nltk.word_tokenize(review)
                    pos_tags = nltk.pos_tag(tokens)
                    for word, pos_tag in pos_tags:
                        if pos_tag in self.pos_list:
                            if word not in self.unqiue_words:
                                self.unqiue_words[word] = 0
                            self.unqiue_words[word] += 1

        for dir in ios_app_dir:
            dir_n = "../dataset/android/" + dir
            if os.path.isdir(dir_n):
                print("Come to dir ", dir)
                data_fn = os.path.join("../dataset/ios/" + dir + "/total_info.txt")
                data_fr = open(data_fn)
                lines = data_fr.readlines()
                for idx, line in enumerate(lines):
                    review = line.split("******")[1]
                    tokens = nltk.word_tokenize(review)
                    pos_tags = nltk.pos_tag(tokens)
                    for word, pos_tag in pos_tags:
                        if pos_tag in self.pos_list:
                            if word not in self.unqiue_words:
                                self.unqiue_words[word] = 0
                            self.unqiue_words[word] += 1

        sorted_words = {k: v for k, v in sorted(self.unqiue_words.items(), key=lambda item: item[1])}
        print("Number of opinion words is ", len(list(sorted_words.keys())))
        fw = open("../dataset/opinion_words.json", "w")
        json.dump(sorted_words, fw)
        fw.close()



    def cohen_d(self, x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (mean(x) - mean(y)) / sqrt(((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)

    def sig_test(self, v1, v2):
        pvalue = stats.wilcoxon(v1, v2)
        print("Wilcoxon test p value is ", pvalue)
        print("Cohens d result is ", self.cohen_d(v1, v2))

    def vis_opinion_words(self, fn):
        fp = open(fn)
        opinion_dict = json.load(fp)
        fp.close()

        sorted_opinions = sorted(opinion_dict.items(),key=operator.itemgetter(1),reverse=True)
        keys = []
        values = []
        for key, value in sorted_opinions:
            keys.append(key)
            values.append(value)
        x_ticks = arange(len(keys))
        plt.bar(x_ticks, values, align='center', alpha=0.5)
        # plt.xticks(x_ticks, list(sorted_opinions.keys()))
        plt.show()

    def save_top_words(self, fn, fwn, top_n=500):
        fp = open(fn)
        opinion_dict = json.load(fp)
        fp.close()

        sorted_opinions = sorted(opinion_dict.items(), key=operator.itemgetter(1), reverse=True)
        top_words = []
        for key, value in sorted_opinions[:top_n]:
            top_words.append(key)
        print(top_words[450:])

        # with open(fwn, "w") as fw:
        #     writer = csv.writer(fw, delimiter=',')
        #     for top_word in top_words:
        #         writer.writerow([top_word])


if __name__ == "__main__":
    # DataAnalysis().readApp()
    # v1_p = [0.523, 0.431, 0.440, 0.227, 0.523, 0.531]*2
    # v2_p = [0.586, 0.55, 0.686, 0.646, 0.707, 0.699]*2
    # v1_s = [0.636, 0.526, 0.638, 0.58, 0.587, 0.546]*2
    # v2_s = [0.71, 0.841, 0.821, 0.857, 0.832, 0.793]*2
    # phrase = [0.586, 0.55, 0.686, 0.646, 0.707, 0.699]
    # sentence = [0.71, 0.841, 0.821, 0.857, 0.832, 0.793]
    # DataAnalysis().sig_test(v1_p, v2_p)
    # DataAnalysis().sig_test(v1_s, v2_s)
    # DataAnalysis().sig_test(phrase, sentence)

    ### Visualize the opinion words by frequency
    # DataAnalysis().vis_opinion_words("../dataset/opinion_words.json")

    ### Output top-frequency words
    DataAnalysis().save_top_words("../dataset/opinion_words.json", "../dataset/top_words.csv")
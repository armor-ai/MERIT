# Read reviews and extract issues, including memory, CPU, battery, and traffic.
# Use word2vec and kmeans to find the similar words
# -*- coding: utf-8 -*-
# __author__  = "Cuiyun Gao"
# __version__ = "1.0"


import os
import logging
import itertools
import json
import re
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import Word2Vec, Phrases, LdaMulticore, TfidfModel
from sklearn.cluster import KMeans, SpectralClustering
from extractSentenceWords import *
from collections import defaultdict
import pickle

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

bigram = None
trigram = None

new_app_fn = {
    "waze_gp": "../dataset/waze_gp/waze.txt",
    "waze_ios": "../dataset/waze_ios/waze.txt",
    "snapchat": "../dataset/snapchat/snapchat.txt",
    "shareit": "../dataset/shareit/shareit.txt"
}

# Get cleaned mongodb reviews, including 1547 apps
def get_all_reviews():
    doc_words = []
    doc_sent_words = []
    num_docs = 0
    num_words = 0
    apk_path = os.path.join("..", "dataset", "raw")
    apk_names = os.listdir(os.path.join(apk_path, "mongodb"))
    apk_review_paths = [os.path.join(apk_path, "mongodb", apk_name, "review.txt") for apk_name in apk_names]
    apk_rate_paths = [os.path.join(apk_path, "mongodb", apk_name, "info.txt") for apk_name in apk_names]
    for root, dirs, files in os.walk(apk_path, "mysql"):
        for name in files:
            filename = os.path.join(root, name)
            if re.match(r'.*clean_review\.txt', filename):
                apk_review_paths.append(filename)
                apk_names.append(name)
            if re.match(r'.*clean_data\.txt', filename):
                apk_rate_paths.append(filename)

    for idx, item in enumerate(apk_review_paths):
        logging.info(item)
        with open(item) as fin, open(apk_rate_paths[idx]) as frin:
            review_lines = fin.readlines()
            rate_lines = frin.readlines()
            if len(review_lines) != len(rate_lines):
                logging.error("length not equal at %s"%item)
            for j, line in enumerate(review_lines):
                words_sents, wc = extractSentenceWords(line, sent=False)
                doc_words.append(words_sents)
                num_docs += 1
                num_words += wc

    for apk, fn in new_app_fn.items():
        with open(fn) as fin:
            lines = fin.readlines()
        for line in lines:
            line = line.strip()
            terms = line.split("******")
            review_o = terms[1]
            review_p, wc = extractSentenceWords(review_o, sent=False)
            num_docs += 1
            num_words += wc
            doc_words.append(review_p)
    logging.info("Read %d docs, %d words!" % (num_docs, num_words))
    return doc_words


def extract_phrases(doc_words, save=False):
    logging.info("Extracting phrases...")
    global bigram
    global trigram
    bigram = Phrases(doc_words, threshold=5, min_count=5)
    trigram = Phrases(bigram[doc_words], threshold=3, min_count=3)
    if save:
        bigram.save("../model/bigram.model")
        trigram.save("../model/trigram.model")

    return trigram[bigram[doc_words]]


# Input "reviews" is a list of sentences, whose words are split.
def training(reviews):
    # convert glove format to word2vec, we use the twitter model of 200 dimensions from https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models
    # glove_model_path = os.path.join("..", "model", "glove.twitter.27B", "glove.twitter.27B.200d.txt")
    # word2vec_pre_model = os.path.join("..", "data", "pre_twitter_word2vec.model")
    # glove2word2vec.glove2word2vec(glove_model_path, word2vec_pre_model)


    # laoding the pre-trained model and retrain with the reviews
    logging.info("Training word2vec...")
    model = Word2Vec(reviews, size=200, min_count=3, workers=8)
    logging.info("Saving word2vec model...")
    model.save(os.path.join("..", "model", "wv", "appreviews_word2vec.model"))
    # model = Word2Vec.load_word2vec_format(word2vec_pre_model, binary=False)
    # model.build_vocab(sentences, update=True)
    # model.train(sentences)
    #
    # # model = Word2Vec(bigram_transformer[reviews], size=128, window=5, min_count=3)
    # output_path = os.path.join("..", "model", "reviews.model.bin")
    # model.save(output_path, binary=True)
    return model

def load_model():
    bigram = Phrases.load(os.path.join("..", "model", "bigram.model"))
    trigram = Phrases.load(os.path.join("..", "model", "trigram.model"))
    wv_model = Word2Vec.load(os.path.join("..", "model", "appreviews_word2vec.model"))
    logging.info("Load word2vec model finished")
    return bigram, trigram, wv_model

# find similar word to each issue
def get_similar_word(model, keywords, similar_num):
    issue_dict = {}
    for issue in keywords:
        origi_words = model.most_similar(issue, topn=similar_num)
        origi_words = [word[0] for word in origi_words]
        issue_dict[issue] = origi_words
    return issue_dict


if __name__ == "__main__":
    doc_words = get_all_reviews()
    pickle.dump(doc_words, open("doc_words.pkl", "wb"))

    reviews = extract_phrases(doc_words, save=True)
    model = training(reviews)

#!/usr/bin/python

import os
import timeit
import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from utils import plot_roc_curves, plot_confusion_matrix, GENRE_DIR, GENRE_LIST, TEST_DIR

from ceps import read_ceps, create_ceps_test, read_ceps_test

from pydub import AudioSegment

genre_list = GENRE_LIST

clf = None

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def test_model_on_single_file(file_path):
    clf = joblib.load('saved_model/model_ceps.pkl')
    X, y = read_ceps_test(create_ceps_test(test_file)+".npy")
    probs = clf.predict_proba(X)
    print "\t".join(str(x) for x in traverse)
    print "\t".join(str("%.3f" % x) for x in probs[0])
    probs=probs[0]
    max_prob = max(probs)
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    
    print max_prob_index
    predicted_genre = traverse[max_prob_index]
    print "\n\npredicted genre = ",predicted_genre
    return predicted_genre

if __name__ == "__main__":

    global traverse
    traverse = genre_list
    for path, dirs, files in os.walk(TEST_DIR):
        for fi in files:
            if fi.endswith('wav'):
                test_file = os.path.join(path, fi)
                print "\n###", test_file
                predicted_genre = test_model_on_single_file(test_file)



'''
    global traverse
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection(set(GENRE_LIST)))
        break

    test_file = "/home/florencia/whales/data/mySamples/human/music_speech/music/jazz.wav"
    # should predict genre as "ROCK"
    predicted_genre = test_model_on_single_file(test_file)
    
'''

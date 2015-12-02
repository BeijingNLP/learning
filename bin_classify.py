# -*- coding: utf-8 -*-
#from __future__ import print_function

import loaddata
import jieba
import scipy
import datetime


import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

from time import time
import codecs
import re

g_stopwordList=[]
g_pattern=None
g_pattern1=None
 
def getTrainAndVerifyDataSets():
    articles=loaddata.loadJsonObjFromFile('data/data.json')
    pos_list,neg_list=loaddata.loadDataSets(articles)

    len_pos=len(pos_list)
    len_neg=len(neg_list)

    len1=int(len_pos*0.7)
    len2=int(len_neg*0.7)
    train_data= pos_list[:len1 ] + neg_list[:len2]
    verify_data= pos_list[len1:] + neg_list[len2:]

    #train_data=  random.shuffle(train_data)
    #verify_data = random.shuffle(verify_data)

    return train_data,verify_data



def isNotAStopWord(w):
    global g_stopwordList
    global g_pattern    
    if len(g_stopwordList)<1:
        g_pattern=re.compile(r"[a-zA-Z]*\d+\.{0,1}\d*") 
        #g_pattern1= re.compile(r"[a-zA-Z]*\d+\.{0,1}\d*") 
        g_stopwordList=set()
        lines=codecs.open('stopwords.txt',encoding='utf-8',mode='r').readlines()
        for l in lines:
            l=l.strip('\n')
            l=l.strip('\r')
            g_stopwordList.add(l)
    else:
        
        match= g_pattern.match(w)
        if match:
            return False
            
        if w in g_stopwordList:
           return False 
        
    return True
    

def tokenizeText(text):
    tokens= [w for w in jieba.cut(text) if isNotAStopWord(w)]
    return tokens

def getValdiatedDocAndLabel(objList):
    X=[]
    Y=[]    
    for doc in objList:
        label=doc['label'] 
        text=doc['body']
        if label in [u'0',u'1']:
            X.append(text)
            Y.append(label)
        else:
            print 'bad label, value=%s, ric=%s' %( label, doc['ric'])
    return X,Y

def getStringListFromObjList(trainObjList,testObjList):

    trainDocList,Y_train=getValdiatedDocAndLabel(trainObjList)
    testDocList,Y_test = getValdiatedDocAndLabel(testObjList)    

    return trainDocList,testDocList,Y_train,Y_test

def buildFeature(trainList,testList):

        
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, tokenizer=tokenizeText, ngram_range=(1,2),
    #                             stop_words='english')
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, tokenizer=tokenizeText, 
                                 stop_words='english')
    X_train = vectorizer.fit_transform(trainList)
    
    t0 = time()
    X_test = vectorizer.transform(testList)
    duration = time() - t0    
        
    return X_train,X_test,vectorizer

def runBenchmark(clf, X_train,Y_train,X_test,Y_test):
    try:
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, Y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.f1_score(Y_test, pred)
        print("f1-score:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))


        print()

        print("classification report:")
        print(metrics.classification_report(Y_test, pred))

      
        print("confusion matrix:")
        print(metrics.confusion_matrix(Y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    except Exception as e:
        print e

def binaryLabel(y):
    lb = LabelBinarizer()
    y_train = np.array([number[0] for number in lb.fit_transform(y)])
    return y_train
    
if __name__ == '__main__' :
    
      t0 = time()
      train_data,test_data=getTrainAndVerifyDataSets()
      #trainDocList,tstDocList,Y_train,Y_test=getStringListFromObjList(train_data[:100],test_data[:100])
      trainDocList,tstDocList,Y_train,Y_test=getStringListFromObjList(train_data,test_data)
      X_train,X_test,vectorizer=buildFeature(trainDocList,tstDocList)
      clf=MultinomialNB(alpha=.01)
      Y_train= binaryLabel(Y_train)
      Y_test = binaryLabel(Y_test)
      
      #print 'run with MultinomialNB...'
      #runBenchmark(clf, X_train,Y_train,X_test,Y_test)
      
      #print 'run with BernoulliNB...'
      #clf= BernoulliNB(alpha=.01)
      #runBenchmark(clf, X_train,Y_train,X_test,Y_test)

      for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"), 
        (BernoulliNB(alpha=.01),"BernoulliNB"),
        (MultinomialNB(alpha=0.01),"MultinomialNB"),
        (LinearSVC(loss='l2', penalty="L1", dual=False, tol=1e-3),"LinearSVC-L1"),
        (LinearSVC(loss='l2', penalty="L2", dual=False, tol=1e-3),"LinearSVC-L2"),
        (SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),"SGDClassifier")

         ):
            print 'run with %s...' %name
            runBenchmark(clf, X_train,Y_train,X_test,Y_test)

      t1 = time()   
      print 'done in ', (t1-t0)
      
      feature_names = np.asarray(vectorizer.get_feature_names()).tolist()
      
      words='\r\n'.join(feature_names)
      fp=codecs.open('out_words.txt',encoding='utf-8',mode='w')
      fp.write(words)
      fp.close()
     
      
      

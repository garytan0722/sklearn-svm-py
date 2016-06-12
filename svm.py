# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:23:10 2016

@author: garytan
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
train_label=[]
test_label=[]
train_feature=[]
test_feature=[]
with open('nvalue_train.txt') as f:
    for  line in f:    
        res=line.split(" ", 1)       
        train_label.append(int(res[0]))
        train_feature.append(res[1])
with open('nvalue_test.txt') as f:
    for  line in f:    
        res=line.split(" ", 1)       
        test_label.append(int(res[0]))
        test_feature.append(res[1])
        
X_train = np.array(train_feature)
#y = np.array([1, 1, 2, 2]) 
X_test = np.array(test_feature)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print count_vect.vocabulary_.get(u'abc')
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                             alpha=1e-3, n_iter=5, random_state=42)),])
                      
text_clf.fit(X_train, train_label)
predicted = text_clf.predict(X_test)
print accuracy_score(test_label,predicted)
print(metrics.classification_report(test_label, predicted))
#clf = SVC()
#clf.fit(X, y) 
#print(clf)
#print(clf.predict([[-0.8, -1]]))
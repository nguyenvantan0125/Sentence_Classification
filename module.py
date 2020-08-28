# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:48:03 2020

@author: nguye
"""
from pyvi import ViTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
import pandas 

def readcsv(filename):
            a = pandas.read_csv(filename,header = None)
            return a 
def vitoken(corpus):       
            for idx, txt in enumerate(corpus):
                corpus[idx] = ViTokenizer.tokenize(txt) 
            return corpus
            
#%%
class vectorizer():
    def countvector(corpus):
        vect = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
        vect.fit(corpus)
        return vect
    def countvector_array(vect,corpus):
        X = vect.transform(corpus).toarray()
        return X
    def bag_of_words(vect,cotpus):
        bow = vect.vocabulary_
        return bow
    def tfidf(corpus):
        vect_tfidf = TfidfVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
        vect_tfidf.fit(corpus)
        return vect_tfidf
    def tfidf_array(vect,corpus):
        X = vect.transform(corpus).toarray()
        return X
#%%
class preprocessing(vectorizer):
        
    def encoder (y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            return y
    @classmethod
    def processing(cls,vectorizer,corpus,test): # corpus và test đã mã hóa, 
        X_train = super().countvector_array(vectorizer,corpus)
        X_test = super().countvector_array(vectorizer,test)
        yield X_train
        yield X_test
#%%
class models_selections(object):
    def model_SVC_linear (X,y,test_data):
        model = svm.SVC( probability = True,kernel='linear')
        model.fit(X,y)
        #predict
        predicted = model.predict(test_data)
        print('\n\t Dung SVC-linear de tinh:')
        print ("\nPredicted Value:", predicted)
        
    def model_SVC_rbf (X,y,test_data):
        model = svm.SVC( probability = True, kernel='rbf') 
        model.fit(X,y)
        #predict
        predicted = model.predict(test_data)
        print('\n\t Dung SVC_rbf de tinh:')
        print ("\nPredicted Value:", predicted)
        
    def pipeline_NB(corpus,y,test):
        pipe_line = Pipeline([
            ("vect", CountVectorizer()),#bag-of-words
            # ("tfidf", TfidfTransformer()),#tf-idf
            ("clf", MultinomialNB()) #model
            ])
        pipe_line.fit(corpus,y)
        predicted = pipe_line.predict(test)
        print('\n\t Dung Naive Bayer de tinh:')
        print ("\nPredicted Value:", predicted)
        
        
#%%

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
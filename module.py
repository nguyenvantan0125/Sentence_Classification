from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
import pandas 
    
    # read file CSV
def readcsv(filename):
    a = pandas.read_csv(filename,header = None)
    return a 


    # Tokenizer - tách từ tiếng việt
class tokenizer():
    def __init__(self,corpus):
        self.corpus = corpus
    def vi_tokenizer(self):
        for idx, txt in enumerate(self.corpus):
            self.corpus[idx] = ViTokenizer.tokenize(txt)
        return self.corpus

    
class SVC_linear():
    """
    1. Khởi tạo và thực hiện tách từ tiếng việt
    2. Tạo pipeline từ scikit-learn 
        2.1. Encoder bằng CountVectorizer
        2.2. Chọn model SVC-learn
    3. Fit và Predict 
    4. Print kết quả 
    """
   
    def __init__(self, raw_corpus,y, sentence):
        self.corpus = tokenizer(raw_corpus).vi_tokenizer()
        self.y = y
        self.sentence = tokenizer(sentence).vi_tokenizer()     
    def processing(self):
        pipe_line = Pipeline([
            ("vect", CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")),#bag-of-words
            ("clf", svm.SVC(probability = True, kernel='linear')) #model
            ])
        pipe_line.fit(self.corpus,self.y)
        predicted = pipe_line.predict(self.sentence)
        predict_proba = pipe_line.predict_proba(self.sentence)
        print('\n\t SVC-linear:')
        print ("\nPredicted Value:", *predicted)
        print ("\nPredict_Proba: ", max(*predict_proba))

class NavieBayes():
    """
    1. Khởi tạo và thực hiện tách từ tiếng việt
    2. Tạo pipeline từ scikit-learn 
        2.1. Encoder bằng CountVectorizer
        2.2. Chọn model Navie Bayes
    3. Fit và Predict 
    4. Print kết quả 
    """
    def __init__(self, raw_corpus,y, sentence):
        self.corpus = tokenizer(raw_corpus).vi_tokenizer()
        self.y = y
        self.sentence = tokenizer(sentence).vi_tokenizer()     
    def processing(self):
        pipe_line = Pipeline([
            ("vect", CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")),
            ("clf", MultinomialNB())
            ])
        pipe_line.fit(self.corpus,self.y)
        predicted = pipe_line.predict(self.sentence)
        predict_proba = pipe_line.predict_proba(self.sentence)
        print('\n\t Naive Bayer:')
        print ("\nPredicted Value:", *predicted)
        print ("\nPredict_Proba: ", max(*predict_proba))

class Cosine_Sim():
    def __init__(self,raw_corpus, y, sentence):
        self.corpus = tokenizer(raw_corpus).vi_tokenizer()
        self.y = y
        self.sentence = tokenizer(sentence).vi_tokenizer()  
    def processing(self):
        tfidf = TfidfVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
        X_train = tfidf.fit_transform(self.corpus).toarray()
        X_test = tfidf.transform(self.sentence).toarray()
        result = []
        for idx in range(len(X_train)):
            x = X_train[idx].reshape(1,len(X_train[idx]))
            a =  cosine_similarity(x,X_test)
            result.append(a[0])
        m = max(result)
        for i, j in enumerate(result): 
            if j == m:
                idx = i
                break
    # show result
        print('\n\t Consine Similarity:')
        print('\nPredict Values: ',self.y[idx])
        

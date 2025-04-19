from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np

class TFIDF_LR_Classifier:
    def __init__(self, max_features=50000, ngram_range=(1,2), C=1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        #self.model = OneVsRestClassifier(LogisticRegression(C=C, solver='liblinear')) too slow! changed
        self.model = OneVsRestClassifier(
            LogisticRegression(C=C, solver='saga', max_iter=1000),
            n_jobs=-1  # parallel
        )
        self.mlb = MultiLabelBinarizer()

    def fit(self, texts, labels, weights=None):
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)
        Y = self.mlb.fit_transform(labels)
        print("Training Logistic Regression...")
        self.model.fit(X, Y, weights)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        Y_pred = self.model.predict(X)
        return self.mlb.inverse_transform(Y_pred)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        Y_scores = self.model.predict_proba(X)
        return Y_scores

    def evaluate(self, texts, labels):
        X = self.vectorizer.transform(texts)
        Y_true = self.mlb.transform(labels)
        Y_pred = self.model.predict(X)
        micro = f1_score(Y_true, Y_pred, average='micro')
        macro = f1_score(Y_true, Y_pred, average='macro')
        return {'micro_f1': micro, 'macro_f1': macro}

    def save(self, path_prefix):
        with open(path_prefix + '_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(path_prefix + '_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(path_prefix + '_mlb.pkl', 'wb') as f:
            pickle.dump(self.mlb, f)

    def load(self, path_prefix):
        with open(path_prefix + '_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(path_prefix + '_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(path_prefix + '_mlb.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
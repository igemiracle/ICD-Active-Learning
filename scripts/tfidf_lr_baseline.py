#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle



class TFIDF_LR_Classifier:
    def __init__(self, max_features=10000, ngram_range=(1, 2), C=1.0, n_jobs=-1):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.C = C
        self.n_jobs = n_jobs
        self.mlb = MultiLabelBinarizer()
        self.models = []  # List of trained classifiers

    def _train_one(self, X, y_col):
        if np.sum(y_col == 0) > 0 and np.sum(y_col == 1) > 0:
            model = LogisticRegression(
                C=self.C,
                solver='saga',
                penalty='l2',
                max_iter=1000,
                n_jobs=1,
                random_state=42
            )
            model.fit(X, y_col)
            return model
        else:
            return None

    def fit(self, texts, labels):
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)
        Y = self.mlb.fit_transform(labels)
        print("Training Logistic Regression classifiers (parallelized)...")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_one)(X, Y[:, i]) for i in tqdm(range(Y.shape[1]), desc="Training per-label models")
        )

        self.models = results

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        predictions = []
        for model in self.models:
            if model is not None:
                pred = model.predict(X)
            else:
                pred = np.zeros(X.shape[0], dtype=int)
            predictions.append(pred)
        Y_pred = np.stack(predictions, axis=1)
        return self.mlb.inverse_transform(Y_pred)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        probabilities = []
        for model in self.models:
            if model is not None:
                prob = model.predict_proba(X)[:, 1]
            else:
                prob = np.zeros(X.shape[0])
            probabilities.append(prob)
        return np.stack(probabilities, axis=1)

    def evaluate(self, texts, labels):
        X = self.vectorizer.transform(texts)
        Y_true = self.mlb.transform(labels)
        predictions = []
        for model in self.models:
            if model is not None:
                pred = model.predict(X)
            else:
                pred = np.zeros(X.shape[0], dtype=int)
            predictions.append(pred)
        Y_pred = np.stack(predictions, axis=1)
        micro = f1_score(Y_true, Y_pred, average='micro')
        macro = f1_score(Y_true, Y_pred, average='macro')
        return {'micro_f1': micro, 'macro_f1': macro}

    def save(self, path_prefix):
        with open(path_prefix + '_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(path_prefix + '_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        with open(path_prefix + '_mlb.pkl', 'wb') as f:
            pickle.dump(self.mlb, f)

    def load(self, path_prefix):
        with open(path_prefix + '_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(path_prefix + '_models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        with open(path_prefix + '_mlb.pkl', 'rb') as f:
            self.mlb = pickle.load(f)

def main():
    df = pd.read_pickle('../data/mimic3_data_test.pkl')
    texts = df['TEXT'].tolist()
    labels = df['ICD9_CODE'].tolist()

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)


    # 初始化与训练模型
    clf = TFIDF_LR_Classifier(max_features=10000, ngram_range=(1,2), C=1.0)
    clf.fit(X_train, y_train)


    # 评估
    results = clf.evaluate(X_val, y_val)
    print("Validation Micro-F1:", results['micro_f1'])
    print("Validation Macro-F1:", results['macro_f1'])


    # 可选：保存模型
    clf.save('../models/tfidf_lr_baseline')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import csv
import pandas
import spacy
import sklearn
import sklearn.feature_extraction
import sklearn.model_selection
import sklearn.pipeline
import sklearn.naive_bayes
import sklearn.svm
import sklearn.tree

CSV_FIELDS = ['reviewRequestId', 'repository', 'reviewRequestSubmitter', 'reviewId', 'diffCommentId', 'replyId',
              'replyDiffCommentId', 'type',  'username', 'timestamp', 'text', 'isDesign', 'concept', 'hadToLookAtCode']
DATASET_PATH = '../training-data/discussions-sample-1000-annotated.csv'


def getTokensAndLemmas(nlp, text):
    tokens = nlp(text)
    tokens = [t for t in tokens if not t.is_stop]
    tokens = [t for t in tokens if not t.is_punct]
    tokens = [t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_ for t in tokens]
    return tokens


def loadDataset():
    return pandas.read_csv(DATASET_PATH)


def loadSpacy():
    return spacy.load('en_core_web_lg')


def preprocess(df):
    # df = df[~df['username'].isin(['asfbot', 'aurorabot', 'mesos-review', 'mesos-review-windows'])]
    return df


def createCountVectorizer(tokenizer):
    return sklearn.feature_extraction.text.CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))


def createTfidfVectorizer(tokenizer):
    return sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))


def main():
    print('Loading dataset...')
    df = loadDataset()
    df = preprocess(df)

    spacyNlp = loadSpacy()
    spacyTokenizer = lambda x: getTokensAndLemmas(spacyNlp, x)
    # countVectorizer = createCountVectorizer(spacyTokenizer)
    tfidfVectorizer = createTfidfVectorizer(spacyTokenizer)

    X = df['text']
    y = df['isDesign']

    decisionTreeClassifier = sklearn.tree.DecisionTreeClassifier(max_depth=10)
    decisionTreePipe = sklearn.pipeline.Pipeline([
        # ("cleaner", predictors()),
        ('vectorizer', tfidfVectorizer),
        ('classifier', decisionTreeClassifier)])

    naiveBayesClassifier = sklearn.naive_bayes.GaussianNB()
    naiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', naiveBayesClassifier)])

    mnNaiveBayesClassifier = sklearn.naive_bayes.MultinomialNB()
    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', mnNaiveBayesClassifier)])

    linearSvcClassifier = sklearn.svm.LinearSVC()
    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', linearSvcClassifier)])

    pipes = {'decisionTreePipe': decisionTreePipe,
             'gaussNaiveBayesPipe': naiveBayesPipe,
             'mnNaiveBayesPipe': mnNaiveBayesPipe,
             'linearSvcPipe': linearSvcPipe}

    # # XXX random_state is a fixed number
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    # pipe.fit(X_train, y_train)
    # sample_prediction = pipe.predict(X_test)
    #
    # for (sample, pred) in zip(X_test, sample_prediction):
    #     print(sample[:50], "Prediction=>", pred)
    #
    # print("pipe.score(X_test, y_test): ", pipe.score(X_test, y_test))
    # print("pipe.score(X_test, sample_prediction): ", pipe.score(X_test, sample_prediction))
    # print("pipe.score(X_train, y_train): ", pipe.score(X_train, y_train))

    print('Running cross validation on models...')
    for pipeName in pipes:
        scores = sklearn.model_selection.cross_validate(pipes[pipeName], X, y, cv=5,
                                                        scoring=['accuracy', 'precision', 'recall', 'f1'],
                                                        return_train_score=True)
        print('Scores for %s:' % pipeName)
        for k in scores:
            print(k, scores[k])
        print()


if __name__ == '__main__':
    main()


# #Custom transformer using spaCy
# class predictors(TransformerMixin):
#     def transform(self, X, **transform_params):
#         return [clean_text(text) for text in X]
#     def fit(self, X, y=None, **fit_params):
#         return self
#     def get_params(self, deep=True):
#         return {}
#
# # Basic function to clean the text
# def clean_text(text):
#     return text.strip().lower()
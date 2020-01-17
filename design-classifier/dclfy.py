#!/usr/bin/env python3

import os
import csv
import pandas
import spacy
import sklearn
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
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

    decisionTreeClassifier = sklearn.tree.DecisionTreeClassifier(max_depth=15)
    decisionTreePipe = sklearn.pipeline.Pipeline([
        # ("cleaner", predictors()),
        ('vectorizer', tfidfVectorizer),
        ('classifier', decisionTreeClassifier)])

    rfClassifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    rfPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', rfClassifier)])

    mnNaiveBayesClassifier = sklearn.naive_bayes.MultinomialNB()
    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', mnNaiveBayesClassifier)])

    linearSvcClassifier = sklearn.svm.LinearSVC()
    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', linearSvcClassifier)])

    gbcClassifier = sklearn.ensemble.GradientBoostingClassifier()
    gbcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', gbcClassifier)])

    logRegClassifier = sklearn.linear_model.LogisticRegression()
    logRegPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', logRegClassifier)])

    mlpClassifier = sklearn.neural_network.MLPClassifier()
    mlpPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', mlpClassifier)])

    pipes = {'Decision Tree': decisionTreePipe,
             'Random Forest': rfPipe,
             'Multinomial Naive Bayes': mnNaiveBayesPipe,
             'Linear SVC': linearSvcPipe,
             'GBC': gbcPipe,
             'Logistic Regression': logRegPipe,
             'Multi-layer Perceptron': mlpPipe,
             }

    # TEST
    # pipe = rfPipe
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=40)
    # pipe.fit(X_train, y_train)
    #
    # print("pipe.score(X_test, y_test): ", pipe.score(X_test, y_test))
    # print("pipe.score(X_train, y_train): ", pipe.score(X_train, y_train))
    #
    # y_test_pred = pipe.predict(X_test)
    # y_train_pred = pipe.predict(X_train)
    #
    # print("precision_score(y_test, y_test_pred)", sklearn.metrics.precision_score(y_test, y_test_pred))
    # print("precision_score(y_train, y_train_pred)", sklearn.metrics.precision_score(y_train, y_train_pred))
    # print("recall_score(y_test, y_test_pred)", sklearn.metrics.recall_score(y_test, y_test_pred))
    # print("recall_score(y_train, y_train_pred)", sklearn.metrics.recall_score(y_train, y_train_pred))
    # print("roc_auc_score(y_test, y_test_pred)", sklearn.metrics.roc_auc_score(y_test, y_test_pred))
    # print("roc_auc_score(y_train, y_train_pred)", sklearn.metrics.roc_auc_score(y_train, y_train_pred))

    for pipeName in pipes:
        print('Scores for %s:' % pipeName)

        scores = sklearn.model_selection.cross_validate(pipes[pipeName], X, y, cv=5,
                                                        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                                        return_train_score=True)

        for k in scores:
            print("%s: %0.3f (+/- %0.3f)" % (k, scores[k].mean(), scores[k].std() * 2))
        print()

        #XXX Clear reference to classifier to free memory
        pipes[pipeName] = None


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
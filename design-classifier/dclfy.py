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


def testSimple(pipe, X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=40)
    pipe.fit(X_train, y_train)

    print("pipe.score(X_test, y_test): ", pipe.score(X_test, y_test))
    print("pipe.score(X_train, y_train): ", pipe.score(X_train, y_train))

    y_test_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    print("precision_score(y_test, y_test_pred)", sklearn.metrics.precision_score(y_test, y_test_pred))
    print("precision_score(y_train, y_train_pred)", sklearn.metrics.precision_score(y_train, y_train_pred))
    print("recall_score(y_test, y_test_pred)", sklearn.metrics.recall_score(y_test, y_test_pred))
    print("recall_score(y_train, y_train_pred)", sklearn.metrics.recall_score(y_train, y_train_pred))
    print("roc_auc_score(y_test, y_test_pred)", sklearn.metrics.roc_auc_score(y_test, y_test_pred))
    print("roc_auc_score(y_train, y_train_pred)", sklearn.metrics.roc_auc_score(y_train, y_train_pred))


def compareDefaultModels(X, y, tfidfVectorizer):
    decisionTreePipe = sklearn.pipeline.Pipeline([
        # ("cleaner", predictors()),
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.tree.DecisionTreeClassifier(max_depth=15))
    ])

    rfPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.ensemble.RandomForestClassifier(n_estimators=100))
    ])

    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.naive_bayes.MultinomialNB())
    ])

    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.svm.LinearSVC())
    ])

    gbcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.ensemble.GradientBoostingClassifier())
    ])

    logRegPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.linear_model.LogisticRegression())
    ])

    mlpPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.neural_network.MLPClassifier())
    ])

    pipes = {'Decision Tree': decisionTreePipe,
             'Random Forest': rfPipe,
             'Multinomial Naive Bayes': mnNaiveBayesPipe,
             'Linear SVC': linearSvcPipe,
             'GBC': gbcPipe,
             'Logistic Regression': logRegPipe,
             'Multi-layer Perceptron': mlpPipe,
             }

    for pipeName in pipes:
        print('Scores for %s:' % pipeName)

        scores = sklearn.model_selection.cross_validate(pipes[pipeName], X, y, cv=5,
                                                        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                                        return_train_score=True)

        for k in scores:
            print("%s: %0.3f (+/- %0.3f)" % (k, scores[k].mean(), scores[k].std() * 2))
        print()

        # XXX Clear reference to classifier to free memory
        pipes[pipeName] = None


def compareBestModels(X, y, tfidfVectorizer):
    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.svm.LinearSVC())
    ])
    linearSvcParams = {'classifier__C': [0.1, 0.9, 1, 1.1, 2],
                  'classifier__loss': ['hinge', 'squared_hinge'],
                  # 'classifier_dual': [False, True],
                  'classifier__tol': [1e-2, 1e-4],
                  'classifier__max_iter': [1000, 10000]}

    mlpPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.neural_network.MLPClassifier())
    ])
    mlpParams = {'classifier__solver': ['adam', 'lbfgs'],
                 'classifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                 'classifier__activation': ['tanh', 'relu'],
                 # 'classifier__max_iter': [100, 200, 300],
                 # 'classifier__alpha': [0.0001, 0.05],
                 'classifier__learning_rate': ['constant', 'adaptive'],
                 }

    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.naive_bayes.MultinomialNB())
    ])
    mnNaiveBayesParams = {'classifier__alpha': [0, 0.5, 0.9, 1.0, 1.1, 1.5]}

    logRegPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.linear_model.LogisticRegression())
    ])
    logRegParams = {'classifier__C': [0.1, 0.9, 1, 1.1, 2],
                    'classifier__solver': ['lbfgs', 'liblinear', 'saga'],
                    'classifier__max_iter': [100, 200]
                    }

    cvs = {
        # 'MLP': sklearn.model_selection.GridSearchCV(mlpPipe, mlpParams, cv=5, scoring='f1', return_train_score=True),
        # 'Multinomial Naive Bayes': sklearn.model_selection.GridSearchCV(mnNaiveBayesPipe, mnNaiveBayesParams, cv=5, scoring='f1', return_train_score=True),
        # 'Linear SVC': sklearn.model_selection.GridSearchCV(linearSvcPipe, linearSvcParams, cv=5, scoring='f1', return_train_score=True),
        'Logistic Regression': sklearn.model_selection.GridSearchCV(logRegPipe, logRegParams, cv=5, scoring='f1', return_train_score=True),
    }

    for cvName in cvs:
        print('Hyperparameters search for %s' % cvName)

        cvs[cvName].fit(X, y)

        print("cv.best_score_", cvs[cvName].best_score_)
        print("cv.best_params_", cvs[cvName].best_params_)

        #XXX Clear reference to classifier to free memory
        cvs[cvName] = None


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

    #compareDefaultModels(X, y, tfidfVectorizer)
    compareBestModels(X, y, tfidfVectorizer)


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
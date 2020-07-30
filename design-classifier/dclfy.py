#!/usr/bin/env python3

import sys
import os
import csv
import joblib
import numpy
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

# START WORKAROUND https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
# END WORKAROUND https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072

CSV_FIELDS = ['reviewRequestId', 'repository', 'reviewRequestSubmitter', 'reviewId', 'diffCommentId', 'replyId',
              'replyDiffCommentId', 'type',  'username', 'timestamp', 'text', 'isDesign']
TRAINING_DATASET_PATH = '../training-data/discussions-sample-1000-annotated.csv'
DATASET_PATH = '../preprocess-dataset/output/reviews.csv'
OUTPUT_PATH = 'output/reviews-predicted.csv'


def getTokensAndLemmas(nlp, text):
    tokens = nlp(text)
    tokens = [t for t in tokens if not t.is_stop]
    tokens = [t for t in tokens if not t.is_punct]
    tokens = [t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_ for t in tokens]
    return tokens


def loadTrainingDataset():
    return pandas.read_csv(TRAINING_DATASET_PATH)


def loadSpacy():
    return spacy.load('en_core_web_lg')


# def preprocess(df):
    # df = df[~df['username'].isin(['asfbot', 'aurorabot', 'mesos-review', 'mesos-review-windows'])]
    # return df


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


def searchBestParameters(X, y, tfidfVectorizer):
    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.svm.LinearSVC())
    ])
    # linearSvcParams = {'classifier__C': [0.1, 0.9, 1, 1.1, 2],
    #               'classifier__loss': ['hinge', 'squared_hinge'],
    #               # 'classifier_dual': [False, True],
    #               'classifier__tol': [1e-2, 1e-4],
    #               'classifier__max_iter': [1000, 10000]}
    # linearSvcParams = {'classifier__C': [1.05, 1.1, 1.2, 1.5],
    #                    'classifier__loss': ['hinge'],
    #                    'classifier__tol': [1e-1, 1e-2],
    #                    'classifier__max_iter': [1000, 1200]}
    linearSvcParams = {'classifier__C': [1.025, 1.05, 1.075],
                       'classifier__loss': ['hinge'],
                       'classifier__tol': [0.75, 0.5, 0.2, 0.15, 1e-1],
                       'classifier__max_iter': [1000]}

    mlpPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.neural_network.MLPClassifier())
    ])
    # mlpParams = {'classifier__solver': ['adam', 'lbfgs'],
    #              'classifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    #              'classifier__activation': ['tanh', 'relu'],
    #              # 'classifier__max_iter': [100, 200, 300],
    #              # 'classifier__alpha': [0.0001, 0.05],
    #              'classifier__learning_rate': ['constant', 'adaptive'],
    #              }
    # mlpParams = {'classifier__solver': ['adam'],
    #              'classifier__hidden_layer_sizes': [(50, 100, 50), (100, 100, 100)],
    #              'classifier__activation': ['tanh'],
    #              'classifier__learning_rate': ['constant'],
    #              }
    mlpParams = {'classifier__solver': ['adam'],
                 'classifier__hidden_layer_sizes': [(100, 100, 100), (150, 150, 150)],
                 'classifier__activation': ['tanh'],
                 'classifier__learning_rate': ['constant'],
                 }

    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.naive_bayes.MultinomialNB())
    ])
    # mnNaiveBayesParams = {'classifier__alpha': [0, 0.01, 0.9, 1.0, 1.1, 1.5]}
    # mnNaiveBayesParams = {'classifier__alpha': [0, 0.1, 0.01]}
    mnNaiveBayesParams = {'classifier__alpha': [0.05, 0.1, 0.15]}

    logRegPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.linear_model.LogisticRegression())
    ])
    # logRegParams = {'classifier__C': [0.1, 0.9, 1, 1.1, 2],
    #                 'classifier__solver': ['lbfgs', 'liblinear', 'saga'],
    #                 'classifier__max_iter': [100, 200]
    #                 }
    # logRegParams = {'classifier__C': [1.9, 2, 2.1, 3, 5, 10],
    #                 'classifier__solver': ['saga'],
    #                 }
    logRegParams = {'classifier__C': [8, 10, 20, 50, 100],
                    'classifier__solver': ['saga'],
                    'classifier__max_iter': [1000]
                    }

    cvs = {
        'Linear SVC': sklearn.model_selection.GridSearchCV(linearSvcPipe, linearSvcParams, cv=5, scoring='f1', return_train_score=True),
        'MLP': sklearn.model_selection.GridSearchCV(mlpPipe, mlpParams, cv=5, scoring='f1', return_train_score=True),
        'Multinomial Naive Bayes': sklearn.model_selection.GridSearchCV(mnNaiveBayesPipe, mnNaiveBayesParams, cv=5, scoring='f1', return_train_score=True),
        'Logistic Regression': sklearn.model_selection.GridSearchCV(logRegPipe, logRegParams, cv=5, scoring='f1', return_train_score=True),
    }

    for cvName in cvs:
        print('Hyperparameters search for %s' % cvName)

        cvs[cvName].fit(X, y)

        print("cv.best_score_", cvs[cvName].best_score_)
        print("cv.best_params_", cvs[cvName].best_params_)

        #FIXME Clear reference to classifier to free memory
        cvs[cvName] = None


def buildBestModels(tfidfVectorizer):
    linearSvcPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.svm.LinearSVC(C=1.05, loss='hinge', tol=0.5, max_iter=1000))
    ])

    mlpPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier',
         sklearn.neural_network.MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100), activation='tanh',
                                              learning_rate='constant'))
    ])

    mnNaiveBayesPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.naive_bayes.MultinomialNB(alpha=0.1))
    ])

    logRegPipe = sklearn.pipeline.Pipeline([
        ('vectorizer', tfidfVectorizer),
        ('classifier', sklearn.linear_model.LogisticRegression(C=50, max_iter=1000, solver='saga'))
    ])

    return {
        'Linear SVC': linearSvcPipe,
        'MLP': mlpPipe,
        'Multinomial Naive Bayes': mnNaiveBayesPipe,
        'Logistic Regression': logRegPipe,
    }


def compareBestModels(X, y, tfidfVectorizer):
    pipes = buildBestModels(tfidfVectorizer)

    for pipeName in pipes:
        print('Scores for %s:' % pipeName)

        scores = sklearn.model_selection.cross_validate(pipes[pipeName], X, y, cv=5,
                                                        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                                        return_train_score=True)

        for k in scores:
            print("%s: %0.3f (+/- %0.3f)" % (k, scores[k].mean(), scores[k].std() * 2))
        print()

        # FIXME Clear reference to classifier to free memory
        pipes[pipeName] = None


def compareEnsembleOfBestModels(X, y, tfidfVectorizer):
    estimators = list(buildBestModels(tfidfVectorizer).items())
    estimators = [x for x in estimators if x[0] != 'Linear SVC'] # LinearSVC crashes the ensemble b/c it doesn't support predict_proba() and decision_function()
    mergedPipe = sklearn.ensemble.VotingClassifier(estimators=estimators, voting='soft')
    scores = sklearn.model_selection.cross_validate(mergedPipe, X, y, cv=5,
                                                    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                                    return_train_score=True)
    for k in scores:
        print("%s: %0.3f (+/- %0.3f)" % (k, scores[k].mean(), scores[k].std() * 2))


def getModuleVersions():
    # TODO Do this automatically.
    return ['joblib ' + joblib.__version__, 'pandas ' + pandas.__version__, 'spacy ' + spacy.__version__,
            'sklearn ' + sklearn.__version__]


def serializeBestModels(X, y, tfidfVectorizer):
    # TODO: Fix bug with Spacy (see https://stackoverflow.com/questions/53885198/using-spacy-as-tokenizer-in-sklearn-pipeline)
    raise Exception('not implemented')
    # for name, pipe in buildBestModels(tfidfVectorizer).items():
    #     pipe.fit(X, y)
    #     id_ =  name.lower().replace(' ', '-')
    #     joblib.dump(pipe, id_ + '.joblib')
    #     with open(id_ + '.meta', 'w') as fin:
    #         fin.write('\n'.join(getModuleVersions()))


def unserializeModels():
    models = []
    for root, dirs, files in os.walk(os.curdir):
        for f in files:
            if f.endswith('joblib'):
                path = os.path.join(root, f)
                pipe = joblib.load(path)
                models.append((f.split('.')[0], pipe))
    return models


'''
Pre-condition: numbers in row must be int (not str).
'''
def getRowFromTrainingDataset(trainingDf, row):
    # The dataset has no unique row ids so we have to resort to using the natural key to find a row.
    NATURAL_KEY_FIELDS = ('reviewRequestId', 'repository', 'reviewRequestSubmitter', 'reviewId', 'diffCommentId', 'replyId', 'replyDiffCommentId')

    df = trainingDf
    cmp = df['reviewRequestId'] == df['reviewRequestId'] # FIXME Use better method to create array filled with true.
    for k in NATURAL_KEY_FIELDS:
        if row[k]:
            cmp = cmp & (df[k] == row[k])
        else:
            cmp = cmp & (pandas.isna(df[k]))

    resultDf = df.loc[cmp]

    return dict(resultDf.iloc[0]) if not resultDf.empty else None


def classify(classifier, trainingDf, truncateLength=32767):
    def convertNumbersToInt(row):
        for k, v in row.items():
            if k.endswith('Id'):
                row[k] = int(v) if v else ''

    with open(OUTPUT_PATH, 'w') as fout:
        dcfout = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        dcfout.writeheader()

        with open(DATASET_PATH, 'r') as fin:
            dcfin = csv.DictReader(x.replace('\0', '') for x in fin)  # ignore NULL bytes, which crash the csv.reader
            for row in dcfin:
                convertNumbersToInt(row)
                if truncateLength > 0:
                    row['text'] = row['text'][:truncateLength]
                trainingDfRow = getRowFromTrainingDataset(trainingDf, row)
                row['isDesign'] = trainingDfRow['isDesign'] if trainingDfRow else classifier.predict((row['text'],))[0]
                dcfout.writerow(row)


def main():
    print('Loading dataset...')
    df = loadTrainingDataset()

    spacyNlp = loadSpacy()
    spacyTokenizer = lambda x: getTokensAndLemmas(spacyNlp, x)

    tfidfVectorizer = createTfidfVectorizer(spacyTokenizer)

    X = df['text']
    y = df['isDesign']

    # compareDefaultModels(X, y, tfidfVectorizer)
    # searchBestParameters(X, y, tfidfVectorizer)
    # compareBestModels(X, y, tfidfVectorizer)
    # compareEnsembleOfBestModels(X, y, tfidfVectorizer)
    # serializeBestModels(X, y, tfidfVectorizer)

    classifier = buildBestModels(tfidfVectorizer)['MLP']
    print("Fitting model...")
    classifier.fit(X, y)
    print("Classifying dataset...")
    classify(classifier, df)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import json, datetime, time, re, random, csv, os, statistics
from flask import Flask, render_template, send_from_directory, request, abort, redirect, url_for, flash, g
from werkzeug.utils import secure_filename

import numpy as np
import pyswarms as ps
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

######################################################
##                  FLASK APP                       ##
######################################################

app = Flask(__name__)

app.secret_key = 'XpCEnPCz7vFyKEofTGgFi2TB0HGpjlUaNQCBIo37u'
app.config['ENV'] = 'development'

######################################################
##                  STATIC FILES                    ##
######################################################
@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('static/', path)

def get_dataset():
    dataset = []

    try:
        f = open('dataset/dataset.csv', 'r')

        with f:
            reader = csv.DictReader(f)

            for row in reader:
                
                dataset.append([
                    float(row['Category 1']),
                    float(row['Category 2']),
                    float(row['Category 3']),
                    float(row['Category 4']),
                    float(row['Category 5']),
                    float(row['Category 6']),
                    float(row['Category 7']),
                    float(row['Category 8']),
                    float(row['Category 9']),
                    float(row['Category 10'])
                ])
                

        return dataset

    except Exception as e:
        print(e)
        return False

######################################################
##                  API ENDPOINTS                   ##
######################################################

@app.route('/api/dataset', methods=["GET","POST"])
def get_api_dataset():
    if request.method == "GET":
        
        dataset = []
        try:
            f = open('dataset/dataset.csv', 'r')

            with f:
                reader = csv.DictReader(f)

                for row in reader:
                    dataset.append(row)

            return json.dumps({
                'status':'ok',
                'result': dataset
            })

        except Exception as e:
            print(e)
            return json.dumps({
                'status':'error',
                'msg': str(e)
            })
        
        
    else:
        data = request.files["dataset"]

        if data:
            filename = secure_filename(data.filename)
            data.save(os.path.join('dataset', 'dataset.csv'))
            
            return json.dumps({
                'status': 'ok'
            })
        else:
            return json.dumps({
                'status': 'error',
                'msg': 'No file selected'
            })

@app.route('/api/preprocessing')
def preprocessing():
    dataset = get_dataset()
    if dataset == False:
        return json.dumps({'status':'error'})

    dataset = np.array(dataset, dtype=np.float32)

    scaler = MinMaxScaler()
    result = scaler.fit_transform(dataset).tolist()

    local_data = open('dataset/normalized', 'w').write(json.dumps(result))

    return json.dumps({
        'status': 'ok',
        'result': result
    })

@app.route('/api/discretization')
def discretization():
    dataset = get_dataset()
    if dataset == False:
        return json.dumps({'status':'error'})

    dataset = np.array(dataset, dtype=np.float32)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(dataset)

    discrete = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    discrete.fit(normalized)
    result = discrete.transform(normalized)

    return json.dumps({
        'status': 'ok',
        'result': result.tolist()
    })

@app.route('/api/kmeans')
def kmeans():

    n_cluster = int(request.args.get('k'))

    try:
        normalized = json.loads(open('dataset/normalized', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Normalisasi Dataset belum dilakukan'})

    X = np.array(normalized)
    discrete = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    discrete.fit(X)
    X = discrete.transform(X)
    
    km = KMeans(n_clusters=2, random_state=1)
    cluster = km.fit(X)

    transformed = km.transform(X).tolist()

    label = cluster.labels_.tolist()
    result = cluster.cluster_centers_.tolist()

    open('dataset/last_k_means', 'w').write(str(n_cluster))

    return json.dumps({
        'status': 'ok',
        'result': {
            'label': label,
            'value': transformed,
            'centroid': result
        }
    })

@app.route('/api/pso')
def pso():

    c1 = float(request.args.get('c1'))
    c2 = float(request.args.get('c2'))
    w = float(request.args.get('w'))
    n_particle = int(request.args.get('n_particle'))
    iteration = int(request.args.get('iteration'))

    try:
        normalized = json.loads(open('dataset/normalized', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Normalisasi Dataset belum dilakukan'})

    try:
        n_cluster = int(open('dataset/last_k_means', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Clustering belum dilakukan'})

    X = np.array(normalized, dtype=np.float32)
    discrete = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    discrete.fit(X)
    X = discrete.transform(X)
    cluster = KMeans(n_clusters=2, random_state=1).fit(X)

    y = cluster.labels_

    classifier = GaussianNB()

    # Define objective function
    def f_per_particle(m, alpha):
        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        total_features = 10
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = X
        else:
            X_subset = X[:,m==1]
        # Perform classification and store performance in P
        classifier.fit(X_subset, y)
        P = (classifier.predict(X_subset) == y).mean()
        # Compute for the objective function
        j = (alpha * (1.0 - P)
            + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
        return j

    def f(x, alpha=1.0):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    np.random.seed(111111)
    options = {'c1': 2, 'c2': 2, 'w':0.9, 'k': 10, 'p':2}
    dimensions = 10
    optimizer = ps.discrete.BinaryPSO(n_particles=50, dimensions=dimensions, options=options)

    _, pos = optimizer.optimize(f, iters=100)

    open('dataset/pso_result', 'w').write(json.dumps(pos.tolist()))

    return json.dumps({
        'status':'ok',
        'result': pos.tolist()
    })

@app.route('/api/naive_bayes')
def naive_bayes():

    dataset = get_dataset()
    if dataset == False:
        return json.dumps({'status':'error'})

    X_raw = []
    for d in dataset:
        X_raw.append(d[1:])

    X_raw = np.array(X_raw, dtype=np.float32)

    try:
        normalized = json.loads(open('dataset/normalized', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Normalisasi Dataset belum dilakukan'})

    try:
        n_cluster = int(open('dataset/last_k_means', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Clustering belum dilakukan'})

    X = np.array(normalized)

    discrete = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
    discrete.fit(X)
    X = discrete.transform(X)

    cluster_raw = KMeans(n_clusters=2, random_state=1).fit(X_raw)
    cluster = KMeans(n_clusters=2, random_state=1).fit(X)

    y_raw = cluster_raw.labels_
    y = cluster.labels_

    selected_features = json.loads(open('dataset/pso_result', 'r').read())

    selected_X = []
    for data in X:
        subset = []
        for f in range(len(data)):
            if selected_features[f] == 1:
                subset.append(data[f])

        selected_X.append(subset)

    X = np.array(selected_X)

    classifier = GaussianNB()
    kf = KFold(n_splits=10,random_state=11111,shuffle=True)

    cf_matrix_raw = []
    cf_matrix = []
    accuracy_raw = []
    accuracy = []

    for train_index, test_index in kf.split(X_raw):

        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]

        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        
        cf = confusion_matrix(y_test, predicted, labels=[0,1,2,3])
        fp = cf.sum(axis=0) - np.diag(cf)
        fn = cf.sum(axis=1) - np.diag(cf)
        tp = np.diag(cf)
        tn = cf.sum() - (fp + fn + tp)
        cf_matrix_raw.append({
            'tn': sum(tn.astype(int).tolist()),
            'fp': sum(fp.astype(int).tolist()),
            'fn': sum(fn.astype(int).tolist()),
            'tp': sum(tp.astype(int).tolist())
        })
        accuracy_raw.append(accuracy_score(y_test, predicted))

    classifier = GaussianNB()

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        
        cf = confusion_matrix(y_test, predicted, labels=[0,1,2,3])
        fp = cf.sum(axis=0) - np.diag(cf)
        fn = cf.sum(axis=1) - np.diag(cf)
        tp = np.diag(cf)
        tn = cf.sum() - (fp + fn + tp)
        cf_matrix.append({
            'tn': sum(tn.astype(int).tolist()),
            'fp': sum(fp.astype(int).tolist()),
            'fn': sum(fn.astype(int).tolist()),
            'tp': sum(tp.astype(int).tolist())
        })
        accuracy.append(accuracy_score(y_test, predicted))

    return json.dumps({
        'status': 'ok',
        'result': {
            'cf_matrix': cf_matrix,
            'accuracy': accuracy,
        },
        'result_raw': {
            'cf_matrix': cf_matrix_raw,
            'accuracy': accuracy_raw,
        }
    })

@app.route('/api/detail')
def detail():

    try:
        normalized = json.loads(open('dataset/normalized', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Normalisasi Dataset belum dilakukan'})

    try:
        n_cluster = int(open('dataset/last_k_means', 'r').read())
    except FileNotFoundError:
        return json.dumps({'status':'error', 'msg': 'Clustering belum dilakukan'})

    X = np.array(normalized)
    cluster = KMeans(n_clusters=2, random_state=1).fit(X)

    y = cluster.labels_

    classifier = GaussianNB()
    kf = KFold(n_splits=10)

    result = []
    ids = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        
        ids.append([test_index.tolist()])
        result.append(classifier.predict_proba(X_test).tolist())

    return json.dumps({
        'status': 'ok',
        'result': result,
        'index': ids
    })

######################################################
##                      VIEWS                       ##
######################################################

@app.route('/')
def page_home():
    return render_template('index.html')

@app.route('/dataset')
def page_dataset():
    return render_template('dataset.html')

@app.route('/preprocessing')
def page_preprocessing():
    return render_template('preprocessing.html')

@app.route('/discretization')
def page_discretization():
    return render_template('discretization.html')

@app.route('/kmeans')
def page_kmeans():
    return render_template('kmeans.html')

@app.route('/pso')
def page_pso():
    return render_template('pso.html')

@app.route('/naive_bayes')
def page_naive_bayes():
    return render_template('naive_bayes.html')

@app.route('/result')
def page_result():
    return render_template('result.html')

@app.route('/detail')
def page_detail():
    return render_template('detail.html')

if __name__ == "__main__":
    app.run()
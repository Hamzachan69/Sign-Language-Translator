"""
Custom estimator(s) shared by training and the live app (must be importable by both,
otherwise joblib cannot load the saved model).
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KMeansVoteClassifier:
    """Unsupervised K-Means + majority-vote labelling, with predict / predict_proba.

    1. Cluster the (scaled) landmark features WITHOUT labels.
    2. Give every cluster the most common true label among its training members.
    3. Predict = label of the nearest cluster.
    Using more clusters than classes (e.g. 108 for 36 classes) lets one letter own
    several clusters, which is far more accurate than exactly one cluster per letter.
    """

    def __init__(self, n_clusters=108, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.n_classes_ = int(y.max()) + 1
        self.pipe_ = Pipeline([
            ("scaler", StandardScaler()),
            ("km", MiniBatchKMeans(n_clusters=self.n_clusters, n_init=5, max_iter=300,
                                   batch_size=2048, random_state=self.random_state)),
        ]).fit(X)                                   # labels NOT used for clustering
        ids = self.pipe_.named_steps["km"].labels_
        self.cluster_map_ = np.zeros(self.n_clusters, dtype=np.int32)
        for c in range(self.n_clusters):
            m = ids == c
            if m.any():
                self.cluster_map_[c] = np.argmax(np.bincount(y[m], minlength=self.n_classes_))
        return self

    def _dists(self, X):
        return self.pipe_.transform(X)              # (n, n_clusters) distances to centroids

    def predict(self, X):
        return self.cluster_map_[np.argmin(self._dists(X), axis=1)]

    def predict_proba(self, X):
        d = self._dists(X)
        cls_d = np.full((len(d), self.n_classes_), np.inf, dtype=np.float64)
        for c in range(self.n_clusters):            # nearest cluster per class
            k = self.cluster_map_[c]
            cls_d[:, k] = np.minimum(cls_d[:, k], d[:, c])
        z = -(cls_d - cls_d.min(axis=1, keepdims=True))
        e = np.exp(z)                               # exp(-inf) = 0 for classes with no cluster
        return e / e.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())

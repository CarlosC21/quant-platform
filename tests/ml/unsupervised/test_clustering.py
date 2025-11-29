import pandas as pd
from quant_platform.ml.unsupervised.clustering import KMeansClustering


def test_kmeans_clustering_basic():
    df = pd.DataFrame({"x": [1, 2, 10, 11]})

    model = KMeansClustering(n_clusters=2).fit(df)

    labels = model.predict(df)
    assert labels.shape == (4,)
    assert set(labels) == {0, 1}

    emb = model.transform(df)
    assert emb.shape == (4, 2)  # distances to 2 centers

    scores = model.score_samples(df)
    assert scores.shape == (4,)

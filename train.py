
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
batch_size = 128
init = "nndsvda"


def get_tf_vectorizer(samples=None, write_model=False, filename="vectorizer.pkl", **kwargs):
    """Get a term-frequency vectorizer. If no samples are provided, loads a pretrained model from filename.
    If write_model is True, write the model to filename."""

    if samples is None:
        print("Loading pretrained vectorizer...")
        with open(filename, "rb") as f:
            return pickle.load(f)

    # parameters
    kwargs.setdefault("max_df", 0.02)
    kwargs.setdefault("min_df", 2)
    kwargs.setdefault("max_features", n_features)
    kwargs.setdefault("stop_words", "english")

    print("Initializing vectorizer...")
    tf_vectorizer = CountVectorizer(**kwargs)

    print("Fitting vectorizer...")
    _ = tf_vectorizer.fit_transform(samples)

    if write_model:
        print("Saving vectorizer...")
        with open(filename, "wb") as f:
            pickle.dump(tf_vectorizer, f)

    return tf_vectorizer


def get_lda(samples=None, write_model=False, filename="lda.pkl", **kwargs):
    """Get a LDA model. If no samples (term-frequency documents) are provided, loads a pretrained model from filename.
    If write_model is True, write the model to filename."""

    if samples is None:
        print("Loading pretrained LDA model...")
        with open(filename, "rb") as f:
            return pickle.load(f)

    # parameters
    kwargs.setdefault("max_iter", 5)
    kwargs.setdefault("n_components", n_components)
    kwargs.setdefault("learning_method", "batch")  # vs online
    kwargs.setdefault("learning_offset", 50.0)
    kwargs.setdefault("random_state", 0)

    print("Initializing LDA model...")
    lda = LatentDirichletAllocation(**kwargs)

    print("Fitting LDA model...")
    lda.fit(samples)

    if write_model:
        print("Saving LDA model...")
        with open(filename, "wb") as f:
            pickle.dump(lda, f)

    return lda

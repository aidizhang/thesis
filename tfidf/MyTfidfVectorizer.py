from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re

class MyTfidfVectorizer:
    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary

    """
    Builds a preprocessor that converts to lowercase
    """
    def build_preprocessor(self):
        return lambda x: x.lower()
    
    """
    Returns a function that splits a string into a sequence of tokens
    """
    def build_tokenizer(self):
        token_pattern = '(?u)\\b\\w\\w+\\b'
        token_pattern = re.compile(token_pattern)
        return lambda doc: token_pattern.findall(doc)
    
    """
    Turns stop words into sequence of n-grams after stop words filtering
    """
    def word_ngrams(self, tokens, stop_words=None):
        return
    
    """
    Build analyzer for documents
    """
    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()
        return lambda doc: tokenize(preprocess(doc))

    """
    Count number of non-zero values for each feature in sparse X
    """
    def doc_freq(self, X):
        if sp.issparse(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(sp.csc_matrix(X, copy=False).indptr)
    
    """
    Fit idf vectors
    """
    def fit_idf(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        n_samples, n_features = X.shape
        df = self.doc_freq(X)
        
        # perform idf smoothing
        df += 1
        n_samples += 1
        
        # log+1 instead of log makes sure terms with zero freq don't get suppressed
        # completely
        idf = np.log(float(n_samples) / df) + 1.0
        self.idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features, format='csr')
    
    """
    Creates sparse feature matrix, and vocabulary if fixed_vocab is False
    """
    def count_vocab(self, rawdocs, fixed_vocab):
        if fixed_vocab:
            vocabulary = self.vocabulary
        else:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__
        
        analyze = self.build_analyzer()
        indices = []
        values = []
        indptr = [0]
        numdocs = 0
        for doc in rawdocs:
            numdocs += 1
            word_counter = {}
            for word in analyze(doc):
                try:
                    word_idx = vocabulary[word]
                    if word_idx not in word_counter:
                        word_counter[word_idx] = 1
                    else:
                        word_counter[word_idx] += 1
                except KeyError:
                    continue
            indices.extend(word_counter.keys())
            values.extend(word_counter.values())
            indptr.append(len(indices))
        
        if not fixed_vocab:
            # disable default dict behavior
            vocabulary = dict(vocabulary)
        
        indices = np.asarray(indices, dtype=np.intc)
        indptr = np.asarray(indptr, dtype=np.intc)
        values = np.asarray(values, dtype=np.intc)
        X = csr_matrix((values, indices, indptr), shape=(len(indptr)-1, len(vocabulary)), dtype=np.float64)
        X.sort_indices()
        return vocabulary, X

    """
    Builds vocabulary from raw documents.
    """
    def fit(self, rawdocs):
        new_vocabulary, X = self.count_vocab(rawdocs, False)
        self.vocabulary = new_vocabulary
        self.fit_idf(X) # populate self.idf_diag
        return self

    """
    Transforms into document-term matrix. Extracts token counts out of raw text
    documents using the vocabulary fitted with fit.
    """
    def transform(self, rawdocs):
        new_vocabulary, X = self.count_vocab(rawdocs, True)
        
        # sanity check
        expected_n_features = self.idf_diag.shape[0]
        assert(expected_n_features == X.shape[1])
        
        # transform count matrix to tf-idf representation
        X = X * self.idf_diag
        # normalize by row
        X = normalize(X, norm='l2', axis=1)
        return X
import codecs
import argparse
import h5py
import sys
import re
import csv
import collections
import numpy as np
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

max_context_len = 160
max_utterance_len = 80

FILE_PATHS = ("data/train.csv",
                "data/valid.csv",
                "data/test.csv",
                "data/glove.6B.50d.txt")
args = {}
word_to_idx = {}
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

"""
Lower, stem, tokenize word
"""
def transform_word(word):
    lw = word.lower()
    tok = stemmer.stem(lw)
    tw = lemmatizer.lemmatize(tok, pos='v')
    return tw

"""
Get GloVe embeddings map: transformed word -> embedding (dim 50).
"""
def get_embeddings(embed_file):
    emb_map = {}
    with codecs.open(embed_file, "r", encoding="utf-8") as f:
        for line in f:
            cemb = line.split()
            new_word = transform_word(cemb[0])
            if new_word not in emb_map:
                emb_map[new_word] = np.array(cemb[1:], dtype=np.float32)
    return emb_map

"""
Saves word_to_idx map into .txt file; 
"""
def save_word_to_idx(word_to_idx):
    if args.include_missing:
        np.save('word_to_idx_missing', word_to_idx)
    else:
        np.save('word_to_idx', word_to_idx)

def save_idx_to_embedding(idx_to_embedding):
    idx = range(len(idx_to_embedding))
    idx2emb_map = dict(zip(idx, idx_to_embedding))
    if args.include_missing:
        np.save('idx_to_embedding_missing', idx2emb_map)
    else:
        np.save('idx_to_embedding', idx2emb_map)

ZERO_IDX = 0

"""
Get all words that appear in Ubuntu datasets (train/valid/test).
Returns word index to embedding numpy array.
"""
def get_vocab(filelist, embeddings):
    idx_to_embedding = []
    # zero token for padding
    idx_to_embedding.append(np.zeros(50))
    idx = 1
    missing = collections.Counter()
    for filename in filelist:
        with open(filename, "rb") as f:
            df = pd.read_csv(filename)
            # iterate through relevant columns of dataframe
            for col in df.columns:
                if "Context" in col or "Utterance" in col or "Distractor" in col:
                    for line in df[col].values:
                        words = line.split()
                        for w in words:
                            # if word has not been encountered yet but in our embeddings
                            if w in embeddings:
                                if w not in word_to_idx:
                                    word_to_idx[w] = idx
                                    idx_to_embedding.append(embeddings[w])
                                    idx += 1
                            else:
                                if w not in word_to_idx:
                                    word_to_idx[w] = idx
                                    idx_to_embedding.append(np.zeros(50))
                                    idx += 1
                                missing[w] += 1

    # save word/idx/embedding maps to file
    save_word_to_idx(word_to_idx)
    save_idx_to_embedding(idx_to_embedding)
    
    print len(word_to_idx), "words have been counted"
    print len(missing), "unknown words"
    print missing.most_common(20)
    print len(idx_to_embedding), "len of idx_to_embedding"
    return np.array(idx_to_embedding, dtype=np.float32)

"""
Pad or truncate vector of word indices to fit length.
"""
def pad_truncate(word_idx, max_len):
    if (len(word_idx) < max_len):
        word_idx += [ZERO_IDX for i in xrange(max_len-len(word_idx))]
    else:
        word_idx = word_idx[:max_len]
    return word_idx

"""
Gets fixed length of train word indices for (Context, Utterance, Label).
"""
def convert_train(filename):
    df = pd.read_csv(filename)

    # get contexts
    contexts = []
    for c in df.Context.values:
        context_idx = []
        for w in c.split():
            if w in word_to_idx:
                context_idx.append(word_to_idx[w])
            else:
                print w, "not found in train"
        context_idx = pad_truncate(context_idx, max_context_len)
        contexts.append(context_idx)

    # get utterances
    utterances = []
    for u in df.Utterance.values:
        utter_idx = []
        for w in u.split():
            if w in word_to_idx:
                utter_idx.append(word_to_idx[w])
            else:
                print w, "not found in train"
        utter_idx = pad_truncate(utter_idx, max_utterance_len)
        utterances.append(utter_idx)

    npcontexts = np.array(contexts, dtype=np.int)
    nputterances = np.array(utterances, dtype=np.int)
    
    # get targets
    nptargets = np.array(df.Label.values, dtype=np.int)
    print len(npcontexts), len(nputterances), len(nptargets)
    return npcontexts, nputterances, nptargets

"""
Gets fixed legnth of test/validation word indices for
(Context, Utterance, Distractors).
"""
def convert_test(filename):
    df = pd.read_csv(filename)

    # get contexts
    contexts = []
    for c in df.Context.values:
        context_idx = []
        for w in c.split():
            if w in word_to_idx:
                context_idx.append(word_to_idx[w])
            else:
                print w, "not found"
        context_idx = pad_truncate(context_idx, max_context_len)
        contexts.append(np.array(context_idx, dtype=np.int))

    # get ground truth utterances
    utterances = []
    for u in df.iloc[:,1].values:
        utter_idx = []
        for w in u.split():
            if w in word_to_idx:
                utter_idx.append(word_to_idx[w])
            else:
                print w, "not found"
        utter_idx = pad_truncate(utter_idx, max_utterance_len)
        utterances.append(np.array(utter_idx, dtype=np.int))

    # get all distractors
    all_distractors = []
    for i in xrange(9):
        distractors = []
        # iterate through all phrases in distractor_i
        for j in xrange(len(df)):
            dist_idx = []
            dist = df.iloc[j, i+2]
            for w in dist.split():
                if w in word_to_idx:
                    dist_idx.append(word_to_idx[w])
                else:
                    print w, "not found"
            dist_idx = pad_truncate(dist_idx, max_utterance_len)
            distractors.append(dist_idx)
        all_distractors.append(np.array(distractors, dtype=np.int))

    npcontexts = np.array(contexts, dtype=np.int)
    nputterances = np.array(utterances, dtype=np.int)
    npdistractors = np.array(all_distractors, dtype=np.int)
    print len(npcontexts), len(nputterances), len(npdistractors)
    return npcontexts, nputterances, npdistractors

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('include_missing', help="Include non glove words?", 
                        const=True, nargs="?", default=True, type=bool)
    args = parser.parse_args(arguments)
    include_missing = args.include_missing
    train, valid, test, embed_file = FILE_PATHS

    # get GloVe pretrained word embeddings
    print "Reading embeddings"
    # embeddings = {"hello": np.zeros(50)}
    embeddings = get_embeddings(embed_file)

    # get vocab from data, fills in word_to_idx
    print "Reading vocab"
    idx_to_embedding = get_vocab([train, valid, test], embeddings)
    V = len(word_to_idx)
    print("Embeddings: ", len(embeddings))
    print("Embedding len: ", len(embeddings["hello"]))
    print("Vocab size: ", V)

    # convert all data
    print "Converting data"
    train_contexts, train_utterances, train_targets = convert_train(train)
    val_contexts, val_utterances, val_all_distractors = convert_test(valid)
    test_contexts, test_utterances, test_all_distractors = convert_test(test)

    # saves data + idx_to_embedding to hdf5 file
    print "Saving data"
    filename = "data/data.hdf5"
    with h5py.File(filename, "w") as f:
        f["train_contexts"] = train_contexts
        f["train_utterances"] = train_utterances
        f["train_targets"] = train_targets
        f["val_contexts"] = val_contexts
        f["val_utterances"] = val_utterances
        f["val_all_distractors"] = val_all_distractors
        f["test_contexts"] = test_contexts
        f["test_utterances"] = test_utterances
        f["test_all_distractors"] = test_all_distractors
        f["idx_to_embedding"] = idx_to_embedding

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))



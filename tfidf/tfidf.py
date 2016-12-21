from MyTfidfVectorizer import MyTfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
val_df = pd.read_csv("../data/valid.csv")

# target values
y_target = np.zeros(len(test_df))

# recall@(k,10) metric
def evaluate_recall(y, y_target, k=1):
    total = float(len(y))
    correct = 0
    for predictions, label in zip(y, y_target):
        if label in predictions[:k]:
            correct += 1
    return correct/total

# fit
class TfidfPredictor:
    def __init__(self, own=True):
        if own:
            self.vectorizer = MyTfidfVectorizer()
        else:
            self.vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=True)

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predict(self, context, utterances):
        # convert context and utterances into tf-idf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)

        # measure similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        
        # sort by top results and return indices in descending order
        return np.argsort(result, axis=0)[::-1]

# evaluate
my_mod = TfidfPredictor(True)
my_mod.train(train_df)
my_y = [my_mod.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
print("Results using manual implementation of tf-idf vectorizer")
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(my_y, y_target, n)))

mod = TfidfPredictor(False)
mod.train(train_df)
y = [mod.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
print("Results using sklearn tf-idf vectorizer")
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_target, n)))




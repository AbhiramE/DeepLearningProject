import cPickle as p
import itertools
import json
from collections import Counter

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import data_explore as de

DATA_FILE = '../data/booksummaries.txt'
PICKLE_DUMP = '../data/dataset.p'
FORMATTED_DUMP = '../data/formatted_data.p'
LDA_DUMP = '../data/lda_dump.p'
LDA_MODEL = '../data/lda_model.p'



def lda_transform(data_frame, n_topics, lda_dump, lda_model_file):
    summaries = data_frame['summary'].tolist()

    tf_vectorizer = CountVectorizer(stop_words='english')
    vectors = tf_vectorizer.fit_transform(summaries)
    lda_model = LatentDirichletAllocation(n_topics=n_topics,
                                          learning_method='batch')
    transformed_docs = lda_model.fit_transform(vectors)

    print transformed_docs[0]
    p.dump(transformed_docs, open(lda_dump, 'wb'))
    p.dump(lda_model, open(lda_model_file, 'wb'))

    return transformed_docs

if __name__ == '__main__':
    
    df = de.read_data(DATA_FILE, use_dump=False, dump=PICKLE_DUMP)
    altered_dataframe = de.format_data(df, FORMATTED_DUMP)

    transformed_docs = lda_transform(altered_dataframe, 200, LDA_DUMP,LDA_MODEL)



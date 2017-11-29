import cPickle as p
import sys
import json
import os.path as o


import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.explore import data_explore as de

root = o.abspath(o.dirname(__file__))
DATA_FILE = o.join(root, '../../data/booksummaries.txt')
PICKLE_DUMP = o.join(root, '../../data/dataset.p')
FORMATTED_DUMP = o.join(root, '../../data/formatted_data.p')
LDA_DUMP = o.join(root, '../../data/lda_dump.p')
LDA_MODEL = o.join(root, '../../data/lda_model.p')


def lda_transform(data_frame, n_topics, lda_dump, lda_model_file):
    summaries = data_frame['summary'].tolist()

    tf_vectorizer = CountVectorizer(stop_words='english')
    vectors = tf_vectorizer.fit_transform(summaries)
    lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch')
    transformed_docs = lda_model.fit_transform(vectors)

    print transformed_docs[0]
    p.dump(transformed_docs, open(lda_dump, 'wb'))
    p.dump(lda_model, open(lda_model_file, 'wb'))

    return transformed_docs


def clean_summaries(data_frame):
    new_df = pd.DataFrame(data_frame['summary'].str.split(' ').str.len())
    return data_frame[(new_df['summary'] >= 50) & (new_df['summary'] <= 2500)]


def clean_genres(data_frame):
    alt_frame = data_frame.drop('summary', axis=1)
    data_frame = pd.DataFrame(data_frame['summary'])
    alt_frame = alt_frame[alt_frame.columns[(alt_frame.sum() > 100)]]
    data_frame = pd.concat([data_frame, alt_frame], axis=1)
    return data_frame


if __name__ == '__main__':
    df = de.read_data(DATA_FILE, use_dump=False, dump=PICKLE_DUMP)
    altered_dataframe = de.format_data(df, FORMATTED_DUMP)
    altered_dataframe = clean_summaries(altered_dataframe)
    altered_dataframe = clean_genres(altered_dataframe)
    p.dump(altered_dataframe, open(FORMATTED_DUMP, 'wb'))
    transformed_docs = lda_transform(altered_dataframe, 200, LDA_DUMP, LDA_MODEL)

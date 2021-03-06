import cPickle as p
import json
from collections import Counter
import os.path as o
root = o.abspath(o.dirname(__file__))
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

DATA_FILE = o.join(root, '../../data/booksummaries.txt')
PICKLE_DUMP = o.join(root, '../../data/dataset.p')
FORMATTED_DUMP = o.join(root, '../../data/formatted_data.p')
LDA_DUMP = o.join(root, '../../data/lda_dump.p')
LDA_MODEL = o.join(root, '../../data/lda_model.p')


def read_data(filename, use_dump, dump=None):
    # Whether or not to use pickle dump
    if not use_dump:
        all_data = pd.read_csv(filename, sep='\t', header=None)
        all_data.columns = ['id', 'some_id', 'title', 'author', 'rel_date', 'genres', 'summary']

        all_data = all_data[pd.notnull(all_data['genres'])]
        genres = pd.Series(all_data['genres']).tolist()
        # Get the first genre in the list of genres
        for i in range(len(genres)):
            genres[i] = [x.encode('utf-8') for x in json.loads(genres[i]).values()]
        all_data['genres'] = pd.Series(genres)
        p.dump(all_data, open(dump, 'wb'))
    else:
        all_data = p.load(open(dump, 'rb'))
    return all_data


def format_data(data_frame, dump):
    adf = data_frame.drop(['id', 'some_id',
                           'title', 'author', 'rel_date'], axis=1)
    adf = adf[pd.notnull(data_frame['genres'])]

    # One hot encoding the genres.
    mlb = MultiLabelBinarizer()
    adf = adf.join(pd.DataFrame(mlb.fit_transform(adf.pop('genres')),
                                columns=mlb.classes_,
                                index=adf.index))
    p.dump(adf, open(dump, 'wb'))
    return adf


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


def get_sequence_lengths(data_frame):
    all_summaries = data_frame['summary']
    list_of_all_summaries = all_summaries.tolist()

    len_list = []

    for summary in list_of_all_summaries:
        len_list.append(len(summary.split(" ")))

    return len_list


def plot_sequence_lengths(data_frame):
    seq_lengths = get_sequence_lengths(data_frame)
    sns.distplot(seq_lengths, hist=True, norm_hist=False, kde=False, axlabel='Lengths')
    plt.show()


def plot_genres(data_frame):
    pd.Series(data_frame.drop('summary', axis=1).sum(axis=0)).sort_values(axis=0, ascending=False). \
        plot(kind='area').locator_params(axis='x', nticks=3)
    plt.show()


if __name__ == '__main__':
    df = read_data(DATA_FILE, use_dump=False, dump=PICKLE_DUMP)
    altered_dataframe = format_data(df, FORMATTED_DUMP)
    plot_genres(altered_dataframe)
    plot_sequence_lengths(altered_dataframe)

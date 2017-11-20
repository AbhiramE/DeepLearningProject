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

DATA_FILE = '../data/booksummaries.txt'
PICKLE_DUMP = '../data/dataset.p'
FORMATTED_DUMP = '../data/formatted_data.p'
LDA_DUMP = '../data/lda_dump.p'
LDA_MODEL = '../data/lda_model.p'


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
    ax = sns.distplot(seq_lengths, hist=True,norm_hist=False,kde=False,axlabel = 'Lengths')
    #ax = sns.rugplot(seq_lengths)
    plt.show()

def plot_genres(data_frame):
    pd.Series(altered_dataframe.drop('summary', axis=1).sum(axis=0)).sort_values(axis=0, ascending=False). \
        plot(kind='area').locator_params(axis='x', nticks=3)
    plt.show()

if __name__ == '__main__':
    df = read_data(DATA_FILE, use_dump=False, dump=PICKLE_DUMP)
    altered_dataframe = format_data(df, FORMATTED_DUMP)
    #pd.Series(altered_dataframe.drop('summary', axis=1).sum(axis=0)).sort_values(axis=0, ascending=False). \
     #   plot(kind='line').locator_params(axis='x', nticks=3)
    #plt.show()

    #summaries = altered_dataframe['summary'].tolist()
    #altered_dataframe.info(null_counts=True, verbose=True)
    #transformed_docs = lda_transform(altered_dataframe, 200, LDA_DUMP,
                                     #LDA_MODEL)
    #seq_lengths = get_sequence_lengths(altered_dataframe)
    #print len(seq_lengths)
    #trim_seq_lengths = [x for x in seq_lengths if x>50 and x<1500]
    #print len(trim_seq_lengths)
    #print sum(seq_lengths)/len(seq_lengths)
    #print max(seq_lengths)
    #print min(seq_lengths)
    #print sorted(seq_lengths)[len(seq_lengths)/2]
    #x = [x for x in seq_lengths if x<2500]
    #print len(x)
    
    #plot_sequence_lengths(altered_dataframe)
    plot_genres(altered_dataframe)
    
    '''
    n, bins, patches = plt.hist(seq_lengths, 50, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma))
    plt.xlabel('Frequency')
    plt.ylabel('KLengths')
    plt.title('Histogram of lengths')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.plot()
    plt.show()
    
    #plt.hist(seq_lengths).plot()
    #plt.show()
    '''

    #print sum(seq_lengths)/float(len(seq_lengths))

# df.info(null_counts=True,verbose=True)

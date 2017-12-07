# import sys
# sys.paths.append('/mnt/nfs/scratch1/ssubraveti')

from nltk.tokenize import sent_tokenize
import skipthoughts
import data_explore as de
import cPickle as p
import pandas as pd
import numpy as np

AVERAGED_VECTORS_DUMP = 'averaged_vectors.p'
MAXED_OUT_DUMP = 'maxed_out_vectors.p'
DATA_FILE = 'booksummaries.txt'
PICKLE_DUMP = 'dataset.p'
FORMATTED_DUMP = 'formatted_data.p'

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)


def run_encoder(summaries):
    return encoder.encode(summaries)


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

    # f = open('mnt/nfs/scratch1/ssubraveti/x.txt','rb')
    print "Able to import"

    df = de.read_data(DATA_FILE, use_dump=False, dump=PICKLE_DUMP)
    altered_dataframe = de.format_data(df, FORMATTED_DUMP)
    altered_dataframe = clean_summaries(altered_dataframe)
    altered_dataframe = clean_genres(altered_dataframe)['summary']
    print len(altered_dataframe)
    maxed_out_vectors = np.empty((0, 4800))
    averaged_vectors = np.empty((0, 4800))

    for index, row in altered_dataframe.iteritems():
        summary_vectors = run_encoder(sent_tokenize(row.decode('utf-8').encode('ascii', 'ignore')))
        sampleInd = np.random.choice(summary_vectors.shape[0], size=(min(10, len(summary_vectors)),))
        sample = summary_vectors[sampleInd]
        averaged_vectors = np.append(averaged_vectors, np.mean(sample, axis=0).reshape(1, 4800), axis=0)
        maxed_out_vectors = np.append(maxed_out_vectors, np.max(sample, axis=0).reshape(1, 4800), axis=0)
    p.dump(averaged_vectors, open(AVERAGED_VECTORS_DUMP, 'wb'))
    p.dump(maxed_out_vectors, open(MAXED_OUT_DUMP, 'wb'))

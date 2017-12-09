from nltk.tokenize import sent_tokenize
import skipthoughts
import data_explore as de
import cPickle as p
import pandas as pd
import numpy as np

VECTORS_DUMP = 'skip_thought_vectors.p'
DATA_FILE = 'booksummaries.txt'
PICKLE_DUMP = 'dataset.p'
FORMATTED_DUMP = 'formatted_data.p'
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)


def run_encoder(summaries):
    return np.mean(encoder.encode(summaries), axis=0).reshape(1, 4800)


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

    vectors = np.empty((0, 4800))

    for index, row in df.iterrows():
        vectors = np.append(vectors,
                            run_encoder(sent_tokenize(row['summary'].decode('utf-8').encode('ascii', 'ignore'))),
                            axis=0)
    p.dump(vectors, open(VECTORS_DUMP, 'wb'))

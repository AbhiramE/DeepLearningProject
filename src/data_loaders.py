import json
import numpy as np
import pandas as pd
import cPickle as p
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter

DATA_FILE = '../data/booksummaries.txt'
PICKLE_DUMP = '../data/dataset.p'

def read_data(filename,use_dump):

    # Whether or not to use pickle dump
    if(not use_dump):
        all_data = pd.read_csv(filename, sep = '\t',header=None)
        all_data.columns = ['id','some_id','title','author','rel_date','genres','summary']
        
        all_data = all_data[pd.notnull(all_data['genres'])]
        genres = pd.Series(all_data['genres']).tolist()
        # Get the first genre in the list of genres
        for i in range(len(genres)):
            genres[i] = json.loads(genres[i]).values()[0].encode('utf-8')
        all_data['genres'] = pd.Series(genres)
        p.dump(all_data,open(PICKLE_DUMP,'wb'))
    else:
        all_data = p.load(open(PICKLE_DUMP,'rb'))
    return all_data

def format_data(data_frame):
    
    adf = data_frame.drop(['id','some_id',
                        'title','author','rel_date'],axis=1)

    
    adf = adf[pd.notnull(df['genres'])]

    genre_list = adf['genres'].tolist()
    print len(genre_list)
    
    # For encoding genres deterministically. Most frequent gets 0, least
    # frequent gets maximum. Ties broken lexicographically.
    counter = Counter(genre_list)
    counts = sorted(counter.most_common(), key=lambda x:(-x[1], x[0]))
    encoding = {}
    for value, key in enumerate(counts):
        encoding[key[0]] = value
    encoded_genres = [encoding[genre] for genre in genre_list]
    adf['genres'] = pd.Series(encoded_genres,index=adf.index)
    return adf


if __name__ == '__main__':

    df = read_data(DATA_FILE,use_dump=False)
    altered_dataframe = format_data(df)
    #print altered_dataframe
    #altered_df = format_data(df)
   
    print "Main"
    altered_dataframe.info(null_counts=True, verbose=True)
    
    
#    df.info(null_counts=True,verbose=True)

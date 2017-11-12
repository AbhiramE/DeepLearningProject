import json
import numpy as np
import pandas as pd
import cPickle as p
import os
import matplotlib.pyplot as plt
DATA_FILE = '../data/booksummaries.txt'
PICKLE_DUMP = '../data/dataset.p'

def read_data(filename,use_dump):

    if(not use_dump):
        all_data = pd.read_csv(filename, sep = '\t',header=None)
        all_data.columns = ['id','some_id','title','author','rel_date','genres','summary']
        all_data = all_data[pd.notnull(all_data['genres'])]
        genres = pd.Series(all_data['genres']).tolist()
        for i in range(len(genres)):
            genres[i] = json.loads(genres[i]).values()[0].encode('utf-8')
        all_data['genres'] = pd.Series(genres)
        p.dump(all_data,open(PICKLE_DUMP,'wb'))
    else:
        all_data = p.load(open(PICKLE_DUMP,'rb'))
    return all_data

if __name__ == '__main__':

    df = read_data(DATA_FILE,use_dump=False)

    x = df['genres'].value_counts().plot()
    plt.show()
    
    
#    df.info(null_counts=True,verbose=True)

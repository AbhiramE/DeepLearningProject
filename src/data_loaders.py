import json
import numpy as np
import pandas as pd


def get_data_and_labels(filepath):
    data = np.genfromtxt(filepath, delimiter='\t', dtype=str)
    j = 0

    # Make the first genre as the label
    for i in range(len(data)):
        if len(data[i, 5]) == 0:
            j += 1
        else:
            data[i, 5] = np.array(json.loads(data[i, 5]).values())[0].encode('utf-8')

    # Remove all missing values
    df = pd.DataFrame(data)
    df = df[df[5].str.len() > 0]
    return df[6], df[5]


file = "dataset_temp.txt"
X, y = get_data_and_labels(filepath=file)
print X, y

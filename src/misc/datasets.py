import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

"""This function creates a smaller dataset from the real dataset. This can be useful for training and testing initial
models"""


def create_smaller_dataset():
    NUM_OF_LINES = 2000
    filename = '../booksummaries.txt'

    with open(filename) as fin:
        fout = open("dataset_temp.txt", "wb")
        for i, line in enumerate(fin):
            fout.write(line)
            if (i + 1) % NUM_OF_LINES == 0:
                fout.close()
                break

        fout.close()

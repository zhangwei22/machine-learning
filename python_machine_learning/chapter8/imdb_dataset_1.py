import os
import pyprind
import pandas as pd

if __name__ == '__main__':
    pbar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for s in ('tree', 'train'):
        for l in ('pos', 'neg'):
            path = './aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']

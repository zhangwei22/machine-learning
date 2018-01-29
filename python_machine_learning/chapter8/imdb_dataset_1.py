import os
import pyprind
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def initDocs():
    pbar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = '/Users/zhangwei/Desktop/aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('./movie_data.csv', index=False)
    # df = pd.read_csv('./movie_data.csv')
    # print(df.head(3))


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()).join(emoticons).replace('-', '')
    return text


if __name__ == '__main__':
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet'
    ])
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    print(bag.toarray())

    tfidf = TfidfTransformer()
    np.set_printoptions(precision=2)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

    df = pd.read_csv('./movie_data.csv')
    preprocessor(df.loc[0, 'review'][-50:])
    preprocessor("</a>This :) is :( a test :-)!")


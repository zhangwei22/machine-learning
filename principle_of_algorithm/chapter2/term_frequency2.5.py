import sys
import os
from sklearn.datasets.base import Bunch
import pickle

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()


if __name__ == '__main__':
    # 导入分词后的词向量Bunch对象
    path = "train_word_bag/train_set.dat"
    bunch = readbunchobj(path)
    # 构建TF-IDF词向量空间对象
    tfidfspace = Bunch(target_name=bunch.tar_get_name, label=bunch.label, filename=bunch.filename, tdm=[],
                       vocabulary={})
    # 使用TfidfVectorizer初始化向量空间模型
    vectorizer = TfidfVectorizer(stop_words="stpwrdlst", sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()
    # 文本转为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    # 创建词袋的持久化
    space_path = "train_word_bag/tfdifspace.dat"
    writebunchobj(space_path, tfidfspace)

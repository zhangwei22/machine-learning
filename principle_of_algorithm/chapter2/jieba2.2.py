import jieba
from sklearn.datasets.base import Bunch


def splitword():
    seg_list = jieba.cut("小明1995年毕业于北京清华大学", cut_all=False)
    # 默认切分
    print("Default Mode:", " ".join(seg_list))

    seg_list = jieba.cut("小明1995年毕业于北京清华大学")
    print("seg_list:", " ".join(seg_list))

    seg_list = jieba.cut("小明1995年毕业于北京清华大学", cut_all=True)
    # 全切分
    print("Full Mode:", "/ ".join(seg_list))
    # 搜索引擎模式
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    print("/ ".join(seg_list))


def savefile(savepath, content):
    fp = open(savepath, "wb")
    fp.write(content)
    fp.close()


def readfile(path):
    fp = open(path, "rb")
    content = fp.read()
    fp.close()
    return content

def test_seg():
    filepath = "test_corpus/computer/164.txt"
    content = readfile(filepath).strip()
    content = content.replace("\r\n", "").strip()
    content_seg = jieba.cut(content)
    savefile("train_corpus_seg/computer/164.txt", bytes(" ".join(content_seg), encoding="utf-8"))


if __name__ == '__main__':
    #splitword()
    test_seg()




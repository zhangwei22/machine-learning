import pandas as pd
import csv

'''
去除数据中的null项
数据清洗加工
'''


def del_nil_operate(df):
    '''
    :param df:
    :return:
    去掉所有有空的样本数据
    '''

    with open("/Users/zhangwei/Desktop/train_data_3.csv", "a", encoding="utf-8") as csvfile:
        for index, row in df.iterrows():
            if str(row[0]) != "\\N" and str(row[1]) != "\\N" and str(row[2]) != "\\N" and str(row[3]) != "\\N" and str(
                    row[4]) != "\\N":
                print(row.values)
                crt_line = csv.writer(csvfile, dialect="excel")
                crt_line.writerow(row.values)


def rep_nil_with_mind(df):
    '''
    :param df:
    :return:
     用中位数填充缺失项
    '''
    # 对所选项用中位数去填充
    df["onlinePayTermCount"] = df["onlinePayTermCount"].fillna(df["onlinePayTermCount"].median())
    df["offlineLoanCount"] = df["offlineLoanCount"].fillna(df["offlineLoanCount"].median())
    df["phoneConcatCount"] = df["phoneConcatCount"].fillna(df["phoneConcatCount"].median())
    df["zhimaScore"] = df["zhimaScore"].fillna(df["zhimaScore"].median())
    with open("/Users/zhangwei/Desktop/train_data_4.csv", "a", encoding="utf-8") as csvfile:
        for index, row in df.iterrows():
            crt_line = csv.writer(csvfile, dialect="excel")
            crt_line.writerow(row.values)


def main():
    df = pd.read_csv("/Users/zhangwei/Desktop/train_data_2.csv")
    rep_nil_with_mind(df)


if __name__ == '__main__':
    main()

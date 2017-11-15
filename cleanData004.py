import pandas as pd
import csv

'''
去除数据中的null项
数据清洗加工
'''


def operate(df):
    with open("/Users/zhangwei/Desktop/train_data_3.csv", "a", encoding="utf-8") as csvfile:
        for index, row in df.iterrows():
            if str(row[0]) != "\\N" and str(row[1]) != "\\N" and str(row[2]) != "\\N" and str(row[3]) != "\\N" and str(
                    row[4]) != "\\N":
                print(row.values)
                crt_line = csv.writer(csvfile, dialect="excel")
                crt_line.writerow(row.values)


def main():
    df = pd.read_csv("/Users/zhangwei/Desktop/train_data_2.csv")
    operate(df)


if __name__ == '__main__':
    main()

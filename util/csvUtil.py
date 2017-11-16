import pandas as pd
import csv

'''
将文件query_result.csv中的数据处理后写到文件new_data.csv中
注意：数据写到新文件后表头没有了，有需要自行添加
'''


def change_y_values(data):
    '''
    :param data:
    :return:
    y值转换:0,未逾期;1,逾期
    '''
    with open("/Users/zhangwei/Desktop/new_data.csv", "a") as csvfile:
        for i, row in data.iterrows():
            if int(row[4]) != 0:
                row[4] = 1
            crt_line = csv.writer(csvfile, dialect="excel")
            crt_line.writerow(row.values)


def sort_y_list(data):
    '''
    :param data:
    :return:
    y值分类排列
    '''
    with open("/Users/zhangwei/Desktop/sort_new_data_2.csv", "a") as csvfile:
        for i, row in data.iterrows():
            if str(row[4]) == '1.0P':
                row[4] = 'overdue'
                print(row.values)
            else:
                row[4] = 'not-overdue'
                print(row.values)
            crt_line = csv.writer(csvfile, dialect="excel")
            crt_line.writerow(row.values)


def main():
    origin_data = pd.read_csv("/Users/zhangwei/Desktop/sort_new_data.csv")
    sort_y_list(origin_data)


if __name__ == '__main__':
    main()

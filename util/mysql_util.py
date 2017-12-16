import pymysql
import traceback

'''
数据库操作
'''


def get_connect():
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "root", "spider")
    # 设置编码
    db.set_charset('utf8')
    return db


def get_cursor(db):
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')
    return cursor


def add(data):
    '''
    数据插入操作
    :return:
    '''
    sql = "insert into film(`name`, english_name, `type`, country, duration, release_time, score, synopsis, image_url, director, actor, `language`) VALUES('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (
        data.get("name"), data.get("english_name"), data.get("type"), data.get("country"), data.get("duration"),
        data.get("release_time"),
        data.get("score"), data.get("synopsis"), data.get("image_url"), data.get("director"), data.get("actor"),
        data.get("language"))
    print("insert sql:", sql)
    db = get_connect()
    cursor = get_cursor(db)
    try:
        # 使用 execute() 方法执行 SQL
        status = cursor.execute(sql)
        print("insert status:", status)
        # 提交事务
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()
        msg = traceback.format_exc()
        print(msg)
    db.close()


def delete(id):
    '''
    数据删除操作
    :param id:
    :return:
    '''
    sql = "delete from `film` where `id` = '%s'" % (id)
    print("deleta sql:", sql)
    db = get_connect()
    cursor = get_cursor(db)
    try:
        status = cursor.execute(sql)
        print("delete status:", status)
        db.commit()
    except:
        db.rollback()
        msg = traceback.format_exc()
        print(msg)
    db.close()


def update(data):
    '''
    数据修改操作
    :param data:
    :return:
    '''
    sql = "update `film` set `name` = '%s', english_name = '%s' where `id`='%s'" % (
        data.get("name"), data.get("english_name"), data.get("id"))
    print("update sql:", sql)
    db = get_connect()
    cursor = get_cursor(db)
    try:
        status = cursor.execute(sql)
        print("delete status:", status)
        db.commit()
    except:
        db.rollback()
        msg = traceback.format_exc()
        print(msg)
    db.close()


def select(id):
    '''
    数据查询操作
    :param id:
    :return:
    '''
    sql = "select * from `film` where id = '%d'" % (id)
    print("select sql:", sql)
    db = get_connect()
    cursor = get_cursor(db)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print("select result:", results)
        # 提交事务
        db.commit()
    except:
        db.rollback()
        return None
    db.close()
    return results


def create_table():
    '''
    创建表操作
    :return:
    '''
    db = get_connect()
    cursor = get_cursor(db)
    cursor.execute("drop table if EXISTS employee ")
    sql = """CREATE TABLE employee (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""
    cursor.execute(sql)
    db.close()


if __name__ == '__main__':
    data = {'id': '536',
            'name': '寻梦环游记',
            'english_name': 'Coco',
            'score': '9.5',
            'image_url': 'https://img.alicdn.com/bao/uploaded/i2/TB1xsowaG_85uJjSZFlXXXemXXa_.jpg_300x300.jpg',
            'director': '李·昂克里奇',
            'actor': '安东尼·冈萨雷兹,本杰明·布拉特,盖尔·加西亚·贝纳尔,芮妮·维克托',
            'type': '动画,冒险,动作',
            'country': '美国',
            'language': '英语',
            'duration': '105分钟',
            'synopsis': '小男孩米格(安东尼·冈萨雷斯 配音)一心梦想成为音乐家，更希望自己能和偶像歌神德拉库斯(本杰明·布拉特 配音)一样，创造出打动人心的音乐，但他的家族却世代禁止族人接触音乐。米格痴迷音乐，无比渴望证明自己的音乐才能，却因为一系列怪事，来到了五彩斑斓又光怪陆离的神秘世界。在那里，米格遇见了魅力十足的落魄乐手埃克托(盖尔·加西亚·贝纳尔 配音)，他们一起踏上了探寻米格家族不为人知往事的奇妙之旅，并开启了一段震撼心灵、感动非凡、永生难忘的旅程。',
            'release_time': '上映时间：2017-11-24'}

    # add(data)
    # delete(538)
    # update(data)
    # set = select(536)
    # print("set:", set)
    create_table()

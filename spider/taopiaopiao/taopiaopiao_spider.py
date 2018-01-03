import requests
import sys
import os
from urllib.parse import urlencode
from requests.exceptions import RequestException
from bs4 import BeautifulSoup

from util.mysql_util import *


def get_page_link():
    '''
    获取首页全部电影的url
    '''
    allLink = []
    data = {
        'spm': 'a1z21.3046609.header.4.Th0aeu',
        'n_s': 'new'
    }
    url = 'https://dianying.taobao.com/showList.htm?' + urlencode(data)
    # print("url:",url)
    try:
        res = requests.get(url)
        if res.status_code == 200:
            htm = res.text
            soup = BeautifulSoup(htm, 'lxml')
            list = soup.select('.movie-card-wrap')
            listCount = len(list)
            for i in range(listCount):
                url = soup.select('.movie-card-wrap')[i].select('.movie-card')[0].attrs['href']
                print(url)
                allLink.append(url)
            return allLink
        else:
            return None
    except RequestException:
        print("获取页面链接异常")
        return None


def get_page_content(url):
    '''
    解析电影详情页面
    :param url:
    :return:
    '''
    film = {}
    try:
        res = requests.get(url)
        if res.status_code == 200:
            htm = res.text
            soup = BeautifulSoup(htm, 'lxml')
            content = soup.select(".detail-cont")[0]

            title = content.select(".cont-title")[0].text
            titleStr = str(title)
            titleArr = titleStr.split(" ")
            print(titleArr)
            film["name"] = titleArr[0]

            if len(titleArr) >= 3 and ("）" not in titleArr[-1]):
                englishName = ''
                for i in range(len(titleArr) - 2):
                    englishName += " "
                    englishName += titleArr[i + 1]
                englishName = englishName[1:]

                film["english_name"] = englishName
                film["score"] = titleArr[-1]

            film["image_url"] = content.select(".cont-pic")[0].select("img")[0].attrs["src"]

            infoList = content.select(".cont-info")[0].select("li")
            print("infoList:", infoList[0].text)
            for i in range(len(infoList)):
                crtLi = infoList[i].text
                if crtLi.startswith("导演："):
                    film["director"] = crtLi.split("：")[1]
                elif crtLi.startswith("主演："):
                    film["actor"] = crtLi.split("：")[1]
                elif crtLi.startswith("类型："):
                    film["type"] = crtLi.split("：")[1]
                elif crtLi.startswith("制片国家/地区："):
                    film["country"] = crtLi.split("：")[1]
                elif crtLi.startswith("语言："):
                    film["language"] = crtLi.split("：")[1]
                elif crtLi.startswith("片长："):
                    film["duration"] = crtLi.split("：")[1]
                elif crtLi.startswith("剧情介绍："):
                    film["synopsis"] = crtLi.strip().split("：")[1]
                else:
                    continue

            film["release_time"] = content.select(".cont-time")[0].text
            print(film)
            return film
        return None
    except RequestException:
        return None


def get_one_page(url):
    '''
    获取指定url的内容
    '''
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.text
        return None
    except RequestException:
        return None


def write_content(content):
    '''
    写入指定的内容到文件中
    '''
    f = open("persistence/stock_rnn-tmp.txt", "w")
    f.write(content)
    f.close()


if __name__ == '__main__':
    print(123)
    # allLink = get_page_link()
    # for i in range(len(allLink)):
    #     print("crt index:", str(i), ", crt link:", allLink[i])
    #     beanData = get_page_content(allLink[i])
    #     print(beanData)
    #     if beanData != None:
    #         add(beanData)

    # aa = ''
    # data = get_page_content("https://dianying.taobao.com/showDetail.htm?showId=219255&n_s=new&source=current")
    # print("name:", data.get("rr"))
    # aa = data.get("rr")
    # print(aa == None)

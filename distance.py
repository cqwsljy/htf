# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:52:11 2017

@author: Alpha
"""

from numpy import sin
from numpy import cos
from numpy import arccos
import urllib.request
import json
from math import pi
import numpy as np

def getLatLon(qAdress,key,citySearch=None):
    if not citySearch:# citySearch is none
        citySearch = " "
    gdapi = 'http://restapi.amap.com/v3/geocode/geo?key='
    url = gdapi+key+'&address='+ urllib.request.quote(qAdress) +'&'+'city='+urllib.request.quote(citySearch)
    flag = False
    try:
        response = urllib.request.urlopen(url)
        html = response.read()
        if not isinstance(html,str):
            html = html.decode("utf8")
        adinfo = json.loads(html)
        flag = adinfo['info'] == "DAILY_QUERY_OVER_LIMIT"
        flag = flag + 1
    except Exception as e:
        print(e)
        return [None,None,flag]
    if adinfo["status"] == "1":
        try: #可能出现异常错误，目前没有检查出来是怎么回事
            lon,lat = adinfo['geocodes'][0]["location"].split(",")
            lon = np.float(lon)
            lat = np.float(lat)
            return [lon,lat,flag]
        except:
            return [None,None,flag]
    else:
        return [None,None,flag]
    
def getDist(lonA,latA,lonB,latB):
    '''
    return dist ,the distance(km) between A and B,angle is radian measure
    reference:http://blog.sina.com.cn/s/blog_45eaa01a0102w6ai.html    
    you can check the answer by website http://www.storyday.com/wp-content/uploads/2008/09/latlung_dis.html
    '''
    R = 6371.004 
    MlatA = (latA)*pi/180
    MlatB = (latB)*pi/180
    MlonA = lonA*pi/180
    MlonB = lonB*pi/180
    C = sin(MlatA) * sin(MlatB) + cos(MlonA - MlonB) * cos(MlatA) * cos(MlatB)
    dist = R * arccos(C)
    return dist

with open("gaode.txt") as f:
    keys = f.readlines()
    keys = [i.strip("\n") for i in keys]

if __name__ == "__main__":
    '''
    '''
    key = keys[0]
    qAdressA = "上海市浦东新区来安路500号"
    qAdressB = "上海市闵行区东川路800号"
    lonA,latA,flag = getLatLon(qAdressA,key)
    lonB,latB,flag = getLatLon(qAdressA,key)
    dist = getDist(lonA,latA,lonB,latB)
    print("The distance is %f km" %(dist))
    
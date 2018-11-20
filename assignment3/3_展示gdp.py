# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:55:42 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:58 2018

@author: Administrator
"""

from skimage import transform,data
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.collections import PolyCollection
#显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi'] #指定默认字体，FangSong等等
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

#读取数据，此数据集包含国家名称、经纬度、1992-2011年GDP
GDP=pd.read_csv("./data/GDP_has_lonlat.csv")
Country=[]
Continent=[]
ID=[]
lat=[]
lon=[]
gdp2011=[]

for i in range(161):
    Country.append(GDP.iat[i,0])
    Continent.append(GDP.iat[i,1])
    ID.append(GDP.iat[i,2])
    lat.append(int(GDP.iat[i,3]))
    lon.append(int(GDP.iat[i,4]))
    gdp2011.append(int(GDP.iat[i,5]))

     
fig = plt.figure(figsize=(23.4,16.8))
m = Basemap(projection='mbtfpq',lon_0=0,resolution='c')
m.drawcoastlines(linewidth=0.1)
m.drawcountries(linewidth=0.1)
m.drawmapboundary(fill_color='lightcyan')  
m.fillcontinents(color='coral',zorder=0)
m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1],color='blue')
m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],color='blue')  


polys = []
for polygon in m.landpolygons:
    polys.append(polygon.get_coords())

lc = PolyCollection(polys[0:1],edgecolor='salmon',facecolor='salmon', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[1:2],edgecolor='pink',facecolor='pink', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[2:3], edgecolor='blue',facecolor='blue', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[3:4], edgecolor='salmon',facecolor='salmon', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[4:5],edgecolor='bisque', facecolor='bisque', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[5:6],edgecolor='palevioletred', facecolor='palevioletred', closed=False)
plt.gca().add_collection(lc)


lc = PolyCollection(polys[6:7],edgecolor='green', facecolor='green', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[7:8], edgecolor='gold', facecolor='gold', closed=False)
plt.gca().add_collection(lc)

lc = PolyCollection(polys[8:3000], edgecolor='bisque', facecolor='bisque', closed=False)
plt.gca().add_collection(lc)

x,y=m(lon,lat)
me=max(gdp2011)


y_offset=15
rotation=-30
size_factor = 400.0

for i,j,g,c,e in zip(x,y,Continent,ID,gdp2011):
    if g=="[Africa]":
        size = size_factor*e/me
        cs=m.scatter(i,j,s=size,marker='s',color='red') 
        plt.text(i,j+y_offset,c,rotation=-rotation,fontsize=13)
    if g=="[Asia]":
        size = size_factor*e/me
        cs=m.scatter(i,j,s=size,marker='*',color='green') 
        plt.text(i,j+y_offset,c,rotation=-rotation,fontsize=13)
    if g=="[Europe]":
        size = size_factor*e/me
        cs=m.scatter(i,j,s=size,marker='^',color='blue') 
        plt.text(i,j+y_offset,c,rotation=-rotation,fontsize=13)
    if g=="[America]":
        size = size_factor*e/me
        cs=m.scatter(i,j,s=size,marker='o',color='yellow') 
        plt.text(i,j+y_offset,c,rotation=-rotation,fontsize=13)

plt.title("世界各国2011年GDP")
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:05:13 2025

@author: HP
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv("C:\\Users\HP\Downloads\Audi_A1_listings.csv")
print(df.head(3))
df=df.drop(columns = ['index','href','MileageRank','PriceRank','PPYRank','Score'])
print(df.head(3))
df.columns = ["yil","kasa","mil","motor","ps","vites","yakit","sahip","fiyat","ppy"]
df.head(3)
df["motor"]=df["motor"].str.replace("L","") #motor kolonundaki L harfini boslukla değiştirdi
df["motor"]= pd.to_numeric(df["motor"])#nümerik değere çevirdi
df = pd.get_dummies(df.columns["kasa","vites","yakit"],drop_first = True)# bu stunları sayıya çevir
y = df[['fiyat']]
x = df.drop("fiyat",axis = 1)

lm = LinearRegression()
model = lm.fit(x,y)
test_data = np.array([[2018,44000,1.6,114,1,2500,1,0]])
print(model.predict(test_data))
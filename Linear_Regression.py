# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:40:59 2025

@author: HP
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
#import numpy as np
df = pd.read_csv("C:\\Users\HP\Downloads\Student_Marks.csv")
print(df.head(3))
y = df[["Marks"]]
x = df[["number_courses","time_study"]]
lm = LinearRegression()
model = lm.fit(x,y)#fit() öğren demek 
print(model.coef_) #katsayı
print(model.intercept_)#sabit
print(model.predict([[4,5]]))
print(4*1.86405074+(5*5.39917879)+(-7.45634623))
print(model.predict([[3,4.508]]))#22.47530397 alması gerekiyordu ama 19.202 almış

#print(model.score(x, y)) 
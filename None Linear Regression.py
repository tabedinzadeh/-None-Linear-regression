
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 
import csv

df = pd.read_csv('ghana.gpd.csv')
print(df.describe())

plt.figure(figsize=(8,5))
x_data, y_data = (df['year'].values,df['GPD'].values)
plt.plot(x_data, y_data , color='blue')
plt.xlabel('year')
plt.ylabel('GPD')
plt.show()


def sigmoid(x, b1, b2):
    y = 1 / (1 + np.exp(-b1*(x-b2)))
    return y

b1 = 0.1
b2 = 1990.0
y_data1 = y_data/5.3

y_pred = sigmoid(x_data, b1, b2)
plt.plot(x_data, y_pred*15000000000.)
plt.plot(x_data, y_data1, 'r')
plt.show()

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print('beta1 = %f, beta2 = %f' % (popt[0], popt[1]))

x = np.linspace(1960, 2023, 63)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata , 'r', label='data')
plt.plot(x, y,  label='fit')
plt.legend(loc='best')
plt.xlabel('year')
plt.ylabel('GPD')
plt.show()
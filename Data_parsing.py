import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Deep import Datasets


ratios = pd.read_csv('/Users/usuario/Desktop/Pycharm/Deep_Networks/New/Deep/ratios.csv', sep=',',index_col=0)
stock = pd.read_csv('/Users/usuario/Desktop/Pycharm/Deep_Networks/New/Deep/stocks.csv', sep=',', index_col=0)


df = Datasets.ratios_stocks()

corr = df.corr()

print(corr.columns)
print(corr.currentRatio[corr.currentRatio == '1'].index.tolist())

high_corr = []
low_corr = []
for i in range(0, 55):
    for j in range(0, 55):
        if .8<=corr.iloc[i][j]<=1:
            high_corr.append(corr.iloc[i][j])
        else:
            low_corr.append(corr.iloc[i][j])
#print("High Correlation: {}".format(high_corr))
#print(low_corr)
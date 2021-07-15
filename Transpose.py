import numpy as np
import pandas as pd

ratios = pd.read_csv('/Users/usuario/Desktop/Pycharm/Deep_Networks/New/Deep/ratios.csv', sep=',', index_col=0)

print(ratios.T)
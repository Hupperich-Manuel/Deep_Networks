import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def ratios_stocks():

    ratios = pd.read_csv('/Users/usuario/Desktop/Pycharm/Deep_Networks/New/Deep/ratios.csv', sep=',',index_col=0)
    stock = pd.read_csv('/Users/usuario/Desktop/Pycharm/Deep_Networks/New/Deep/stocks.csv', sep=',', index_col=0)

    def data(ratios, stock):
        # Ratios
        ratio = ratios.T
        ratio.drop('period', 1, inplace=True)
        ratio.index = pd.to_datetime(ratio.index).to_period('M')

        # Stock
        stock.index = pd.to_datetime(stock.index).to_period('M')
        stock = stock.sort_index(ascending=False)  # ['close']
        stock = stock['2021-03':'1989-12']

        # Merge
        df = ratio.merge(stock, how='outer', left_index=True, right_index=True)
        # df.dropna(0, inplace=True)
        return df


    df = data(ratios, stock)

    def organized(df):
        '''Organize the data which goes from implementing stock returns, through
        np.nan elimination, until the Normalization of the
        data in order to implement in a much more efficient way the Networks'''

        df.drop(['low', 'adjclose', 'open', 'high', 'volume'], 1, inplace=True)
        df = df.astype(str)
        df.replace('None', 0, inplace=True)
        df = df.astype(float)
        df.close = np.log(df.close/df.close.shift(1))
        df.drop(df.index[0], 0, inplace=True)
        #df.dropna(0, inplace=True)
        df.close = df.close.apply(lambda x: 1. if x >= 0 else 0.)
        # Eliminating NaN data
        for j in df.columns:
            for i in range(df.shape[0]):
                if np.isnan(df[j][i]) == True:
                    df[j] = df[j].replace(df[j][i], (df[j][i - 1] + df[j][i + 1])/2)
                else:
                    continue

        return df

    df = organized(df)

    def Normalization(df, plot=False, heat=False):
        columns = df.columns[:-1]
        # Normalization mean and variance
        for i in columns:
            df[i] = df[i] - df[i].mean(axis=0)
            df[i] = df[i]/df[i].std(axis=0)

            # Scatter plots
        if plot == True:
            plt.figure(figsize=(16, 32))
            for i in range(0, X.shape[1]):
                plt.subplot(20, 3, i + 1)
                plt.scatter(y, X[:, i])
                plt.title(df.columns[i] + '(x) with Apple stocks(y)')
                plt.tight_layout(pad=2)

        # Heatmaps of Correlations
        if heat == True:
            plt.figure(figsize=(16, 5))
            ax = sns.heatmap(df.corr())


        return df


    df = Normalization(df)
    return df

#df = ratios_stocks()
#train_X, train_Y, test_X, test_Y = np.array(df[df.columns[0:-1]][:125].T), np.array(df[df.columns[-1]][:125]), np.array(df[df.columns[0:-1]][125:].T), np.array(df[df.columns[-1]][125:])
#print(train_X.shape)
#print(train_Y.shape)
#print(test_X.shape)
#print(test_Y.shape)
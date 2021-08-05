# Deep_Networks
***This is only a small part that comprises a leveraged analysis of stocks, where you only have to put the ticker you are interested in, and the dates, and the programm returns the result of an adjusted Deep Nerual Network for Prediction. Moreover, I am currently working on a Java application which aimes to make this all more easy to visualize, specially for those who are not that familiar with coding.***
Unadjusted Deep Neural Network for Apple Stocks
In order to properly explain my analysis, I will go through each of the files independently and then, join them together and see how they work harmoniously together for a test accuracy.
First of all, we need to get the input layers. It has been historically prooved that aiming to get the real intrinsic value of a stock, one has to study its fundamentals. Therefore, I picked up several ratios which cover from the eranings ratios as the Price Earning Ratio to liquidity ratios, e.g.: the Current Ratio. 
The outcome y_hat which I used to train the model where the stock prices from Apple. Accroding to an article published from [applyed science](file:///Users/usuario/Downloads/applsci-10-08142-v2.pdf) .


## Datasets
For this project, two types of data where required. On the one hand the time series data for AAPL stocks from the last 30 years and on the other hand the AAPL financial ratios.
For the data cleaning we firstly had to access the information on the csv file, and then turn it into a monthly datetime index (since the index was the common column for both sets). The reason for a 'M' datetime is that ratios do only get published on a quarterly basis. Therefore, the prediction for an investment decision would be on a monthly timeframe. 

After having merged the two sets, we had to get rid of the unnecessary information for this specific analysis, as were the adjusted closed price, the open, or the highest and lowest price of that month. The closed price of the stock is where we had to put our main focus, the Output Layer. However, according to several papers i read during this project I found out that making a precise estimation of the sotck price is not very reliable, nontheless, since using a logarithmic predicition method I had to convert the close column into a readable binomial column of values 0 or 1. To do so I chose to replace the positive returns with 1 and the negative with 0.

Comming to the ratios. Stock ratios get very different values, one ranging from 0 to 1 and others, as "days of inventory outsatnding" could be from 0 to 365, as we are talking about days. So, to normalize the ratios I used mean and variance normalization. Those ratios are now ready to become the input layers.

## Gradient Descent (Adam Optimization Algorithm)
The Adam combines two gradient descent tehniques, the Route Mean Square Prop (RMS) with the Momentum.

It basically calculates exponential weighted average of past gradients and stores it in a variable _v_, which then will be corrected from bias.
Sdw = _Beta*Sdw + (1-Beta)*dw^2_
Sdb = _Beta*Sdb + (1-Beta)*db^2_

w = _w - alpha*dw/(sqrt(Sdw)+e)_
b = _b - alpha*db/(sqrt(Sdb)+e)_

where e is for numerical stability 10^-8



Since the dataset is not large enough to implement stochastic gradient descent (or minibatch GD), Adam optimization algorithm was chosen.



## Learning Rate Decay
When talking about this hyperparameter, a fixed number muight struggle to converge, staying in a wide oscilation tha returns low accuracy. The idea is to adjust the alpha depending on the position of your gradient, if you have a high cost function you will be interested to move quickly from that area, wether if you are near low cost function, reducing the alpha with every iterationn helps to proper the optimum cost value.

import math
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style


style.use('fivethirtyeight')

'''

Defining my own liner regression algorithm
Linear Regression is basically the best fist line

 -> applying linear algebra for Linear Regression

y = mx + b

m = ( mean(x) x mean(y) - mean(xy) ) / ( mean(x)^2 - mean(x^2)  )

'''


xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs,ys):
    ''' Returns the slope of the best fit line for the expecified xs and ys'''
    top = ( mean(xs) * mean(ys) ) - mean(xs*ys)
    bottom = (mean(xs)**2) - mean(xs**2)   
    m  = top / bottom
    return m

def y_inter(xs, ys, m):
    ''' Returns the y intercept of the line with a specified slope
        m and points xs and ys'''
    return mean(ys) - ( m * mean(xs) )

m = best_fit_slope(xs, ys)
b = y_inter(xs, ys, m)
print(f'm is {m}')
print(f'b is {b}')



''' Create a line that fits the data that we have
    we have m and b all we need is a list of Ys '''

regression_line = [ (m*x) + b for x in xs]

#now we can predict
predict_x = 8
predict_y = (m*predict_x) + b


plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()




##plt.scatter (xs, ys)
##plt.show()

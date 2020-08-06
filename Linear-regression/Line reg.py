#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 04:43:35 2019

@author: nish
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')
#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
def data(hm,var,step=2,correlation=False):
        val=1
        ys=[]
        for i in range(hm):
                y=val+random.randrange(-var,var)
                ys.append(y)
                if correlation and correlation=='pos':
                        val+=step
                elif correlation and correlation=='neg':
                        val-=step
        xs=[i for i in range(len(ys))]
        return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)
def best_fit(xs,ys):
        m=((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))
        b=mean(ys)-m*mean(xs)
        return(m,b)
def sq_error(ys_orig,ys_line):
        return sum((ys_line-ys_orig)**2)
def coef(ys_orig,ys_line):
        ys_mean_line=[mean(ys_orig) for y in ys_orig] 
        sq_error_reg=sq_error(ys_orig,ys_line)
        sq_error_y_mean=sq_error(ys_orig,ys_mean_line)
        return 1-(sq_error_reg/sq_error_y_mean)
xs,ys=data(40,80,2,correlation='pos')
m,b=best_fit(xs,ys)
reg_line=[(m*x)+b for x in xs]
predict_x=int(input('ENter No=='))
predict_y=(m*predict_x)+b
r_sq=coef(ys,reg_line)
print(r_sq)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs,reg_line)
plt.show()



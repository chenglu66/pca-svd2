# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:23:34 2017

@author: Lenovo-Y430p
"""
from numpy import *

datamat=[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
def pca(datamat,nums):
    datamat=mat(datamat)
    meanmat=mean(datamat,axis=0)
    #print(shape(meanmat))#得到的向量是行向量
    #使用矩阵的广播，因此不需要在重新做
    #在两个数组上运行时，NumPy将元素的形状进行比较。它从尾随的维度开始，并向前推进。两个尺寸兼容
    #他们是平等的
    #其中一个是1
    redmeadata=datamat-meanmat
    covmat=cov(redmeadata,rowvar=0)#return 是数组
    eigval,eigmat=linalg.eig(mat(covmat))
    #print(type(eigval))
    eigindex=argsort(eigval)#按大小排序并取index是个ndarray
    #print(eigval[eigindex])
    reeigindex=eigindex[:-(nums+1):-1]
    reeigmat=eigmat[:,reeigindex]
    redatamat=redmeadata*reeigmat
    rebulid=redatamat*reeigmat.T+meanmat
    wucha=rebulid-datamat
    m,n=shape(rebulid)
    sum1=0
    for i in range(m):
        for j in range(n):
            sum1+=wucha[i,j]**2
    print(sum1)
    return redatamat
def main():
    #pca(datamat,nums=3)#误差29
    pca(datamat,nums=7)#误差0.73136
if __name__=='__main__':
    main()
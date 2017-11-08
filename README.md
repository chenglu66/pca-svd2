# pca-svd2
占坑以后填
在日常生活中数据通常是高维稀疏矩阵，这样距离、密度等用来衡量相似性或者异常检测的手段在高维时意义不大，因为整个矩阵比较稀疏，几何意义上就是点比较散，因此有必要使用一些降维的方法，让点变得不那么稀疏。但是在降维时，我们希望数据的本来信息不会丢失太多，最好是能去掉噪声保留原有信息，但是怎么去掉噪声，假设我对原数据按某种波形分解，假设噪声信号变换比较小，那么其方差也是比较小的，这样一来我可以做找到一组正交基底做投影，利用特征值去掉大小较小的，这样就可以，方差只是自己这一个特征的概念，扩展特征以后就是协方差的概念，最直观的就是协方差矩阵。PCA是把原始样本在特征空间内达到方差最大，这样就能尽可能保持原有信息。方法：在保持各个维度量纲一致的情况下，找到一个向量U使得数据在该向量的投影距离尽可能大，即方差尽可能大这样原本数据在特征空间内的信息就有可能最大限度保留。这个向量可以理解为协方差矩阵的主特征向量。应用场景：数据的可视化，把高维数据变成低维数据，压缩，剔除特征来预防过拟合。详细过程为：
![image](https://github.com/chenglu66/pca-svd2/blob/master/PCA%E6%B5%81%E7%A8%8B%E5%9B%BE.png)
######pca算法
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
#ok图不放了，下面看主要内容基于SVD的餐馆推荐系统设计
#首先介绍下SVD算法。

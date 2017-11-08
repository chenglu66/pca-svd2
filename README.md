# pca-svd2
PCA：在日常生活中数据通常是高维矩阵，这样距离、密度等用来衡量相似性或者异常检测的手段在高维时意义不大，因为整个矩阵会比较稀疏，几何意义上就是高维空间上点都比较散，因此有必要使用一些降维的方法，让点分布不那么稀疏。但是在降维时，我们希望数据的本来信息不会丢失太多，最好是能去掉一些无关紧要信息同时保留特征明显的，重要的信息，但是怎么去掉无关紧要的信息呢，假设我对原数据按某种波形分解，假设对模型影响较小的信号变换也比较小，那么其方差也是比较小的，方差只是自己这一个特征的概念，扩展特征以后就是协方差的概念，最直观的就是协方差矩阵。在保持各个维度量纲一致的情况下，找到一个矩阵U使得协方差矩阵数据在该向量的能量尽可能大，向量之间是正交。能量较大就是方差较大，正交就是特征之间不相关程度非常大。缺点：通过矩阵变换的形式只是把原信息做了某种线性组合，因此若是营造非线性的关系还需要借用核的概念，但比较复杂，另一个假设数据是高斯分布，如果数据不是高斯分布舍去的可能不是噪声。应用场景：数据的可视化，把高维数据变成低维数据，压缩，剔除特征来预防过拟合\citeup{Vaswani2017PCA}，使用协方差来衡量，这一点是不同SVD分解的，SVD分解直接滤掉的对原数据影响较小的信息。因此SVD在稀疏矩阵用的较多。。详细过程为：
![image](https://github.com/chenglu66/pca-svd2/blob/master/PCA%E6%B5%81%E7%A8%8B%E5%9B%BE.png)<br />
######pca算法<br />
from numpy import *<br />
datamat=[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],<br />
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],<br />
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],<br />
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],<br />
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],<br />
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],<br />
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],<br />
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],<br />
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],<br />
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],<br />
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]<br />
           
def pca(datamat,nums):<br />
    datamat=mat(datamat)<br />
    meanmat=mean(datamat,axis=0)<br />
    #print(shape(meanmat))#得到的向量是行向量<br />
    #使用矩阵的广播，因此不需要在重新做<br />
    #在两个数组上运行时，NumPy将元素的形状进行比较。它从尾随的维度开始，并向前推进。两个尺寸兼容<br />
    #他们是平等的<br />
    #其中一个是1<br />
    redmeadata=datamat-meanmat<br />
    covmat=cov(redmeadata,rowvar=0)#return 是数组<br />
    eigval,eigmat=linalg.eig(mat(covmat))<br />
    #print(type(eigval))<br />
    eigindex=argsort(eigval)#按大小排序并取index是个ndarray<br />
    #print(eigval[eigindex])<br />
    reeigindex=eigindex[:-(nums+1):-1]<br />
    reeigmat=eigmat[:,reeigindex]<br />
    redatamat=redmeadata*reeigmat<br />
    rebulid=redatamat*reeigmat.T+meanmat<br />
    wucha=rebulid-datamat<br />
    m,n=shape(rebulid)<br />
    sum1=0<br />
    for i in range(m):<br />
        for j in range(n):<br />
            sum1+=wucha[i,j]**2<br />
    print(sum1)<br />
    return redatamat<br />
def main():<br />
    #pca(datamat,nums=3)#误差29<br />
    pca(datamat,nums=7)#误差0.73136<br />
if __name__=='__main__':<br />
    main()<br />
#ok图不放了，下面看主要内容基于SVD的餐馆推荐系统设计<br />
#首先介绍下SVD算法。<br />
对于一般的矩阵，我想保留主要信息，直接操作原矩阵，还是想象成波形，能量小的波形对结果影响也比较小，因此直接做分解得到能量大小，从特征值定义我们知道特征向量是方向，而特征值是大小，因此特征值小的就是能量小。所以同PCA来比，SVD没有使用方差，而直接使用原数据忽略掉一些能量比较小的点，这也意味着在稀疏矩阵比较有用。并且SVD出来的矩阵奇异值与特征值类似，只不过特征值只有伸缩而没有选择变换。这样就ok了。实际效果上应该是SVD好一点。
上代码：<br />
主要是计算相似性上的区别吧。而相似性一般而言最好的是余弦相似，和皮尔逊相关系数。<br />
先上这代码
from numpy import *
#cos度量方式
data=[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
A=mat(data)[:,0]
B=mat(data)[:,4]
def cosrelation(A,B):
    num = float(A.T*B)
    denom = la.norm(A)*la.norm(B)
    #把相似度确定在0-1之间
    return 0.5+0.5*(num/denom)
#欧式距离度量方式
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))
#皮尔逊相关稀疏
def pi(x,y):
    xmean=mean(x,axis=0)
    ymean=mean(y,axis=0)
    xvar=dot((x-xmean).T,(x-xmean))
    print(xvar)
    print(cov(x,rowvar = 0)*6)
    covxy=dot((x-xmean).T,y-ymean)
    xvar=dot((x-xmean).T,x-xmean)
    yvar=dot((y-ymean).T,y-ymean)
    p=covxy/sqrt(xvar*yvar)
    print(cov(x,y,rowvar = 0))
    print(covxy)
    p=0.5+0.5*(p)
    score1=0.5+0.5*corrcoef(A, B, rowvar = 0)[0][1]
    print(float(p))
    print(score1)
'''
def pi(A,B):
    denom = la.norm(A)*la.norm(B)
    h=cov(A,B,rowvar=0)
    print(mat(h))
    print(corrcoef(A, B, rowvar = 0))
    score=0.5+0.5*(mat(h)[0,1]/denom)
    print(score)
    score1=0.5+0.5*corrcoef(A, B, rowvar = 0)[0][1]
    print(score1)
'''
def main():
    #pca(datamat,nums=3)#误差29
    score0=cosrelation(A,B)
    print(score0)
    score3=ecludSim(A,B)
    print(score3)
    pi(A,B)
if __name__=='__main__':
    main()
现在可以计算相似度了，那么推荐系统怎么做呢，我是基于用户还是物品，数据少都差不多，只是高维才会差别，不过我喜欢基于用户，而不是基于内容，这里还是基于物品吧。因为整个相似度都是列向量的计算。



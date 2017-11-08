# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:26:57 2017

@author: Lenovo-Y430p
"""
from numpy import *
#cos度量方式
from numpy import linalg as la
data=[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
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
    '''
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
    '''
    score1=0.5+0.5*corrcoef(x, y, rowvar = 0)[0][1]
    #print(float(p))
    #print(score1)
    return score1
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
def standEst(datamat,user,simmeans,item):
    n=shape(datamat)[1]
    simtotal=0.0;ratsimtotal=0.0
    for j in range(n):
        userrating=datamat[user,j]
        if userrating==0:
            continue
        overlap=nonzero(logical_and(datamat[:,item].A>0,datamat[:,j].A>0))[0]
        if len(overlap)==0:
            similartity=0
        else:
            similartity=simmeans(datamat[overlap,item],datamat[overlap,j])
            print(similartity)
        simtotal+=similartity
        ratsimtotal+=similartity*userrating
        if simtotal==0:
            return 0
        else:
            return ratsimtotal/simtotal
        
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4]*Sig4.I  #create transformed items
    #xformedItems1=U[:,:4] * Sig4*VT[:4,:]
    #print(xformedItems1)
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
            
            
def recommend(datamat,user,N=3,simMeas=cosrelation,estMethod=standEst):
    unratedItems = nonzero(datamat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(datamat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]    
        
        
'''
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
'''

def main():
    #pca(datamat,nums=3)#误差29
    #score0=cosrelation(A,B)
    #print(score0)
    #score3=ecludSim(A,B)
    #print(score3)
    #pi(A,B)
    t=recommend(mat(data),user=1,N=3,estMethod=standEst)
    t1=recommend(mat(data),user=1,N=3,estMethod=svdEst)
    print(t)
    print(t1)
if __name__=='__main__':
    main()
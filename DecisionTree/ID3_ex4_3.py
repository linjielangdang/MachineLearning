# -*- coding: utf-8 -*-
'''生成树的函数'''###############################################################

from numpy import *  
import numpy as np  
import pandas as pd  
from math import log  
import operator  
### 计算数据集的信息熵(Information Gain)增益函数(机器学习实战中信息熵叫香农熵)
def calcInfoEnt(dataSet):#本题中Label即好or坏瓜		#dataSet每一列是一个属性(列末是Label)
	numEntries = len(dataSet)						#		每一行是一个样本	
	labelCounts = {}								#给所有可能的分类创建字典labelCounts						
	for featVec in dataSet:							#按行循环：即rowVev取遍了数据集中的每一行
		currentLabel = featVec[-1]					#故featVec[-1]取遍每行最后一个值即Label
		if currentLabel not in labelCounts.keys():	#如果当前的Label在字典中还没有
			labelCounts[currentLabel] = 0			#则先赋值0来创建这个词
		labelCounts[currentLabel] += 1				#计数, 统计每类Label数量(这行不受if限制)
	InfoEnt = 0.0
	for key in labelCounts:							#遍历每类Label
		prob = float(labelCounts[key])/numEntries   #各类Label熵累加
		InfoEnt -= prob * log(prob,2)				#ID3用的信息熵增益公式
	return InfoEnt
  
### 对于离散特征: 取出该特征取值为value的所有样本
def splitDiscreteDataSet(dataSet, axis, value):		#dataSet是当前结点(待划分)集合
													#axis指示划分所依据的属性
													#value该属性用于划分的取值
	retDataSet = []									#为return Data Set分配一个列表用来储存																			
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]			#该特征之前的特征仍保留在样本dataSet中
			reducedFeatVec.extend(featVec[axis+1:]) #该特征之后的特征仍保留在样本dataSet中
			retDataSet.append(reducedFeatVec)		#把这个样本加到list中
	return retDataSet
 
### 对于连续特征: 返回特征取值大于value的所有样本(以value为阈值将集合分成两部分)
def splitContinuousDataSet(dataSet, axis, value): 
    retDataSetG=[]									#将储存取值大于value的样本
    retDataSetL=[]									#将储存取值小于value的样本  
    for featVec in dataSet:  
		if featVec[axis]>value:  
			reducedFeatVecG=featVec[:axis]
			reducedFeatVecG.extend(featVec[axis+1:])  
			retDataSetG.append(reducedFeatVecG)
		else:
			reducedFeatVecL=featVec[:axis]
			reducedFeatVecL.extend(featVec[axis+1:])  
			retDataSetL.append(reducedFeatVecL)
    return retDataSetG,retDataSetL					#返回两个集合, 是含2个元素的tuple形式
 
### 根据InfoGain选择当前最好的划分特征(以及对于连续变量还要选择以什么值划分)
def chooseBestFeatureToSplit(dataSet,labels):  
    numFeatures=len(dataSet[0])-1  
    baseEntropy=calcInfoEnt(dataSet)  
    bestInfoGain=0.0; bestFeature=-1  
    bestSplitDict={}  
    for i in range(numFeatures):  
		#遍历所有特征：下面这句是取每一行的第i个, 即得当前集合所有样本第i个feature的值
        featList=[example[i] for example in dataSet]  
		#判断是否为离散特征
        if not (type(featList[0]).__name__=='float' or type(featList[0]).__name__=='int'): 
### 对于离散特征：求若以该特征划分的熵增
			uniqueVals = set(featList)				#从列表中创建集合set(得列表唯一元素值)
			newEntropy = 0.0	
			for value in uniqueVals:				#遍历该离散特征每个取值
				subDataSet = splitDiscreteDataSet(dataSet, i, value)#计算每个取值的信息熵
				prob = len(subDataSet)/float(len(dataSet))
				newEntropy += prob * calcInfoEnt(subDataSet)#各取值的熵累加
			infoGain = baseEntropy - newEntropy		#得到以该特征划分的熵增 
### 对于连续特征：求若以该特征划分的熵增(区别：n个数据则需添n-1个候选划分点, 并选最佳划分点) 
        else: 
            #产生n-1个候选划分点  
            sortfeatList=sorted(featList)  
            splitList=[]  
            for j in range(len(sortfeatList)-1):  	#产生n-1个候选划分点
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)  
            bestSplitEntropy=10000                  #设定一个很大的熵值(之后用)
            #遍历n-1个候选划分点: 求选第j个候选划分点划分时的熵增, 并选出最佳划分点
            for j in range(len(splitList)):
                value=splitList[j]  
                newEntropy=0.0  
                DataSet = splitContinuousDataSet(dataSet, i, value)
                subDataSetG=DataSet[0]
                subDataSetL=DataSet[1]  
                probG = len(subDataSetG) / float(len(dataSet))  
                newEntropy += probG * calcInfoEnt(subDataSetG)  
                probL = len(subDataSetL) / float(len(dataSet))  
                newEntropy += probL * calcInfoEnt(subDataSetL)
                if newEntropy < bestSplitEntropy:  
                    bestSplitEntropy=newEntropy  
                    bestSplit=j
            bestSplitDict[labels[i]] = splitList[bestSplit]#字典记录当前连续属性的最佳划分点
            infoGain = baseEntropy - bestSplitEntropy	   #计算以该节点划分的熵增		
### 在所有属性(包括连续和离散)中选择可以获得最大熵增的属性
        if infoGain>bestInfoGain:  
            bestInfoGain=infoGain  
            bestFeature=i  
    #若当前节点的最佳划分特征为连续特征，则需根据“是否小于等于其最佳划分点”进行二值化处理
	#即将该特征改为“是否小于等于bestSplitValue”, 例如将“密度”变为“密度<=0.3815”
	#注意：以下这段直接操作了原dataSet数据, 之前的那些float型的值相应变为0和1
	#【为何这样做?】在函数createTree()末尾将看到解释
    if type(dataSet[0][bestFeature]).__name__=='float' or \
	type(dataSet[0][bestFeature]).__name__=='int':        
        bestSplitValue=bestSplitDict[labels[bestFeature]]          
        labels[bestFeature]=labels[bestFeature]+'<='+str(bestSplitValue)  
        for i in range(shape(dataSet)[0]):  
            if dataSet[i][bestFeature]<=bestSplitValue:  
                dataSet[i][bestFeature]=1  
            else:  
                dataSet[i][bestFeature]=0  
    return bestFeature      
  
### 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票：计算每类Label个数, 取max者
def majorityCnt(classList):  						
    classCount={}  									#将创建键值为Label类型的字典
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote]=0  					#第一次出现的Label加入字典
        classCount[vote]+=1  						#计数
    return max(classCount)
	
### 主程序：递归产生决策树
	# dataSet：当前用于构建树的数据集, 最开始就是data_full，然后随着划分的进行越来越小。这是因为进行到到树分叉点上了. 第一次划分之前17个瓜的数据在根节点，然后选择第一个bestFeat是纹理. 纹理的取值有清晰、模糊、稍糊三种；将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这时应该将划分的类别减少1以便于下次划分。 
	# labels：当前数据集中有的用于划分的类别(这是因为有些Label当前数据集没了, 比如假如到某个点上西瓜都是浅白没有深绿了)
	# data_full：全部的数据 
	# label_full:全部的类别 
numLine = numColumn = 2 #这句是因为之后要用global numLine……至于为什么我一定要用global
# 我也不完全理解。如果我只定义local变量总报错，我只好在那里的if里用global变量了。求解。
def createTree(dataSet,labels,data_full,labels_full):  
    classList=[example[-1] for example in dataSet] 
	#递归停止条件1：当前节点所有样本属于同一类；(注：count()方法统计某元素在列表中出现的次数)
    if classList.count(classList[0])==len(classList):  
        return classList[0]  
	#递归停止条件2：当前节点上样本集合为空集(即特征的某个取值上已经没有样本了)：
	global numLine,numColumn
	(numLine,numColumn)=shape(dataSet)
    if float(numLine)==0:  
        return 'empty'
	#递归停止条件3：所有可用于划分的特征均使用过了，则调用majorityCnt()投票定Label；
    if float(numColumn)==1:  
        return majorityCnt(classList)  
	#不停止时继续划分：
    bestFeat=chooseBestFeatureToSplit(dataSet,labels)#调用函数找出当前最佳划分特征是第几个
    bestFeatLabel=labels[bestFeat]  				#当前最佳划分特征
    myTree={bestFeatLabel:{}}  
    featValues=[example[bestFeat] for example in dataSet]  
    uniqueVals=set(featValues)  
    if type(dataSet[0][bestFeat]).__name__=='str':  
        currentlabel=labels_full.index(labels[bestFeat])  
        featValuesFull=[example[currentlabel] for example in data_full]  
        uniqueValsFull=set(featValuesFull)  
    del(labels[bestFeat]) #划分完后, 即当前特征已经使用过了, 故将其从“待划分特征集”中删去
    #【递归调用】针对当前用于划分的特征(beatFeat)的每个取值，划分出一个子树。  
    for value in uniqueVals:						#遍历该特征【现存的】取值
        subLabels=labels[:]  
        if type(dataSet[0][bestFeat]).__name__=='str':  
            uniqueValsFull.remove(value)  			#划分后删去(从uniqueValsFull中删!)
        myTree[bestFeatLabel][value]=createTree(splitDiscreteDataSet\
         (dataSet,bestFeat,value),subLabels,data_full,labels_full)#用splitDiscreteDataSet()
	#是由于, 所有的连续特征在划分后都被我们定义的chooseBestFeatureToSplit()处理成离散取值了。
    if type(dataSet[0][bestFeat]).__name__=='str':  #若该特征离散【更详见后注】
        for value in uniqueValsFull:#则可能有些取值已经不在【现存的】取值中了
									#这就是上面为何从“uniqueValsFull”中删去
									#因为那些现有数据集中没取到的该特征的值，保留在了其中
            myTree[bestFeatLabel][value]=majorityCnt(classList)  
    return myTree 

'''生成树调用的语句'''###########################################################

df=pd.read_csv('watermelon3_0_En.csv')  
data=df.values[:,1:].tolist()  
data_full=data[:]  
labels=df.columns.values[1:-1].tolist()  
labels_full=labels[:]  
myTree=createTree(data,labels,data_full,labels_full)  

'''绘决策树的函数'''#############################################################

import matplotlib.pyplot as plt  
decisionNode=dict(boxstyle="sawtooth",fc="0.8")  	#定义分支点的样式
leafNode=dict(boxstyle="round4",fc="0.8")  			#定义叶节点的样式
arrow_args=dict(arrowstyle="<-")  					#定义箭头标识样式
  
### 计算树的叶子节点数量  
def getNumLeafs(myTree):  
    numLeafs=0  
    firstStr=myTree.keys()[0]  
    secondDict=myTree[firstStr]  
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':  
            numLeafs+=getNumLeafs(secondDict[key])  
        else: numLeafs+=1  
    return numLeafs  

### 计算树的最大深度  
def getTreeDepth(myTree):  
    maxDepth=0  
    firstStr=myTree.keys()[0]  
    secondDict=myTree[firstStr]  
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':  
            thisDepth=1+getTreeDepth(secondDict[key])  
        else: thisDepth=1  
        if thisDepth>maxDepth:  
            maxDepth=thisDepth  
    return maxDepth  
  
### 画出节点  
def plotNode(nodeTxt,centerPt,parentPt,nodeType):  
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
    xytext=centerPt,textcoords='axes fraction',va="center", ha="center",\
    bbox=nodeType,arrowprops=arrow_args)  
  
### 标箭头上的文字  
def plotMidText(cntrPt,parentPt,txtString):  
    lens=len(txtString)  
    xMid=(parentPt[0]+cntrPt[0])/2.0-lens*0.002  
    yMid=(parentPt[1]+cntrPt[1])/2.0  
    createPlot.ax1.text(xMid,yMid,txtString)  
      
def plotTree(myTree,parentPt,nodeTxt):  
    numLeafs=getNumLeafs(myTree)  
    depth=getTreeDepth(myTree)  
    firstStr=myTree.keys()[0]  
    cntrPt=(plotTree.x0ff+\
    (1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)  
    plotMidText(cntrPt,parentPt,nodeTxt)  
    plotNode(firstStr,cntrPt,parentPt,decisionNode)  
    secondDict=myTree[firstStr]  
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD  
    for key in secondDict.keys():  
        if type(secondDict[key]).__name__=='dict':  
            plotTree(secondDict[key],cntrPt,str(key))  
        else:  
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW  
            plotNode(secondDict[key],\
            (plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)  
            plotMidText((plotTree.x0ff,plotTree.y0ff)\
            ,cntrPt,str(key))  
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD  
  
def createPlot(inTree):  
    fig=plt.figure(1,facecolor='white')  
    fig.clf()  
    axprops=dict(xticks=[],yticks=[])  
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)  
    plotTree.totalW=float(getNumLeafs(inTree))  
    plotTree.totalD=float(getTreeDepth(inTree))  
    plotTree.x0ff=-0.5/plotTree.totalW  
    plotTree.y0ff=1.0  
    plotTree(inTree,(0.5,1.0),'')  
    plt.show()

'''命令绘决策树的图'''#############################################################

createPlot(myTree)

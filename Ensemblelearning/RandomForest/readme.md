###随机森林是数据挖掘中非常常用的分类预测算法，以分类或回归的决策树为基分类器。算法的一些基本要点：
####
　　*对大小为m的数据集进行样本量同样为m的有放回抽样；

        *对K个特征进行随机抽样，形成特征的子集，样本量的确定方法可以有平方根、自然对数等；

        *每棵树完全生成，不进行剪枝；

        *每个样本的预测结果由每棵树的预测投票生成（回归的时候，即各棵树的叶节点的平均）


　　著名的Python机器学习包scikit learn的文档对此算法有比较详尽的介绍: http://scikit-learn.org/stable/modules/ensemble.html#random-forests

　　出于个人研究和测试的目的，基于经典的Kaggle 101　泰坦尼克号乘客的数据集，建立模型并进行评估。比赛页面及相关数据集的下载：https://www.kaggle.com/c/titanic

　　泰坦尼克号的沉没，是历史上非常著名的海难。突然感到，自己面对的不再是冷冰冰的数据，而是用数据挖掘的方法，去研究具体的历史问题，也是饶有兴趣。言归正传，模型的主要的目标，是希望根据每个乘客的一系列特征，如性别、年龄、舱位、上船地点等，对其是否能生还进行预测，是非常典型的二分类预测问题。数据集的字段名及实例如下：

```
PassengerId 	Survived 	Pclass 	Name 	Sex 	Age 	SibSp 	Parch 	Ticket 	Fare 	Cabin 	Embarked
1 	0 	3 	Braund, Mr. Owen Harris 	male 	22 	1 	0 	A/5 21171 	7.25 	  	S
2 	1 	1 	Cumings, Mrs. John Bradley (Florence Briggs Thayer) 	female 	38 	1 	0 	PC 17599 	71.2833 	C85 	C
3 	1 	3 	Heikkinen, Miss. Laina 	female 	26 	0 	0 	STON/O2. 3101282 	7.925 	  	S
4 	1 	1 	Futrelle, Mrs. Jacques Heath (Lily May Peel) 	female 	35 	1 	0 	113803 	53.1 	C123 	S
5 	0 	3 	Allen, Mr. William Henry 	male 	35 	0 	0 	373450 	8.05 	  	S
```

####
值得说明的是，SibSp是指sister brother spouse，即某个乘客随行的兄弟姐妹、丈夫、妻子的人数，Parch指parents,children
实际上，上面的代码还有很大的效率提升的空间，数据集不是很大的情况下，如果选择一个较大的输入参数，例如生成100棵树，就会显著地变慢；同时，将预测结果提交至kaggle进行评测，发现在测试集上的正确率不是很高，比使用sklearn里面相应的包进行预测的正确率（0.77512）要稍低一点 :-(  如果要提升准确率，两个大方向： 构造新的特征；调整现有模型的参数。

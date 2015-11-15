
'''test example
x= [
	 [0,20,0,0],
	 [0,20,0,1000],
	 [1,20,0,0],
	 [2,10,0,0],
	 [2,0,100,0],
	 [2,0,100,1000],
	 [1,0,100,1000],
	 [0,10,0,0],
	 [0,0,100,0],
	 [2,10,100,0],
	 [0,10,100,1000],
	 [1,10,0,1000],
	 [1,20,100,0],
	 [2,10,0,1000]
	]

y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
'''

def CalculatePrecisionRate(in_predict,in_label):
	Instance_num = len(in_predict)
	print "Total instance is ",Instance_num
	total_num = 0
	True_League = [0,0,0,0,0,0,0]
	Predict_League_Correct = [0,0,0,0,0,0,0]
	Predict_League = [0,0,0,0,0,0,0]

	for i in range(Instance_num):
		True_League[in_label[i]-1]+=1
		Predict_League[in_predict[i]-1]+=1

		if in_label[i] ==in_predict[i]:
			total_num+=1
			Predict_League_Correct[in_label[i]-1]+=1

	print "Total correct prediction ",total_num
	print "The correct prediction of each league is ",Predict_League_Correct
	print "The prediction of each league is ",Predict_League
	print "The actual instance in each league is ",True_League
			





'''Reading the data from the excel'''
import pandas as pd
d_Records = pd.read_excel('dataset.xlsx', 'training', index_col=None, na_values=['NA'])
d_ClassLabel   = pd.read_excel('dataset.xlsx','trainingValue',index_col= None,na_values=['NA'])
d_Validate = pd.read_excel('dataset.xlsx','validate',index_col = None,na_values =['NA'])
d_ValidateLabel = pd.read_excel('dataset.xlsx','validateValue',index_col = None,na_values =['NA'])

import numpy as np
x =np.asarray(d_Records)
y = np.asarray(d_ClassLabel['LeagueIndex']) 
z = np.asarray(d_Validate) 
m = np.asarray(d_ValidateLabel['LeagueIndex'])

'''this is for decision tree'''
from sklearn.externals.six import StringIO  
from sklearn import tree
import pydot 
dot_data = StringIO() 
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 6)
clf.fit(x,y)
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("decisionTree.pdf") 
'''thi is for decision tree prediction'''
DecisionTree_Predict = clf.predict(z)
CalculatePrecisionRate(DecisionTree_Predict,m)

'''This is for Bayes Classification'''
from sklearn.naive_bayes import GaussianNB
clf_Bayes = GaussianNB()
clf_Bayes.fit(x,y)
'''this is for bayes prediction'''
Bayes_Predict = clf_Bayes.predict(z)
CalculatePrecisionRate(Bayes_Predict,m)


for i in range(len(y)):
	print i
	print map("{0:.9f}".format,y[i])


Total instance is  1113
Total correct prediction  384
The correct prediction of each league is  [17, 59, 14, 142, 3, 146, 3]
The prediction of each league is  [36, 227, 58, 428, 5, 350, 9]
The actual instance in each league is  [42, 113, 189, 268, 270, 219, 12]



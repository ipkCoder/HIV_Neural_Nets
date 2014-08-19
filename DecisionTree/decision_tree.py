# Reference: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

from DataFile import *
#from ImportData import *
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

#data, target = getAllOfTheData()
trainX, trainY, valX, valY, testX, testY = getAllOfTheData()

rt = tree.DecisionTreeRegressor(min_samples_split=5, min_samples_leaf=3)

rt = rt.fit(trainX, trainY)
#rt = rt.fit(data, target)

pred = rt.predict(valX)
for i in range(len(pred)):
    print("{}\t{}".format(valY[i], pred[i]))
r2pred = rt.score(valX, valY)
print("R^2 val = {}".format(r2pred))

dot_data = StringIO()
tree.export_graphviz(rt, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("hiv_decision_tree.pdf") 

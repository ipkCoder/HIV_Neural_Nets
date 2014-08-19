
from ImportData import *
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

data, target = getAllOfTheData()
rt = tree.DecisionTreeRegressor(min_samples_split=5, min_samples_leaf=3)
rt = rt.fit(data, target)

dot_data = StringIO()
tree.export_graphviz(rt, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("hiv_decision_tree.pdf") 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = load_iris()
clf = DecisionTreeClassifier()
clf.fit(df.data, df.target)

plt.figure(figsize=(20, 20))
tree.plot_tree(clf,
               feature_names = df.feature_names, 
               class_names = df.target_names,
               rounded=True, 
               filled = True);

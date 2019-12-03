import numpy
import random
seed = 999 # seed was specified for reproducibility reasons
random.seed(seed)
numpy.random.seed(seed)
import os
import pandas
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix

df = pandas.read_csv("hw8_data.csv")

# pruning parameters
mss = 300 # set to 2 for default value used by the unpruned tree
msl = 10 # set to 1 for default value used by the unpruned tree
# depth is also a pruning parameter, but several depths were observed for comparison

mf = None # default none

mod1 = DecisionTreeClassifier(max_depth = 1, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod2 = DecisionTreeClassifier(max_depth = 2, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod3 = DecisionTreeClassifier(max_depth = 3, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod4 = DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod5 = DecisionTreeClassifier(max_depth = 5, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod6 = DecisionTreeClassifier(max_depth = 6, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod7 = DecisionTreeClassifier(max_depth = 7, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
modNone = DecisionTreeClassifier(max_depth = None, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)

for clf, label in zip([mod1, mod2, mod3, mod4, mod5, mod6, mod7, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7', 'modNone']):  
  print("Confusion matrices for {}".format(label))
  all_conf_matrices = []
  
  test_percent = 0.3 # => train_percent = 1 - test_percent
  val = random.sample(range(0, len(df)), int(test_percent*len(df)))
  test = df.iloc[val]
  train = df.drop(val, axis = 0)
  
  Y_test, Y_train = test["is_attack"], train["is_attack"]
  cols = [column for column in df if column != "is_attack"] # consider all predictors when building the tree
  X_test, X_train = test[cols].astype(float), train[cols].astype(float)

  clf.fit(X_train, Y_train)
  sklearn.tree.plot_tree(clf)

  fname = "depth_{}_tree.dot".format(label)

  export_graphviz(clf, out_file = "trees/{}".format(fname), feature_names=cols, filled=True, class_names=["No", "Yes"]) # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
  os.chdir("trees")
  os.system("dot -Tpng {0} -o {1}.png".format(fname, fname.replace(".dot","")))
  os.chdir("..")
  conf = confusion_matrix(Y_test, clf.predict(X_test))
  print(conf)
  
  print("-"*80)
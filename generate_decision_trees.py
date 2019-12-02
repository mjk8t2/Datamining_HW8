############
# STACKING #
############
import numpy
import random
seed = 999
random.seed(seed)
# numpy.random.seed(906)
import os
import pandas
import subprocess
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix

# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import matplotlib.gridspec as gridspec
# import itertools
# gs = gridspec.GridSpec(2, 2)
# fig = plt.figure(figsize=(10,8))

numpy.set_printoptions(suppress=True) # supress scientific notation
import warnings
warnings.filterwarnings('ignore')

df = pandas.read_csv("hw8_data.csv")

mss = 300
msl = 10
mf = None # default none

mod1 = DecisionTreeClassifier(max_depth = 1, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod2 = DecisionTreeClassifier(max_depth = 2, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod3 = DecisionTreeClassifier(max_depth = 3, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod4 = DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod5 = DecisionTreeClassifier(max_depth = 5, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod6 = DecisionTreeClassifier(max_depth = 6, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
mod7 = DecisionTreeClassifier(max_depth = 7, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
modNone = DecisionTreeClassifier(max_depth = None, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[mod1, mod2, mod3, mod4, mod5, modNone], meta_classifier=lr) # Create the ensemble classifier

for clf, label in zip([mod1, mod2, mod3, mod4, mod5, mod6, mod7, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7', 'modNone']):  
  print("Confusion matrices for {}".format(label))
  all_conf_matrices = []
  
  test_percent = 0.3 # => train_percent = 1 - test_percent
  val = random.sample(range(0, len(df)), int(test_percent*len(df)))
  test = df.iloc[val]
  train = df.drop(val, axis = 0)
  
  # test = df
  # train = df
  
  # print("test", len(test))
  # print("train:", len(train))

  Y_test, Y_train = test["is_attack"], train["is_attack"]
  # cols = [column for column in df if df.nunique()[column] >= 4]
  # cols = ["AIT201", "AIT202", "P101", "P203", "P205", "P301", "P401", "P601", "MV301", "MV302", "MV304", "UV401", "MV101", "MV201", "MV303"]
  # cols = ["AIT203", "LIT301", "DPIT301"]
  cols = [column for column in df if column != "is_attack"]
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




# num_folds = 2
# folds = {ii:[] for ii in range(num_folds)}
# yatk_count = 0
# natk_count = 0
# for index, row in df.iterrows():
#   if row["is_attack"] == 0:
#     folds[natk_count].append(index)
#     natk_count += 1
#   else:
#     folds[yatk_count].append(index)
#     yatk_count += 1
#   natk_count %= num_folds
#   yatk_count %= num_folds

# for clf, label in zip([mod1, mod2, mod3, mod4, mod5, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'modNone']):  
#   print("Confusion matrices for {}".format(label))
#   all_conf_matrices = []
#   for key, val in folds.items():
#     test = df.iloc[val]
#     train = df.drop(val, axis = 0)
  
#     Y_test, Y_train = test["is_attack"], train["is_attack"]
#     cols = [column for column in df if df.nunique()[column] >= 4]
#     X_test, X_train = test[cols].astype(float), train[cols].astype(float)

#     clf.fit(X_train, Y_train)
#     sklearn.tree.plot_tree(clf)
    
#     fname = "tree_{}_{}.dot".format(label, key)
    
#     export_graphviz(clf, out_file = "trees/{}".format(fname), feature_names=cols, filled=True, class_names=["No", "Yes"])
#     os.chdir("trees")
#     os.system("dot -Tpng {0} -o {1}.png".format(fname, fname.replace(".dot","")))
#     os.chdir("..")
#     conf = confusion_matrix(Y_test, clf.predict(X_test))
#     all_conf_matrices.append(conf)

#   avg_confus_matrix = all_conf_matrices[0]
#   for ii in range(1, len(all_conf_matrices)):
#     avg_confus_matrix = numpy.add(avg_confus_matrix, all_conf_matrices[ii])
#   avg_confus_matrix = (avg_confus_matrix/len(all_conf_matrices))
#   print(avg_confus_matrix)
#   print("-"*80)


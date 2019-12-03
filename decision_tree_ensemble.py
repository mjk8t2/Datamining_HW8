############
# ADABOOST #
############
import pandas
import numpy
import os
from sklearn.tree import export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

seed = 999

df = pandas.read_csv("hw8_data.csv")

# cols = [column for column in df if df.nunique()[column] >= 4]
cols = [column for column in df if column != 'is_attack']

Y = df["is_attack"]
X = df[cols].astype(float)

mss = 300
msl = 10

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# print("classifying...")
# mss = 300
# msl = 10
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini"), n_estimators = 20, random_state = seed)
# classifier.fit(X_train, Y_train)

# y_pred = classifier.predict(X_test)

# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
# cf = confusion_matrix(Y_test, y_pred)
# print("Confusion matrix on 30/70 test/train split")
# print(cf)

# # for ii, tree in enumerate(classifier.estimators_):
# #   # print(tree)
# #   # print(classifier.estimator_weights_[ii])
# #   fname = "tree_{}_weight_{}.dot".format(ii+1, classifier.estimator_weights_[ii])
# #   clf = tree
# #   export_graphviz(clf, out_file = "adaboost_decision_trees/{}".format(fname), feature_names=cols, filled=True, class_names=["No", "Yes"], proportion=False) # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# #   os.chdir("adaboost_decision_trees")
# #   os.system("dot -Tpng {0} -o {1}.png".format(fname, fname.replace(".dot","")))
# #   os.chdir("..")

# attribute_importances = sorted([(ii, jj) for ii, jj in zip(cols, classifier.feature_importances_)], key = lambda x : x[1], reverse=True)

# for ii in attribute_importances:
#   print(ii)


# stratified kfold cross validation
df = df.sample(frac=1, random_state=seed) # shuffle the dataframe first
num_folds = 3
folds = {ii:[] for ii in range(num_folds)}
yatk_count = 0
natk_count = 0
for index, row in df.iterrows():
  if row["is_attack"] == 0:
    folds[natk_count].append(index)
    natk_count += 1
  else:
    folds[yatk_count].append(index)
    yatk_count += 1
  natk_count %= num_folds
  yatk_count %= num_folds


# print("Confusion matrices for {}".format(label))
# all_conf_matrices = []
for key, val in folds.items():
  otherval = list(set(range(len(df))) - set(val)) # train on all the other data
  test = df.iloc[val]
  train = df.iloc[otherval]

  # print("testlen", len(test))
  # print(len(train))
  # # s1 = pandas.merge(test, train, how='inner')
  # # print(s1)
  # # print(len(s1))
  # # # train = df.drop(val, axis = 0)

  
  # # # print(len(val), len(othervals))
  # # exit()

  # # test.to_csv("test.csv")
  # # train.to_csv("train.csv")

  Y_test, Y_train = test["is_attack"], train["is_attack"]
  # cols = [column for column in df if df.nunique()[column] >= 4]
  X_test, X_train = test[cols].astype(float), train[cols].astype(float)
  clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini"), n_estimators = 5, random_state = seed)
  clf.fit(X_train, Y_train)
  conf = confusion_matrix(Y_test, clf.predict(X_test))
  print(conf)






# DOESN'T WORK AND I DON'T KNOW WHY stratified kfold cross validation

# from sklearn.model_selection import StratifiedKFold # https://stackoverflow.com/a/51854653
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn import metrics
# folds = 3
# skf = StratifiedKFold(n_splits = folds)
# for train_index, test_index in skf.split(X, Y):
#   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#   Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

#   classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini"), n_estimators = 5, random_state = seed)
#   classifier.fit(X_train, Y_train)


#   y_pred = classifier.predict(X_test)
#   print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
#   cf = confusion_matrix(Y_test, y_pred)
#   print("Confusion matrix on {}-fold split".format(folds))
#   print(cf)














#   OLD #

# mss = 300
# msl = 10
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini"), n_estimators = 5, random_state = seed)
# # scores = cross_val_predict(classifier, X, Y, cv=10)
# # print(scores)
# ypred = cross_val_predict(classifier, X, Y, cv=3)
# conf = confusion_matrix(Y, ypred)
# print(conf)



# classifier.fit(X_train, Y_train)

# y_pred = classifier.predict(X_test)

# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
# cf = confusion_matrix(Y_test, y_pred)
# print("Confusion matrix on 30/70 test/train split")
# print(cf)







# for ii, feature in enumerate(classifier.feature_importances_):
  # print(cols[ii], "\t", feature)

# ############
# # STACKING #
# ############
# import numpy
# import random
# seed = 999 # seed was specified for reproducibility reasons
# random.seed(seed)
# numpy.random.seed(seed)
# import os
# import pandas
# import subprocess
# import sklearn
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from mlxtend.classifier import StackingClassifier
# from sklearn.tree import export_graphviz
# from sklearn.metrics import confusion_matrix

# numpy.set_printoptions(suppress=True) # supress scientific notation
# import warnings
# warnings.filterwarnings('ignore')

# df = pandas.read_csv("hw8_data.csv")

# # pruning parameters
# mss = 300 # set to 2 for default value used by the unpruned tree
# msl = 10 # set to 1 for default value used by the unpruned tree
# # depth is also a pruning parameter, but several depths were observed for comparison

# mf = None # default none

# mod1 = DecisionTreeClassifier(max_depth = 1, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod2 = DecisionTreeClassifier(max_depth = 2, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod3 = DecisionTreeClassifier(max_depth = 3, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod4 = DecisionTreeClassifier(max_depth = 4, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod5 = DecisionTreeClassifier(max_depth = 5, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod6 = DecisionTreeClassifier(max_depth = 6, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# mod7 = DecisionTreeClassifier(max_depth = 7, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# modNone = DecisionTreeClassifier(max_depth = None, min_samples_split=mss, min_samples_leaf=msl, criterion="gini", random_state=seed, max_features = mf)
# lr = LogisticRegression()
# sclf = StackingClassifier(classifiers=[mod1, mod2, mod3, mod4, mod5, modNone], meta_classifier=lr) # Create the ensemble classifier

# for clf, label in zip([mod1, mod2, mod3, mod4, mod5, mod6, mod7, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'mod6', 'mod7', 'modNone']):  
#   print("Confusion matrices for {}".format(label))
#   all_conf_matrices = []
  
#   test_percent = 0.3 # => train_percent = 1 - test_percent
#   val = random.sample(range(0, len(df)), int(test_percent*len(df)))
#   test = df.iloc[val]
#   train = df.drop(val, axis = 0)
  
#   # test = df
#   # train = df
  
#   # print("test", len(test))
#   # print("train:", len(train))

#   Y_test, Y_train = test["is_attack"], train["is_attack"]
#   # cols = [column for column in df if df.nunique()[column] >= 4]
#   # cols = ["AIT201", "AIT202", "P101", "P203", "P205", "P301", "P401", "P601", "MV301", "MV302", "MV304", "UV401", "MV101", "MV201", "MV303"]
#   # cols = ["AIT203", "LIT301", "DPIT301"]
#   cols = [column for column in df if column != "is_attack"]
#   X_test, X_train = test[cols].astype(float), train[cols].astype(float)

#   clf.fit(X_train, Y_train)
#   sklearn.tree.plot_tree(clf)

#   fname = "depth_{}_tree.dot".format(label)

#   export_graphviz(clf, out_file = "trees/{}".format(fname), feature_names=cols, filled=True, class_names=["No", "Yes"]) # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
#   os.chdir("trees")
#   os.system("dot -Tpng {0} -o {1}.png".format(fname, fname.replace(".dot","")))
#   os.chdir("..")
#   conf = confusion_matrix(Y_test, clf.predict(X_test))
#   print(conf)
  
#   print("-"*80)




# # num_folds = 2
# # folds = {ii:[] for ii in range(num_folds)}
# # yatk_count = 0
# # natk_count = 0
# # for index, row in df.iterrows():
# #   if row["is_attack"] == 0:
# #     folds[natk_count].append(index)
# #     natk_count += 1
# #   else:
# #     folds[yatk_count].append(index)
# #     yatk_count += 1
# #   natk_count %= num_folds
# #   yatk_count %= num_folds

# # for clf, label in zip([mod1, mod2, mod3, mod4, mod5, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'modNone']):  
# #   print("Confusion matrices for {}".format(label))
# #   all_conf_matrices = []
# #   for key, val in folds.items():
# #     test = df.iloc[val]
# #     train = df.drop(val, axis = 0)
  
# #     Y_test, Y_train = test["is_attack"], train["is_attack"]
# #     cols = [column for column in df if df.nunique()[column] >= 4]
# #     X_test, X_train = test[cols].astype(float), train[cols].astype(float)

# #     clf.fit(X_train, Y_train)
# #     sklearn.tree.plot_tree(clf)
    
# #     fname = "tree_{}_{}.dot".format(label, key)
    
# #     export_graphviz(clf, out_file = "trees/{}".format(fname), feature_names=cols, filled=True, class_names=["No", "Yes"])
# #     os.chdir("trees")
# #     os.system("dot -Tpng {0} -o {1}.png".format(fname, fname.replace(".dot","")))
# #     os.chdir("..")
# #     conf = confusion_matrix(Y_test, clf.predict(X_test))
# #     all_conf_matrices.append(conf)

# #   avg_confus_matrix = all_conf_matrices[0]
# #   for ii in range(1, len(all_conf_matrices)):
# #     avg_confus_matrix = numpy.add(avg_confus_matrix, all_conf_matrices[ii])
# #   avg_confus_matrix = (avg_confus_matrix/len(all_conf_matrices))
# #   print(avg_confus_matrix)
# #   print("-"*80)


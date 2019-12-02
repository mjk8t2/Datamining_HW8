

############
# ADABOOST #
############
# import pandas
# import numpy
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix


# df = pandas.read_csv("hw8_data.csv")

# Y = df["is_attack"]
# X = df[[column for column in df if df.nunique()[column] >= 4]].astype(float)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# print("classifying...")
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth = None), n_estimators = 200) 
# classifier.fit(X_train, Y_train)

# y_pred = classifier.predict(X_test)

# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
# cf = confusion_matrix(Y_test, y_pred)
# print(cf)

############
# STACKING #
############
import numpy
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

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier(max_depth = None)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf4], meta_classifier=lr) # Create the ensemble classifier

# mod1 = DecisionTreeClassifier(max_depth = 1)
# mod2 = DecisionTreeClassifier(max_depth = 2)
# mod3 = DecisionTreeClassifier(max_depth = 3)
# mod4 = DecisionTreeClassifier(max_depth = 4)
# mod5 = DecisionTreeClassifier(max_depth = 5)
# modNone = DecisionTreeClassifier(max_depth = None)
# sclf = StackingClassifier(classifiers=[mod1, mod2, mod3, mod4, mod5, modNone], meta_classifier=lr) # Create the ensemble classifier

num_folds = 2
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

# for clf, label in zip([mod1, mod2, mod3, mod4, mod5, modNone], ['mod1', 'mod2', 'mod3', 'mod4', 'mod5', 'modNone']):  
for clf, label in zip([clf1, clf2, clf4, sclf], ['KNN', 'Random Forest', 'DecisionTree', 'StackingClassifier']):
  print("Confusion matrices for {}".format(label))
  all_conf_matrices = []
  for key, val in folds.items():
    test = df.iloc[val]
    train = df.drop(val, axis = 0)
  
    Y_test, Y_train = test["is_attack"], train["is_attack"]
    cols = [column for column in df if df.nunique()[column] >= 4]
    X_test, X_train = test[cols].astype(float), train[cols].astype(float)

    clf.fit(X_train, Y_train)
    # sklearn.tree.plot_tree(clf)
    # export_graphviz(clf, out_file = "trees/tree{}_{}.dot".format(label, key))
    # subprocess.call("dot -Tpng tree.dot -o out.png")
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    conf = confusion_matrix(Y_test, clf.predict(X_test))
    all_conf_matrices.append(conf)

  avg_confus_matrix = all_conf_matrices[0]
  for ii in range(1, len(all_conf_matrices)):
    avg_confus_matrix = numpy.add(avg_confus_matrix, all_conf_matrices[ii])
  avg_confus_matrix = (avg_confus_matrix/len(all_conf_matrices))
  print(avg_confus_matrix)
  print("-"*80)


# # https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
# print("Wait while graph displays being prepared...\n")
# y = df["is_attack"].to_numpy()
# # X = df[["AIT201", "AIT202"]].to_numpy()
# # X = df[["AIT203", "LIT301"]].to_numpy()
# # cols = ["PIT502", "PIT503"]
# # cols = ["AIT201", "AIT202"]
# cols = ["AIT203", "LIT301"]
# X1 = df[cols].to_numpy()


# # X = df[["AIT201", "AIT202", "LIT301"]].to_numpy()
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import matplotlib.gridspec as gridspec
# import itertools

# # pca = PCA(n_components = 2)
# # cols = [column for column in df if df.nunique()[column] >= 4]
# # X2 = df[cols].astype(float)
# # X2 = pca.fit_transform(X2)

# clf4.fit(X1, y)
# fig = plt.figure(figsize=(10,8))
# fig = plot_decision_regions(X=X1, y=y, clf=clf4)
# plt.xlabel(cols[0])
# plt.ylabel(cols[1])
# plt.title("Decision Tree")

# # gs = gridspec.GridSpec(2, 2)
# # fig = plt.figure(figsize=(10,8))
# # for clf, lab, grd in zip([clf1, clf2, clf4, sclf], ['KNN', 'Random Forest', 'DecisionTree', 'StackingClassifier'], itertools.product([0, 1], repeat=2)):
# #   clf.fit(X, y)
# #   ax = plt.subplot(gs[grd[0], grd[1]])
# #   fig = plot_decision_regions(X=X, y=y, clf=clf)
# #   plt.xlabel(cols[0])
# #   plt.ylabel(cols[1])
# #   plt.title(lab)
# plt.show()
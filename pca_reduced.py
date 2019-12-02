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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

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


# https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
print("Wait while graph displays being prepared...\n")
y = df["is_attack"].to_numpy()

pca = PCA(n_components = 2)
cols = [column for column in df if df.nunique()[column] >= 2 if column != "is_attack"]
X2 = df[cols].astype(float)
X2 = StandardScaler().fit_transform(X2)
X2 = pca.fit_transform(X2)
print(pca.components_)
print(pca.explained_variance_ratio_, numpy.sum(pca.explained_variance_ratio_))
cols = ["PCA1", "PCA2"]

clf4.fit(X2, y)
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X2, y=y, clf=clf4)
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.title("Decision Tree")

# gs = gridspec.GridSpec(2, 2)
# fig = plt.figure(figsize=(10,8))
# for clf, lab, grd in zip([clf1, clf2, clf4, sclf], ['KNN', 'Random Forest', 'DecisionTree', 'StackingClassifier'], itertools.product([0, 1], repeat=2)):
#   clf.fit(X, y)
#   ax = plt.subplot(gs[grd[0], grd[1]])
#   fig = plot_decision_regions(X=X, y=y, clf=clf)
#   plt.xlabel(cols[0])
#   plt.ylabel(cols[1])
#   plt.title(lab)
plt.show()
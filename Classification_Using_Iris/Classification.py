import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\MACHINE_LEARNING_MINI_PROJECTS\Classification_Using_Iris\Iris.csv")
# print(df.head())

# ----------------- PAIR PLOT -----------------
pair_plot = sns.pairplot(df.drop(["Id"], axis=1), hue='Species', markers=["o", "s", "D"])
plt.savefig(r"C:\MACHINE_LEARNING_MINI_PROJECTS\Classification_Using_Iris\pair_plot.png")
plt.show()

# -----------------VIOLIN PLOT-----------------
plt.figure(figsize=(12, 6))
sns.violinplot(x='Species', y='SepalLengthCm', data=df, inner='quartile', palette='muted')
plt.title('Violin Plot of Sepal Length by Species')
plt.savefig(r"C:\MACHINE_LEARNING_MINI_PROJECTS\Classification_Using_Iris\violin_plot.png")
plt.show()

# ----------------- APPLYING DIMENSIONALITY REDUCTION -----------------
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

iris = df.copy()
data = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
target = iris['Species'].values
pca = PCA(n_components=2)
iris['pca1'] = pca.fit_transform(data)[:, 0]
iris['pca2'] = pca.fit_transform(data)[:, 1]

lda = LDA(n_components=2)
iris['lda1'] = lda.fit(data, target).transform(data)[:, 0]
iris['lda2'] = lda.fit(data, target).transform(data)[:, 1]

# ----------------- PLOT PCA AND LDA RESULTS -----------------
setosa = iris.query("Species == 'Iris-setosa'")
versicolor = iris.query("Species == 'Iris-versicolor'")
virginica = iris.query("Species == 'Iris-virginica'")
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.kdeplot(x=setosa.pca1, y=setosa.pca2, ax=ax[0], label='Iris-setosa', color='blue', shade=True)
sns.kdeplot(x=versicolor.pca1, y=versicolor.pca2, ax=ax[0], label='Iris-versicolor', color='orange', shade=True)
sns.kdeplot(x=virginica.pca1, y=virginica.pca2, ax=ax[0], label='Iris-virginica', color='green', shade=True)
ax[0].set_title('PCA of Iris Dataset')
ax[0].set_xlabel('PCA1')
ax[0].set_ylabel('PCA2')

sns.kdeplot(x=setosa.lda1, y=setosa.lda2, ax=ax[1], label='Iris-setosa', color='blue', shade=True)
sns.kdeplot(x=versicolor.lda1, y=versicolor.lda2, ax=ax[1], label='Iris-versicolor', color='orange', shade=True)
sns.kdeplot(x=virginica.lda1, y=virginica.lda2, ax=ax[1], label='Iris-virginica', color='green', shade=True)
ax[1].set_title('LDA of Iris Dataset')     
ax[1].set_xlabel('LDA1')
ax[1].set_ylabel('LDA2')

plt.savefig(r"C:\MACHINE_LEARNING_MINI_PROJECTS\Classification_Using_Iris\pca_lda_plot.png")
plt.show()

# ----------------- MODEL SELECTION --------------------

x = iris.drop(['Id','Species'], axis=1).copy()
y = iris['Species'].copy()
# print(x.shape, y.shape)

# ------------------ DATA ENCODING FOR SPECIES ------------------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=5)

# ------------------ MODEL CREATION ------------------
from sklearn.metrics import confusion_matrix
k_range = list(range(1, 25))
scrores_train = []
scrores_test = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)
    
    scrores_train.append(metrics.accuracy_score(y_train, y_pred_train))
    scrores_test.append(metrics.accuracy_score(y_test, y_pred_test))
 

# ------------------ PLOT TRAIN AND TEST SCORES ------------------

df_plot = pd.DataFrame({'score_train': scrores_train, 'score_test': scrores_test})
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
df_plot.plot(figsize=(12, 6))
plt.title('Train and Test Scores for KNN Classifier')
plt.xlabel('K Value')
plt.ylabel('Accuracy Score')
plt.xticks(k_range)
plt.grid()
plt.savefig(r"C:\MACHINE_LEARNING_MINI_PROJECTS\Classification_Using_Iris\knn_scores_plot.png")
plt.show()
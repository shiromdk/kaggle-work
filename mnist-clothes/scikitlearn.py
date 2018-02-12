
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

labeled_images = pd.read_csv('fashion-mnist_train.csv')
pca = PCA(n_components = 50)

images = labeled_images.iloc[0:60000,1:]
labels = labeled_images.iloc[0:60000,:1]
images = pca.fit(images).transform(images)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


test_images[test_images>0]=1
train_images[train_images>0]=1

print("TRAINING ")
clf = MLPClassifier(solver='sgd', alpha=1e-5, random_state=21, max_iter=200)
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))



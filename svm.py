from sklearn import svm, linear_model
#from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from PIL import Image
from os import listdir
from random import shuffle
import numpy as np
import pickle

train_size = 0.8

soup_dir = "soup/"
salad_dir = "salad/"
soup_photos = [f for f in listdir(soup_dir)]
salad_photos = [f for f in listdir(salad_dir)]
shuffle(soup_photos)
shuffle(salad_photos)

train_images = []
train_classes = []

test_images = []
test_classes = []
train = []
test = []

# 1 is soup, 0 is salad
print("Saving soups")
for i, s in enumerate(soup_photos):
    try:
        im = Image.open(soup_dir + s).convert("L")
    except:
        continue
    y = np.asarray(im.getdata(), dtype=np.float64)
    if i < int(len(soup_photos)*train_size):
        train.append((y,1))
        #train_images.append(y)
        #train_classes.append(1)
    else:
        test.append((y,1))
        #test_images.append(y)
        #test_classes.append(1)
print("Saving salads")
for i, s in enumerate(salad_photos):
    try:
        im = Image.open(salad_dir + s).convert("L")
    except:
        continue
    y = np.asarray(im.getdata(), dtype=np.float64)
    if i < int(len(salad_photos)*train_size):
        train.append((y,0))
        #train_images.append(y)
        #train_classes.append(0)
    else:
        test.append((y,0))
        #test_images.append(y)
        #test_classes.append(0)

# shuffle everything around
shuffle(train)
shuffle(test)
for i in train:
    train_classes.append(i[1])
    train_images.append(i[0])
for i in test:
    test_classes.append(i[1])
    test_images.append(i[0])


print("Training model")
#clf = svm.SVC()
clf = svm.LinearSVC()
print("Training model")
pca = PCA()
pca.fit(train_images)
X = pca.transform(train_images)
#clf = svm.SVC(kernel='linear')
clf.fit(X, train_classes)
#print("Saving model")
#pickle.dump(clf, open("model.p", "w"))
#print("Saving test images/classes")
#pickle.dump((test_images, test_classes), open("test.p", "w"))

print("Assessing model fit")
predictions = clf.predict(X)
num_right = 0
for i, p in enumerate(predictions):
    if p == train_classes[i]:
        num_right += 1
print(num_right)
print(len(train_classes))

print("Testing model")
X = pca.transform(test_images)
predictions = clf.predict(X)
num_right = 0
for i, p in enumerate(predictions):
    if p == test_classes[i]:
        num_right += 1
print(num_right)
print(len(test_classes))


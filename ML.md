
# sklearn naive bayes GaussianNB官网实例
### 测试数据  
```python
import numpy as np  
features_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])  
labels_train = np.array([1, 1, 1, 2, 2, 2])  
### 引入高斯朴素贝叶斯  
from sklearn.naive_bayes import GaussianNB  
### 实例化  
clf = GaussianNB()  
### 训练数据 fit相当于train  
clf.fit(features_train, labels_train)   
### 输出单个预测结果  
features_test = np.array([-0.8,-1])  
labels_test = np.array([1])  
pred = clf.predict(features_test)  
print(pred)  
### 准确度评估 评估正确/总数  
### v方法1  
accuracy = clf.score(features_test, labels_test)  
### 方法2  
from sklearn.metrics import accuracy_score  
accuracy2 = accuracy_score(pred,labels_test)  
```

# 优达地形数据GaussianNB实例
# 一共四个PY文件
### studentMain.py
```python
#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)



### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
```
### calss_vis.py
```python
#!/usr/bin/python

#from udacityplots import *
import warnings
warnings.filterwarnings("ignore")

import matplotlib 
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

#import numpy as np
#import matplotlib.pyplot as plt
#plt.ioff()

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
    
import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print image_start+json.dumps(data)+image_end
```
### prep_terrain_data.py
```python
#!/usr/bin/python
import random


def makeTerrainData(n_points=1000):
###############################################################################
### make the toy dataset
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

#    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
#            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test
#    return training_data, test_data
```
### ClassifyNB.py
```python
def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
```



# 计算GaussianNB准确性实例
### 两个py文件
### classify.py
```python
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()  

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train) 

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)  


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)  
    return accuracy
```
### studentcode.py
```python
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy
```

# 朴素贝叶斯分类邮件作者实例
### email_preprocess.py
```python
#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
```
### nb_author_id.py
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
            ### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB() 
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)  
print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy

#########################################################

```

# 支持向量机实例
### 与优达地形数据高斯NB使用相同py文件
```python
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")

clf.fit(features_train, labels_train) 

pred = clf.predict(features_test)  



#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred





from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```

# 支持向量机分类邮件实例(线性,和rbf)
### 与朴素贝叶斯分类邮件实例使用相同的email_preprocess.py文件
### svm_author_id.py
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="linear")
#使用1%数据训练
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)  
print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy
#########################################################

```
## 上面的支持向量机使用 RBF 内核并优化 C 值
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C = 10000.0)
#使用1%数据训练
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)  
print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy
#########################################################

```
## 从 SVM 提取预测
### 使用SVM对训练集数据进行训练，然后使用训练后的模型，对测试数据集进行预测，预测结果是0或者1，0代表SARA，1代表Chris。题目想让你回答，第10个元素，第26个元素，第50个元素对应的预测值。 （所以方框里填0，或者1.）
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C = 10000.0)
#使用1%数据训练
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)  
print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy
print pred[10]
print pred[26]
print pred[50]
#########################################################

```
## 预测有多少邮件属于1(Chris)
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C = 10000.0)
#使用1%数据训练
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)  
print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy
#print pred[10]
#print pred[26]
#print pred[50]
#就在这计数的
print list(pred).count(1)
#########################################################

```

# 决策树编码
### 一共两个py文件
### studentMain.py
```python
#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)







#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

```
### classifyDT.py
```python
def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn import tree

    ### create classifier
    clf = tree.DecisionTreeClassifier()  

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train) 

    ### use the trained classifier to predict labels for the test features
    #pred = clf.predict(features_test)  
    
    
    return clf
```
## 上面决策树准确性
```python
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn import tree
from sklearn.metrics import accuracy_score
    ### create classifier
clf = tree.DecisionTreeClassifier()  

    ### fit the classifier on the training features and labels
clf.fit(features_train, labels_train) 

    ### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)  


acc = accuracy_score(pred, labels_test)
### be sure to compute the accuracy on the test set


    
def submitAccuracies():
  return {"acc":round(acc,3)}


```

## 参数一:最小样本分割与决策树准确性 2和50
```python
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively


from sklearn import tree
from sklearn.metrics import accuracy_score

# 2
### create classifier
clf2 = tree.DecisionTreeClassifier(min_samples_split = 2)  
### fit the classifier on the training features and labels
clf2.fit(features_train, labels_train) 
### use the trained classifier to predict labels for the test features
pred2 = clf2.predict(features_test)  
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)

# 50
### create classifier
clf50 = tree.DecisionTreeClassifier(min_samples_split = 50)  
### fit the classifier on the training features and labels
clf50.fit(features_train, labels_train) 
### use the trained classifier to predict labels for the test features
pred50 = clf50.predict(features_test)  
acc_min_samples_split_50 = accuracy_score(pred50, labels_test)


### be sure to compute the accuracy on the test set

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
```

# 决策树邮件分类作者,最小分割40,
```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score
### create classifier
clf = tree.DecisionTreeClassifier(min_samples_split=40)  

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train) 

### use the trained classifier to predict labels for the test features

#pred = clf.predict(features_test)  

accuracy = clf.score(features_test, labels_test)  
print accuracy
### be sure to compute the accuracy on the test set


#########################################################
```
## 你数据中的特征数是多少？
（提示：数据被整理成一个 numpy 数组后，行数是数据点数，列数是特征数；要提取这个数字，只需运行代码:
```python
len(features_train[0])
```
## 更改特征数量  更改percentile(可用特征)特征减少
### email_preprocess.py
```python
#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    

    #此行percentile = 10 -> 1
    #可用特征10% -> 1%


    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
```

# 使用k近邻(KNN)对道路颠簸分类
### 三个py文件
### class_vis.py
### prep_terrain_data.py
### your_algorithm.py
```python
#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary





from sklearn import neighbors  
clf = neighbors.KNeighborsClassifier()  
#使用1%数据训练
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
#t0 = time()
clf.fit(features_train, labels_train) 
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
pred = clf.predict(features_test)  
#print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy
print list(pred).count(1)





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

```

# 随机森林对道路颠簸分类
### 三个py文件
### class_vis.py
### prep_terrain_data.py
### your_algorithm.py
```python
#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary





from sklearn.ensemble import RandomForestClassifier
#决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率。
clf = RandomForestClassifier(n_estimators=10)

#使用1%数据训练
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#t0 = time()
clf.fit(features_train, labels_train) 
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
pred = clf.predict(features_test)  
#print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

```

# adaboost对道路颠簸分类
### 三个py文件
### class_vis.py
### prep_terrain_data.py
### your_algorithm.py
```python
#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary





from sklearn.ensemble import AdaBoostClassifier
#决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率。
clf = AdaBoostClassifier(n_estimators=100)

#使用1%数据训练
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#t0 = time()
clf.fit(features_train, labels_train) 
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
pred = clf.predict(features_test)  
#print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test, labels_test)  
print accuracy




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

```

# 安然迷你项目


```python
%run explore_enron_data.py
```


```python
#有多少人?
len(enron_data)
```




    146




```python
#人名?
for name in enron_data:
    print name
```

    


```python
#有多少特征?
enron_data
```




    




```python
#有多少嫌疑人?
n = 0
for name in enron_data:
    if enron_data[name]['poi']==1:
        n += 1
print n
```

    18
    


```python
#James Prentice名下的股票总值

#和任何字典的字典一样，个人/特征可以这样被访问：
#enron_data["LASTNAME FIRSTNAME"]["feature_name"]
#名字倒过来!

enron_data["PRENTICE JAMES"]["total_stock_value"]
```




    1095040




```python
#有多少来自 Wesley Colwell发给poi的邮件
enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
```




    11




```python
#Jeffrey k Skilling 行使的股票期权价值是多少？
enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
```




    19250000




```python
#谁拿的钱最多?Andrew Fastow  Jeffrey k Skilling     Kenneth Lay  
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
```

    8682716
    2424083
    103559793
    


```python
#多少雇员没工资?
n = 0
for name in enron_data:
    if enron_data[name]['salary']=='NaN':
        n += 1
print n

```

    51
    


```python
#多少员工没邮箱?
n = 0
for name in enron_data:
    if enron_data[name]['email_address']=='NaN':
        n += 1
print n
```

    35
    


```python
#多少员工薪酬总额被设置了“NaN”
n = 0
for name in enron_data:
    if enron_data[name]['total_payments']=='NaN':
        n += 1
print n
```

    21
    


```python
#poi中,总收入为nan的?
n = 0
for name in enron_data:
    if enron_data[name]["poi"]==True:
        if enron_data[name]['total_payments']=='NaN':
            n += 1
print n
```

    0
    


```python
#有多少个poi?
n = 0
for name in enron_data:
    if enron_data[name]["poi"]==True:
        n+=1
print n
```

    18
    


```python
#这就是说，在生成或增大数据集时，如果数据来自不同类的不同来源，你应格外小心。它很容易会造成我们在此展示的偏差或错误类型。可通过多种方法处理此问题。举例而言，如果仅使用了电子邮件数据，则你无需担心此问题（在这种情况下，财务数据中的差异并不重要，因为并未使用财务特征）。还可以通过更复杂的方法来估计这些偏差可能会对你的最终答案造成多大影响，不过此话题超出了本课程的范围。
#目前的结论就是，要非常小心地对待引入来自不同来源（具体取决于类）的特征这个问题！引入此类特征常常会意外地带来偏差和错误。
```


# 线性回归年龄/净值
### 两个py文件
### studentMain.py
```python
#!/usr/bin/python

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



reg = studentReg(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())
```
### studentRegresssion.py
```python
def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    ### your code goes here!
    from sklearn.linear_model import LinearRegression  
    clf = LinearRegression()

    clf.fit(ages_train, net_worths_train) 


    #pred = clf.predict(features_test)  
    #accuracy = clf.score(features_test, labels_test)  
    #print accuracy
    
    return reg
```

#提取线性回归的信息
### 两个py文件
### RegressionQuiz.py
```python
import numpy
import matplotlib.pyplot as plt

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

### get Katie's net worth (she's 27)
### sklearn predictions are returned in an array, so you'll want to index into
### the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
### exact syntax, the point is the [0] at the end). In addition, make sure the
### argument to your prediction function is in the expected format - if you get
### a warning about needing a 2d array for your data, a list of lists will be
### interpreted by sklearn as such (e.g. [[27]]).
km_net_worth = reg.predict([27]) ### fill in the line of code to get the right value

### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
slope = reg.coef_ ### fill in the line of code to get the right value

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
intercept = reg.intercept_ ### fill in the line of code to get the right value


### get the score on test data
test_score = reg.score(ages_test, net_worths_test) ### fill in the line of code to get the right value


### get the score on the training data
training_score = reg.score(ages_train, net_worths_train) ### fill in the line of code to get the right value



def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth":km_net_worth,
            "slope":slope,
            "intercept":intercept,
            "stats on test":test_score,
            "stats on training": training_score}
```
### ages_net_worths.py
```python
import numpy
import random

def ageNetWorthData():

    random.seed(42)
    numpy.random.seed(42)

    ages = []
    for ii in range(100):
        ages.append( random.randint(20,65) )
    net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
### need massage list into a 2d numpy array to get it to work in LinearRegression
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    from sklearn.cross_validation import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test
```

# 可视化回归安然工资和奖金并提取斜率截距测试成绩
### finance_regression.py
```python
#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(feature_train, target_train)



### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
print reg.coef_ ### fill in the line of code to get the right value

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
print reg.intercept_ ### fill in the line of code to get the right value


### get the score on test data
print reg.score(feature_test, target_test) ### fill in the line of code to get the right value


### get the score on the training data
print reg.score(feature_train, target_train) ### fill in the line of code to get the right value








### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

```
# 异常值破坏回归.安然工资预测奖金
### finance_regression.py
```python
#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(feature_train, target_train)



### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
print reg.coef_ ### fill in the line of code to get the right value

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
print reg.intercept_ ### fill in the line of code to get the right value


### get the score on test data
print reg.score(feature_test, target_test) ### fill in the line of code to get the right value


### get the score on the training data
print reg.score(feature_train, target_train) ### fill in the line of code to get the right value








### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
###############################################################
#一条在测试数据上拟合（有异常值），一条在训练数据上拟合（无异常值）
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b") 

### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
print reg.coef_ ### fill in the line of code to get the right value

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
print reg.intercept_ ### fill in the line of code to get the right value
##################################################################

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

```

# 清理异常值安然年龄净值
### outlier_cleaner.py
### 注释插进去报错,在这写吧.error是预测与实际偏差平方,zip把三个列表变成元组对.sorted使用key+匿名函数按error排序,num是计数,最后一行切片.返回最小error90%数据
```python
#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    errors = (predictions - net_worths) ** 2

    triplets = sorted(zip(ages, net_worths, errors),
                      key=lambda triplet: triplet[2])

    num_retain = int(len(predictions) * .9)

    return triplets[:num_retain]
```
### outlier_removal_regression.py
```python
#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner


### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )



### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
print reg.coef_ 
print reg.intercept_ 
print reg.score(ages_test, net_worths_test) 
print reg.score(ages_train, net_worths_train) 




try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()


### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"







### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
    print reg.coef_ 
    print reg.intercept_ 
    print reg.score(ages_test, net_worths_test) 
    print reg.score(ages_train, net_worths_train) 


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"


```
# 识别最大安然异常值total,并清理
### enron_outliers.py
```python
#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

################################################
### delete outlier
data_dict.pop("TOTAL", 0)
################################################
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



```

# k means聚类安然数据集 
## 两个特征,两个聚类中心
### k_means_cluster.py
```python
#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

#############################################################################
### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(finance_features)

pred = kmeans.labels_
#############################################################################

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

```
## 三个特征,两个聚类中心
### k_means_cluster.py
```python
#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

####################################################################################
### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()
####################################################################################
### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(finance_features)

pred = kmeans.labels_

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

```

# 工资特征范围
### 自己编写函数
```python
m = []
for name in data_dict:
    n = (data_dict[name]['salary'])
    if n != 'NaN':
        m.append(n)
print max(m)
print min(m)
```

# 特征缩放小练习
### 自制函数
```python
def featureScaling(arr):
    import numpy as np
    arr = np.array(arr)
    max = np.max(arr)
    min = np.min(arr)
    new=[]
    for item in arr:
        if max != min :
            float (item)
            item =float(item-min)/(max-min)
            new.append(item)
        else:
            item=0.5
            new.append(item)
    return new

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
```
### sklearn函数
```python
from sklearn.preprocessing import MinMaxScaler
import numpy
#这里numpy数组中的是特征，因为此处特征只有一个，所以看起来是这样的
#因为这里应该作为一个浮点数进行运算，所以数字后面要加.
weights = numpy.array([[115.],[140.],[175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
print rescaled_weight
```

# 特征缩放 安然工资股票 + 聚类k means
### k_means_cluster.py
```python
#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

########################################################################
from sklearn.preprocessing import MinMaxScaler

def scaler(feature_before):

    feature_after = MinMaxScaler().fit_transform(feature_before)

    return feature_after

########################################################################


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(scaler(finance_features))

pred = kmeans.labels_

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

```

# 文本分析
### nltk stopwords 
```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words("english")
len(sw)
```

# 使用nltk进行词干优化
```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")
stemmer.stem("responsivity")
stemmer.stem("unresponsive")
```

# 部署词干化
### parse_out_email_text.py
```python
#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
    ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

    ### project part 2: comment out the line below
        # words = text_string

    ### split the text string into individual words, stem each word,
    ### and append the stemmed word to words (make sure there's a single
    ### space between each stemmed word)
    ################################################################################
        words = text_string.split()
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        answ = []
        for w in words:
            s = stemmer.stem(w)
            if s:
                answ.append(s.rstrip())
    answ = ' '.join(answ) 
    return answ
    ###############################################################################   

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()


```

# 清除“签名文字”
### parse_out_email_text.py
```python
#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
    ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

    ### project part 2: comment out the line below
        # words = text_string

    ### split the text string into individual words, stem each word,
    ### and append the stemmed word to words (make sure there's a single
    ### space between each stemmed word)
    ###########################################################################
        words = text_string.split()
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        answ = []
        for w in words:
            s = stemmer.stem(w)
            if s:
                answ.append(s.rstrip())
    answ = ' '.join(answ) 
    return answ
    ############################################################################

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()


```
### vectorize_text.py
```python
#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

###############################################################################################
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter<200:
            ls = ""
            path = os.path.join('..', path[:-1])
            email = open(path, "rb")
            
            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]

            for wd in ['sara','shackleton','chris','germani','sshacklensf','cgermannsf']:
                words = words.replace(wd,'')
            
            ### append the text to word_data
            word_data.append(words)
            
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append("0")
            if name == "chris":
                from_data.append("1")

            email.close()

################################################################################################
print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

print word_data[152]



### in Part 4, do TfIdf vectorization here



```

# 进行 TfIdf
### parse_out_email_text.py
```python
#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        words_original = text_string.split()
        words = []
        # print words_original
        for word in words_original:
            stem_word = stemmer.stem(word)
            words.append(stem_word)

    return ' '.join(words)
    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()


```

### vectorize_text.py
```python
#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter<200000000:
            ls = ""
            path = os.path.join('..', path[:-1])
            email = open(path, "rb")
            
            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]

            for wd in ['sara','shackleton','chris','germani']:
                words = words.replace(wd,'')
            
            ### append the text to word_data
            word_data.append(words)
            
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append("0")
            if name == "chris":
                from_data.append("1")

            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

print word_data[152]



### in Part 4, do TfIdf vectorization here
######################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer(stop_words='english')
transformer.fit_transform(word_data)
print len(transformer.get_feature_names())
######################################################################################
```

# 你 TfId 中的单词编号 34597 是什么？
### vectorize_text.py
```python
transformer.get_feature_names()[34597]
```
# 特征选择,一个新的安然特征练习
### studentmain.py
```python
#!/usr/bin/python

import os
import sys
import zipfile
from poi_flag_email import poiFlagEmail, getToFromStrings

data_dict = {}

with zipfile.ZipFile('emails.zip', "r") as z:
    z.extractall()

for email_message in os.listdir("emails"):
    if email_message == ".DS_Store":
        continue
    message = open(os.getcwd()+"/emails/"+email_message, "r")
    to_addresses, from_addresses, cc_addresses = getToFromStrings(message) 
    
    to_poi, from_poi, cc_poi = poiFlagEmail(message)
    
    for recipient in to_addresses:
        # initialize counter
        if recipient not in data_dict:
            data_dict[recipient] = {"from_poi_to_this_person":0}
        # add to count
        if from_poi:
                data_dict[recipient]["from_poi_to_this_person"] += 1

    message.close()

for item in data_dict:
    print item, data_dict[item]
    
#######################################################    
def submitData():
    return data_dict
```
### poi_flag_email.py
```python
#!/usr/bin/python

###
### in poiFlagEmail() below, write code that returns a boolean
### indicating if a given email is from a POI
###

import sys
import reader
import poi_emails

def getToFromStrings(f):
    '''
    The imported reader.py file contains functions that we've created to help
    parse e-mails from the corpus. .getAddresses() reads in the opening lines
    of an e-mail to find the To: From: and CC: strings, while the
    .parseAddresses() line takes each string and extracts the e-mail addresses
    as a list.
    '''
    f.seek(0)
    to_string, from_string, cc_string   = reader.getAddresses(f)
    to_emails   = reader.parseAddresses( to_string )
    from_emails = reader.parseAddresses( from_string )
    cc_emails   = reader.parseAddresses( cc_string )

    return to_emails, from_emails, cc_emails


### POI flag an email

def poiFlagEmail(f):
    """ given an email file f,
        return a trio of booleans for whether that email is
        to, from, or cc'ing a poi """

    to_emails, from_emails, cc_emails = getToFromStrings(f)

    ### poi_emails.poiEmails() returns a list of all POIs' email addresses.
    poi_email_list = poi_emails.poiEmails()

    to_poi = False
    from_poi = False
    cc_poi   = False

    ### to_poi and cc_poi are boolean variables which flag whether the email
    ### under inspection is addressed to a POI, or if a POI is in cc,
    ### respectively. You don't have to change this code at all.

    ### There can be many "to" emails, but only one "from", so the
    ### "to" processing needs to be a little more complicated
    if to_emails:
        ctr = 0
        while not to_poi and ctr < len(to_emails):
            if to_emails[ctr] in poi_email_list:
                to_poi = True
            ctr += 1
    if cc_emails:
        ctr = 0
        while not cc_poi and ctr < len(cc_emails):
            if cc_emails[ctr] in poi_email_list:
                cc_poi = True
            ctr += 1

    #################################
    ######## your code below ########
    ### set from_poi to True if #####
    ### the email is from a POI #####
    #################################
    if from_emails:
        ctr = 0
        while not from_poi and ctr < len(from_emails):
            if from_emails[ctr] in poi_email_list:
                from_poi = True
            ctr += 1
    
    
    

    #################################
    return to_poi, from_poi, cc_poi
```

# 可视化新特征
### get_data.py
```python

def getData():
    data = {}
    data["METTS MARK"]={'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN', 'total_payments': 1061827, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'mark.metts@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 585062, 'expenses': 94299, 'from_poi_to_this_person': 38, 'exercised_stock_options': 'NaN', 'from_messages': 29, 'other': 1740, 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 702, 'restricted_stock': 585062, 'director_fees': 'NaN'}
    data["BAXTER JOHN C"]={'salary': 267102, 'to_messages': 'NaN', 'deferral_payments': 1295738, 'total_payments': 5634343, 'loan_advances': 'NaN', 'bonus': 1200000, 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -1386055, 'total_stock_value': 10623258, 'expenses': 11200, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 6680544, 'from_messages': 'NaN', 'other': 2660303, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 1586055, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 3942714, 'director_fees': 'NaN'}
    data["ELLIOTT STEVEN"]={'salary': 170941, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 211725, 'loan_advances': 'NaN', 'bonus': 350000, 'email_address': 'steven.elliott@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -400729, 'total_stock_value': 6678735, 'expenses': 78552, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 4890344, 'from_messages': 'NaN', 'other': 12961, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 1788391, 'director_fees': 'NaN'}
    data["CORDES WILLIAM R"]={'salary': 'NaN', 'to_messages': 764, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'bill.cordes@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1038185, 'expenses': 'NaN', 'from_poi_to_this_person': 10, 'exercised_stock_options': 651850, 'from_messages': 12, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 58, 'restricted_stock': 386335, 'director_fees': 'NaN'}
    data["HANNON KEVIN P"]={'salary': 243293, 'to_messages': 1045, 'deferral_payments': 'NaN', 'total_payments': 288682, 'loan_advances': 'NaN', 'bonus': 1500000, 'email_address': 'kevin.hannon@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -3117011, 'total_stock_value': 6391065, 'expenses': 34039, 'from_poi_to_this_person': 32, 'exercised_stock_options': 5538001, 'from_messages': 32, 'other': 11350, 'from_this_person_to_poi': 21, 'poi': True, 'long_term_incentive': 1617011, 'shared_receipt_with_poi': 1035, 'restricted_stock': 853064, 'director_fees': 'NaN'}
    data["MORDAUNT KRISTINA M"]={'salary': 267093, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 628522, 'loan_advances': 'NaN', 'bonus': 325000, 'email_address': 'kristina.mordaunt@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 208510, 'expenses': 35018, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 1411, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 208510, 'director_fees': 'NaN'}
    data["MEYER ROCKFORD G"]={'salary': 'NaN', 'to_messages': 232, 'deferral_payments': 1848227, 'total_payments': 1848227, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'rockford.meyer@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 955873, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 493489, 'from_messages': 28, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 22, 'restricted_stock': 462384, 'director_fees': 'NaN'}
    data["MCMAHON JEFFREY"]={'salary': 370448, 'to_messages': 2355, 'deferral_payments': 'NaN', 'total_payments': 4099771, 'loan_advances': 'NaN', 'bonus': 2600000, 'email_address': 'jeffrey.mcmahon@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1662855, 'expenses': 137108, 'from_poi_to_this_person': 58, 'exercised_stock_options': 1104054, 'from_messages': 48, 'other': 297353, 'from_this_person_to_poi': 26, 'poi': False, 'long_term_incentive': 694862, 'shared_receipt_with_poi': 2228, 'restricted_stock': 558801, 'director_fees': 'NaN'}
    data["HAEDICKE MARK E"]={'salary': 374125, 'to_messages': 4009, 'deferral_payments': 2157527, 'total_payments': 3859065, 'loan_advances': 'NaN', 'bonus': 1150000, 'email_address': 'mark.haedicke@enron.com', 'restricted_stock_deferred': -329825, 'deferred_income': -934484, 'total_stock_value': 803094, 'expenses': 76169, 'from_poi_to_this_person': 180, 'exercised_stock_options': 608750, 'from_messages': 1941, 'other': 52382, 'from_this_person_to_poi': 61, 'poi': False, 'long_term_incentive': 983346, 'shared_receipt_with_poi': 1847, 'restricted_stock': 524169, 'director_fees': 'NaN'}
    data["PIPER GREGORY F"]={'salary': 197091, 'to_messages': 1238, 'deferral_payments': 1130036, 'total_payments': 1737629, 'loan_advances': 'NaN', 'bonus': 400000, 'email_address': 'greg.piper@enron.com', 'restricted_stock_deferred': -409554, 'deferred_income': -33333, 'total_stock_value': 880290, 'expenses': 43057, 'from_poi_to_this_person': 61, 'exercised_stock_options': 880290, 'from_messages': 222, 'other': 778, 'from_this_person_to_poi': 48, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 742, 'restricted_stock': 409554, 'director_fees': 'NaN'}
    data["HUMPHREY GENE E"]={'salary': 130724, 'to_messages': 128, 'deferral_payments': 2964506, 'total_payments': 3100224, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'gene.humphrey@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2282768, 'expenses': 4994, 'from_poi_to_this_person': 10, 'exercised_stock_options': 2282768, 'from_messages': 17, 'other': 'NaN', 'from_this_person_to_poi': 17, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 119, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["NOLES JAMES L"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 774401, 'total_payments': 774401, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -94556, 'deferred_income': 'NaN', 'total_stock_value': 368705, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 463261, 'director_fees': 'NaN'}
    data["BLACHMAN JEREMY M"]={'salary': 248546, 'to_messages': 2475, 'deferral_payments': 'NaN', 'total_payments': 2014835, 'loan_advances': 'NaN', 'bonus': 850000, 'email_address': 'jeremy.blachman@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 954354, 'expenses': 84208, 'from_poi_to_this_person': 25, 'exercised_stock_options': 765313, 'from_messages': 14, 'other': 272, 'from_this_person_to_poi': 2, 'poi': False, 'long_term_incentive': 831809, 'shared_receipt_with_poi': 2326, 'restricted_stock': 189041, 'director_fees': 'NaN'}
    data["SUNDE MARTIN"]={'salary': 257486, 'to_messages': 2647, 'deferral_payments': 'NaN', 'total_payments': 1545059, 'loan_advances': 'NaN', 'bonus': 700000, 'email_address': 'marty.sunde@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 698920, 'expenses': 'NaN', 'from_poi_to_this_person': 37, 'exercised_stock_options': 'NaN', 'from_messages': 38, 'other': 111122, 'from_this_person_to_poi': 13, 'poi': False, 'long_term_incentive': 476451, 'shared_receipt_with_poi': 2565, 'restricted_stock': 698920, 'director_fees': 'NaN'}
    data["GIBBS DANA R"]={'salary': 'NaN', 'to_messages': 169, 'deferral_payments': 504610, 'total_payments': 966522, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'dana.gibbs@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2218275, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 2218275, 'from_messages': 12, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 461912, 'shared_receipt_with_poi': 23, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["LOWRY CHARLES P"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -153686, 'deferred_income': 'NaN', 'total_stock_value': 372205, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 372205, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 153686, 'director_fees': 'NaN'}
    data["COLWELL WESLEY"]={'salary': 288542, 'to_messages': 1758, 'deferral_payments': 27610, 'total_payments': 1490344, 'loan_advances': 'NaN', 'bonus': 1200000, 'email_address': 'wes.colwell@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -144062, 'total_stock_value': 698242, 'expenses': 16514, 'from_poi_to_this_person': 240, 'exercised_stock_options': 'NaN', 'from_messages': 40, 'other': 101740, 'from_this_person_to_poi': 11, 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 1132, 'restricted_stock': 698242, 'director_fees': 'NaN'}
    data["MULLER MARK S"]={'salary': 251654, 'to_messages': 136, 'deferral_payments': 842924, 'total_payments': 3202070, 'loan_advances': 'NaN', 'bonus': 1100000, 'email_address': 's..muller@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -719000, 'total_stock_value': 1416848, 'expenses': 'NaN', 'from_poi_to_this_person': 12, 'exercised_stock_options': 1056320, 'from_messages': 16, 'other': 947, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 1725545, 'shared_receipt_with_poi': 114, 'restricted_stock': 360528, 'director_fees': 'NaN'}
    data["JACKSON CHARLENE R"]={'salary': 288558, 'to_messages': 258, 'deferral_payments': 'NaN', 'total_payments': 551174, 'loan_advances': 'NaN', 'bonus': 250000, 'email_address': 'charlene.jackson@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 725735, 'expenses': 10181, 'from_poi_to_this_person': 25, 'exercised_stock_options': 185063, 'from_messages': 56, 'other': 2435, 'from_this_person_to_poi': 19, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 117, 'restricted_stock': 540672, 'director_fees': 'NaN'}
    data["WESTFAHL RICHARD K"]={'salary': 63744, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 762135, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'dick.westfahl@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -10800, 'total_stock_value': 384930, 'expenses': 51870, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 401130, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 256191, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 384930, 'director_fees': 'NaN'}
    data["WALTERS GARETH W"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 53625, 'total_payments': 87410, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1030329, 'expenses': 33785, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1030329, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["WALLS JR ROBERT H"]={'salary': 357091, 'to_messages': 671, 'deferral_payments': 'NaN', 'total_payments': 1798780, 'loan_advances': 'NaN', 'bonus': 850000, 'email_address': 'rob.walls@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 5898997, 'expenses': 50936, 'from_poi_to_this_person': 17, 'exercised_stock_options': 4346544, 'from_messages': 146, 'other': 2, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 540751, 'shared_receipt_with_poi': 215, 'restricted_stock': 1552453, 'director_fees': 'NaN'}
    data["KITCHEN LOUISE"]={'salary': 271442, 'to_messages': 8305, 'deferral_payments': 'NaN', 'total_payments': 3471141, 'loan_advances': 'NaN', 'bonus': 3100000, 'email_address': 'louise.kitchen@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 547143, 'expenses': 5774, 'from_poi_to_this_person': 251, 'exercised_stock_options': 81042, 'from_messages': 1728, 'other': 93925, 'from_this_person_to_poi': 194, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 3669, 'restricted_stock': 466101, 'director_fees': 'NaN'}
    data["CHAN RONNIE"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -32460, 'deferred_income': -98784, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 32460, 'director_fees': 98784}
    data["BELFER ROBERT"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': -102500, 'total_payments': 102500, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 44093, 'deferred_income': 'NaN', 'total_stock_value': -44093, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 3285, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 3285}
    data["SHANKMAN JEFFREY A"]={'salary': 304110, 'to_messages': 3221, 'deferral_payments': 'NaN', 'total_payments': 3038702, 'loan_advances': 'NaN', 'bonus': 2000000, 'email_address': 'jeffrey.shankman@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2072035, 'expenses': 178979, 'from_poi_to_this_person': 94, 'exercised_stock_options': 1441898, 'from_messages': 2681, 'other': 1191, 'from_this_person_to_poi': 83, 'poi': False, 'long_term_incentive': 554422, 'shared_receipt_with_poi': 1730, 'restricted_stock': 630137, 'director_fees': 'NaN'}
    data["WODRASKA JOHN"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 189583, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'john.wodraska@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 189583, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["BERGSIEKER RICHARD P"]={'salary': 187922, 'to_messages': 383, 'deferral_payments': 'NaN', 'total_payments': 618850, 'loan_advances': 'NaN', 'bonus': 250000, 'email_address': 'rick.bergsieker@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -485813, 'total_stock_value': 659249, 'expenses': 59175, 'from_poi_to_this_person': 4, 'exercised_stock_options': 'NaN', 'from_messages': 59, 'other': 427316, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 180250, 'shared_receipt_with_poi': 233, 'restricted_stock': 659249, 'director_fees': 'NaN'}
    data["URQUHART JOHN A"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 228656, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -36666, 'total_stock_value': 'NaN', 'expenses': 228656, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 36666}
    data["BIBI PHILIPPE A"]={'salary': 213625, 'to_messages': 1607, 'deferral_payments': 'NaN', 'total_payments': 2047593, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'philippe.bibi@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1843816, 'expenses': 38559, 'from_poi_to_this_person': 23, 'exercised_stock_options': 1465734, 'from_messages': 40, 'other': 425688, 'from_this_person_to_poi': 8, 'poi': False, 'long_term_incentive': 369721, 'shared_receipt_with_poi': 1336, 'restricted_stock': 378082, 'director_fees': 'NaN'}
    data["RIEKER PAULA H"]={'salary': 249201, 'to_messages': 1328, 'deferral_payments': 214678, 'total_payments': 1099100, 'loan_advances': 'NaN', 'bonus': 700000, 'email_address': 'paula.rieker@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -100000, 'total_stock_value': 1918887, 'expenses': 33271, 'from_poi_to_this_person': 35, 'exercised_stock_options': 1635238, 'from_messages': 82, 'other': 1950, 'from_this_person_to_poi': 48, 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 1258, 'restricted_stock': 283649, 'director_fees': 'NaN'}
    data["WHALEY DAVID A"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 98718, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 98718, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["BECK SALLY W"]={'salary': 231330, 'to_messages': 7315, 'deferral_payments': 'NaN', 'total_payments': 969068, 'loan_advances': 'NaN', 'bonus': 700000, 'email_address': 'sally.beck@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 126027, 'expenses': 37172, 'from_poi_to_this_person': 144, 'exercised_stock_options': 'NaN', 'from_messages': 4343, 'other': 566, 'from_this_person_to_poi': 386, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 2639, 'restricted_stock': 126027, 'director_fees': 'NaN'}
    data["HAUG DAVID L"]={'salary': 'NaN', 'to_messages': 573, 'deferral_payments': 'NaN', 'total_payments': 475, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'david.haug@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2217299, 'expenses': 475, 'from_poi_to_this_person': 4, 'exercised_stock_options': 'NaN', 'from_messages': 19, 'other': 'NaN', 'from_this_person_to_poi': 7, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 471, 'restricted_stock': 2217299, 'director_fees': 'NaN'}
    data["ECHOLS JOHN B"]={'salary': 182245, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2692324, 'loan_advances': 'NaN', 'bonus': 200000, 'email_address': 'john.echols@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1008941, 'expenses': 21530, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 601438, 'from_messages': 'NaN', 'other': 53775, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 2234774, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 407503, 'director_fees': 'NaN'}
    data["MENDELSOHN JOHN"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 148, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -103750, 'total_stock_value': 'NaN', 'expenses': 148, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 103750}
    data["HICKERSON GARY J"]={'salary': 211788, 'to_messages': 1320, 'deferral_payments': 'NaN', 'total_payments': 2081796, 'loan_advances': 'NaN', 'bonus': 1700000, 'email_address': 'gary.hickerson@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 441096, 'expenses': 98849, 'from_poi_to_this_person': 40, 'exercised_stock_options': 'NaN', 'from_messages': 27, 'other': 1936, 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 69223, 'shared_receipt_with_poi': 900, 'restricted_stock': 441096, 'director_fees': 'NaN'}
    data["CLINE KENNETH W"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -472568, 'deferred_income': 'NaN', 'total_stock_value': 189518, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 662086, 'director_fees': 'NaN'}
    data["LEWIS RICHARD"]={'salary': 'NaN', 'to_messages': 952, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'richard.lewis@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 850477, 'expenses': 'NaN', 'from_poi_to_this_person': 10, 'exercised_stock_options': 850477, 'from_messages': 26, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 739, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["HAYES ROBERT E"]={'salary': 'NaN', 'to_messages': 504, 'deferral_payments': 7961, 'total_payments': 7961, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'robert.hayes@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 151418, 'expenses': 'NaN', 'from_poi_to_this_person': 16, 'exercised_stock_options': 'NaN', 'from_messages': 12, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 50, 'restricted_stock': 151418, 'director_fees': 'NaN'}
    data["KOPPER MICHAEL J"]={'salary': 224305, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2652612, 'loan_advances': 'NaN', 'bonus': 800000, 'email_address': 'michael.kopper@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 985032, 'expenses': 118134, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 907502, 'from_this_person_to_poi': 'NaN', 'poi': True, 'long_term_incentive': 602671, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 985032, 'director_fees': 'NaN'}
    data["LEFF DANIEL P"]={'salary': 273746, 'to_messages': 2822, 'deferral_payments': 'NaN', 'total_payments': 2664228, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'dan.leff@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 360528, 'expenses': 'NaN', 'from_poi_to_this_person': 67, 'exercised_stock_options': 'NaN', 'from_messages': 63, 'other': 3083, 'from_this_person_to_poi': 14, 'poi': False, 'long_term_incentive': 1387399, 'shared_receipt_with_poi': 2672, 'restricted_stock': 360528, 'director_fees': 'NaN'}
    data["LAVORATO JOHN J"]={'salary': 339288, 'to_messages': 7259, 'deferral_payments': 'NaN', 'total_payments': 10425757, 'loan_advances': 'NaN', 'bonus': 8000000, 'email_address': 'john.lavorato@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 5167144, 'expenses': 49537, 'from_poi_to_this_person': 528, 'exercised_stock_options': 4158995, 'from_messages': 2585, 'other': 1552, 'from_this_person_to_poi': 411, 'poi': False, 'long_term_incentive': 2035380, 'shared_receipt_with_poi': 3962, 'restricted_stock': 1008149, 'director_fees': 'NaN'}
    data["BERBERIAN DAVID"]={'salary': 216582, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 228474, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'david.berberian@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2493616, 'expenses': 11892, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1624396, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 869220, 'director_fees': 'NaN'}
    data["DETMERING TIMOTHY J"]={'salary': 210500, 'to_messages': 'NaN', 'deferral_payments': 875307, 'total_payments': 1204583, 'loan_advances': 'NaN', 'bonus': 425000, 'email_address': 'timothy.detmering@enron.com', 'restricted_stock_deferred': -315068, 'deferred_income': -775241, 'total_stock_value': 2027865, 'expenses': 52255, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 2027865, 'from_messages': 'NaN', 'other': 1105, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 415657, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 315068, 'director_fees': 'NaN'}
    data["WAKEHAM JOHN"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 213071, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 103773, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 109298}
    data["POWERS WILLIAM"]={'salary': 'NaN', 'to_messages': 653, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'ken.powers@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -17500, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 'NaN', 'from_messages': 26, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 12, 'restricted_stock': 'NaN', 'director_fees': 17500}
    data["GOLD JOSEPH"]={'salary': 272880, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2146973, 'loan_advances': 'NaN', 'bonus': 750000, 'email_address': 'joe.gold@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 877611, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 436515, 'from_messages': 'NaN', 'other': 819288, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 304805, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 441096, 'director_fees': 'NaN'}
    data["BANNANTINE JAMES M"]={'salary': 477, 'to_messages': 566, 'deferral_payments': 'NaN', 'total_payments': 916197, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'james.bannantine@enron.com', 'restricted_stock_deferred': -560222, 'deferred_income': -5104, 'total_stock_value': 5243487, 'expenses': 56301, 'from_poi_to_this_person': 39, 'exercised_stock_options': 4046157, 'from_messages': 29, 'other': 864523, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 465, 'restricted_stock': 1757552, 'director_fees': 'NaN'}
    data["DUNCAN JOHN H"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 77492, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -25000, 'total_stock_value': 371750, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 371750, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 102492}
    data["SHAPIRO RICHARD S"]={'salary': 269076, 'to_messages': 15149, 'deferral_payments': 'NaN', 'total_payments': 1057548, 'loan_advances': 'NaN', 'bonus': 650000, 'email_address': 'richard.shapiro@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 987001, 'expenses': 137767, 'from_poi_to_this_person': 74, 'exercised_stock_options': 607837, 'from_messages': 1215, 'other': 705, 'from_this_person_to_poi': 65, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 4527, 'restricted_stock': 379164, 'director_fees': 'NaN'}
    data["SHERRIFF JOHN R"]={'salary': 428780, 'to_messages': 3187, 'deferral_payments': 'NaN', 'total_payments': 4335388, 'loan_advances': 'NaN', 'bonus': 1500000, 'email_address': 'john.sherriff@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 3128982, 'expenses': 'NaN', 'from_poi_to_this_person': 28, 'exercised_stock_options': 1835558, 'from_messages': 92, 'other': 1852186, 'from_this_person_to_poi': 23, 'poi': False, 'long_term_incentive': 554422, 'shared_receipt_with_poi': 2103, 'restricted_stock': 1293424, 'director_fees': 'NaN'}
    data["SHELBY REX"]={'salary': 211844, 'to_messages': 225, 'deferral_payments': 'NaN', 'total_payments': 2003885, 'loan_advances': 'NaN', 'bonus': 200000, 'email_address': 'rex.shelby@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -4167, 'total_stock_value': 2493616, 'expenses': 22884, 'from_poi_to_this_person': 13, 'exercised_stock_options': 1624396, 'from_messages': 39, 'other': 1573324, 'from_this_person_to_poi': 14, 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 91, 'restricted_stock': 869220, 'director_fees': 'NaN'}
    data["LEMAISTRE CHARLES"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 87492, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -25000, 'total_stock_value': 412878, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 412878, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 112492}
    data["DEFFNER JOSEPH M"]={'salary': 206121, 'to_messages': 714, 'deferral_payments': 'NaN', 'total_payments': 1208649, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'joseph.deffner@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 159211, 'expenses': 41626, 'from_poi_to_this_person': 115, 'exercised_stock_options': 17378, 'from_messages': 74, 'other': 25553, 'from_this_person_to_poi': 4, 'poi': False, 'long_term_incentive': 335349, 'shared_receipt_with_poi': 552, 'restricted_stock': 141833, 'director_fees': 'NaN'}
    data["KISHKILL JOSEPH G"]={'salary': 174246, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 704896, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'joe.kishkill@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -51042, 'total_stock_value': 1034346, 'expenses': 116335, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 465357, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 1034346, 'director_fees': 'NaN'}
    data["WHALLEY LAWRENCE G"]={'salary': 510364, 'to_messages': 6019, 'deferral_payments': 'NaN', 'total_payments': 4677574, 'loan_advances': 'NaN', 'bonus': 3000000, 'email_address': 'greg.whalley@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 6079137, 'expenses': 57838, 'from_poi_to_this_person': 186, 'exercised_stock_options': 3282960, 'from_messages': 556, 'other': 301026, 'from_this_person_to_poi': 24, 'poi': False, 'long_term_incentive': 808346, 'shared_receipt_with_poi': 3920, 'restricted_stock': 2796177, 'director_fees': 'NaN'}
    data["MCCONNELL MICHAEL S"]={'salary': 365038, 'to_messages': 3329, 'deferral_payments': 'NaN', 'total_payments': 2101364, 'loan_advances': 'NaN', 'bonus': 1100000, 'email_address': 'mike.mcconnell@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 3101279, 'expenses': 81364, 'from_poi_to_this_person': 92, 'exercised_stock_options': 1623010, 'from_messages': 2742, 'other': 540, 'from_this_person_to_poi': 194, 'poi': False, 'long_term_incentive': 554422, 'shared_receipt_with_poi': 2189, 'restricted_stock': 1478269, 'director_fees': 'NaN'}
    data["PIRO JIM"]={'salary': 'NaN', 'to_messages': 58, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'jim.piro@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 47304, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 'NaN', 'from_messages': 16, 'other': 'NaN', 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 3, 'restricted_stock': 47304, 'director_fees': 'NaN'}
    data["DELAINEY DAVID W"]={'salary': 365163, 'to_messages': 3093, 'deferral_payments': 'NaN', 'total_payments': 4747979, 'loan_advances': 'NaN', 'bonus': 3000000, 'email_address': 'david.delainey@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 3614261, 'expenses': 86174, 'from_poi_to_this_person': 66, 'exercised_stock_options': 2291113, 'from_messages': 3069, 'other': 1661, 'from_this_person_to_poi': 609, 'poi': True, 'long_term_incentive': 1294981, 'shared_receipt_with_poi': 2097, 'restricted_stock': 1323148, 'director_fees': 'NaN'}
    data["SULLIVAN-SHAKLOVITZ COLLEEN"]={'salary': 162779, 'to_messages': 'NaN', 'deferral_payments': 181993, 'total_payments': 999356, 'loan_advances': 'NaN', 'bonus': 100000, 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1362375, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1362375, 'from_messages': 'NaN', 'other': 162, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 554422, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["WROBEL BRUCE"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 139130, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 139130, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["LINDHOLM TOD A"]={'salary': 236457, 'to_messages': 'NaN', 'deferral_payments': 204075, 'total_payments': 875889, 'loan_advances': 'NaN', 'bonus': 200000, 'email_address': 'tod.lindholm@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 3064208, 'expenses': 57727, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 2549361, 'from_messages': 'NaN', 'other': 2630, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 175000, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 514847, 'director_fees': 'NaN'}
    data["MEYER JEROME J"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2151, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -38346, 'total_stock_value': 'NaN', 'expenses': 2151, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 38346}
    data["LAY KENNETH L"]={'salary': 1072321, 'to_messages': 4273, 'deferral_payments': 202911, 'total_payments': 103559793, 'loan_advances': 81525000, 'bonus': 7000000, 'email_address': 'kenneth.lay@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -300000, 'total_stock_value': 49110078, 'expenses': 99832, 'from_poi_to_this_person': 123, 'exercised_stock_options': 34348384, 'from_messages': 36, 'other': 10359729, 'from_this_person_to_poi': 16, 'poi': True, 'long_term_incentive': 3600000, 'shared_receipt_with_poi': 2411, 'restricted_stock': 14761694, 'director_fees': 'NaN'}
    data["BUTTS ROBERT H"]={'salary': 261516, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 1271582, 'loan_advances': 'NaN', 'bonus': 750000, 'email_address': 'bob.butts@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -75000, 'total_stock_value': 417619, 'expenses': 9410, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 150656, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 175000, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 417619, 'director_fees': 'NaN'}
    data["OLSON CINDY K"]={'salary': 329078, 'to_messages': 1184, 'deferral_payments': 77716, 'total_payments': 1321557, 'loan_advances': 'NaN', 'bonus': 750000, 'email_address': 'cindy.olson@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2606763, 'expenses': 63791, 'from_poi_to_this_person': 20, 'exercised_stock_options': 1637034, 'from_messages': 52, 'other': 972, 'from_this_person_to_poi': 15, 'poi': False, 'long_term_incentive': 100000, 'shared_receipt_with_poi': 856, 'restricted_stock': 969729, 'director_fees': 'NaN'}
    data["MCDONALD REBECCA"]={'salary': 'NaN', 'to_messages': 894, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'rebecca.mcdonald@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1691366, 'expenses': 'NaN', 'from_poi_to_this_person': 54, 'exercised_stock_options': 757301, 'from_messages': 13, 'other': 'NaN', 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 720, 'restricted_stock': 934065, 'director_fees': 'NaN'}
    data["CUMBERLAND MICHAEL S"]={'salary': 184899, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 807956, 'loan_advances': 'NaN', 'bonus': 325000, 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 207940, 'expenses': 22344, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 713, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 275000, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 207940, 'director_fees': 'NaN'}
    data["GAHN ROBERT S"]={'salary': 192008, 'to_messages': 'NaN', 'deferral_payments': 73122, 'total_payments': 900585, 'loan_advances': 'NaN', 'bonus': 509870, 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -1042, 'total_stock_value': 318607, 'expenses': 50080, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 83237, 'from_messages': 'NaN', 'other': 76547, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 235370, 'director_fees': 'NaN'}
    data["BADUM JAMES P"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 178980, 'total_payments': 182466, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 257817, 'expenses': 3486, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 257817, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["HERMANN ROBERT J"]={'salary': 262663, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 1297461, 'loan_advances': 'NaN', 'bonus': 700000, 'email_address': 'robert.hermann@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -280000, 'total_stock_value': 668132, 'expenses': 48357, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 187500, 'from_messages': 'NaN', 'other': 416441, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 150000, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 480632, 'director_fees': 'NaN'}
    data["FALLON JAMES B"]={'salary': 304588, 'to_messages': 1755, 'deferral_payments': 'NaN', 'total_payments': 3676340, 'loan_advances': 'NaN', 'bonus': 2500000, 'email_address': 'jim.fallon@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2332399, 'expenses': 95924, 'from_poi_to_this_person': 42, 'exercised_stock_options': 940257, 'from_messages': 75, 'other': 401481, 'from_this_person_to_poi': 37, 'poi': False, 'long_term_incentive': 374347, 'shared_receipt_with_poi': 1604, 'restricted_stock': 1392142, 'director_fees': 'NaN'}
    data["GATHMANN WILLIAM D"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -72419, 'deferred_income': 'NaN', 'total_stock_value': 1945360, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1753766, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 264013, 'director_fees': 'NaN'}
    data["HORTON STANLEY C"]={'salary': 'NaN', 'to_messages': 2350, 'deferral_payments': 3131860, 'total_payments': 3131860, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'stanley.horton@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 7256648, 'expenses': 'NaN', 'from_poi_to_this_person': 44, 'exercised_stock_options': 5210569, 'from_messages': 1073, 'other': 'NaN', 'from_this_person_to_poi': 15, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 1074, 'restricted_stock': 2046079, 'director_fees': 'NaN'}
    data["BOWEN JR RAYMOND M"]={'salary': 278601, 'to_messages': 1858, 'deferral_payments': 'NaN', 'total_payments': 2669589, 'loan_advances': 'NaN', 'bonus': 1350000, 'email_address': 'raymond.bowen@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -833, 'total_stock_value': 252055, 'expenses': 65907, 'from_poi_to_this_person': 140, 'exercised_stock_options': 'NaN', 'from_messages': 27, 'other': 1621, 'from_this_person_to_poi': 15, 'poi': True, 'long_term_incentive': 974293, 'shared_receipt_with_poi': 1593, 'restricted_stock': 252055, 'director_fees': 'NaN'}
    data["GILLIS JOHN"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 85641, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 9803, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 75838, 'director_fees': 'NaN'}
    data["FITZGERALD JAY L"]={'salary': 199157, 'to_messages': 936, 'deferral_payments': 'NaN', 'total_payments': 1414857, 'loan_advances': 'NaN', 'bonus': 350000, 'email_address': 'jay.fitzgerald@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1621236, 'expenses': 23870, 'from_poi_to_this_person': 1, 'exercised_stock_options': 664461, 'from_messages': 16, 'other': 285414, 'from_this_person_to_poi': 8, 'poi': False, 'long_term_incentive': 556416, 'shared_receipt_with_poi': 723, 'restricted_stock': 956775, 'director_fees': 'NaN'}
    data["MORAN MICHAEL P"]={'salary': 'NaN', 'to_messages': 672, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'michael.moran@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 221141, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 59539, 'from_messages': 19, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 127, 'restricted_stock': 161602, 'director_fees': 'NaN'}
    data["REDMOND BRIAN L"]={'salary': 96840, 'to_messages': 1671, 'deferral_payments': 'NaN', 'total_payments': 111529, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'brian.redmond@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 7890324, 'expenses': 14689, 'from_poi_to_this_person': 204, 'exercised_stock_options': 7509039, 'from_messages': 221, 'other': 'NaN', 'from_this_person_to_poi': 49, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 1063, 'restricted_stock': 381285, 'director_fees': 'NaN'}
    data["BAZELIDES PHILIP J"]={'salary': 80818, 'to_messages': 'NaN', 'deferral_payments': 684694, 'total_payments': 860136, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1599641, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1599641, 'from_messages': 'NaN', 'other': 874, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 93750, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["BELDEN TIMOTHY N"]={'salary': 213999, 'to_messages': 7991, 'deferral_payments': 2144013, 'total_payments': 5501630, 'loan_advances': 'NaN', 'bonus': 5249999, 'email_address': 'tim.belden@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -2334434, 'total_stock_value': 1110705, 'expenses': 17355, 'from_poi_to_this_person': 228, 'exercised_stock_options': 953136, 'from_messages': 484, 'other': 210698, 'from_this_person_to_poi': 108, 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 5521, 'restricted_stock': 157569, 'director_fees': 'NaN'}
    data["DIMICHELE RICHARD G"]={'salary': 262788, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2368151, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'richard.dimichele@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 8317782, 'expenses': 35812, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 8191755, 'from_messages': 'NaN', 'other': 374689, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 694862, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 126027, 'director_fees': 'NaN'}
    data["DURAN WILLIAM D"]={'salary': 210692, 'to_messages': 904, 'deferral_payments': 'NaN', 'total_payments': 2093263, 'loan_advances': 'NaN', 'bonus': 750000, 'email_address': 'w.duran@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1640910, 'expenses': 25785, 'from_poi_to_this_person': 106, 'exercised_stock_options': 1451869, 'from_messages': 12, 'other': 1568, 'from_this_person_to_poi': 3, 'poi': False, 'long_term_incentive': 1105218, 'shared_receipt_with_poi': 599, 'restricted_stock': 189041, 'director_fees': 'NaN'}
    data["THORN TERENCE H"]={'salary': 222093, 'to_messages': 266, 'deferral_payments': 16586, 'total_payments': 911453, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'terence.thorn@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 4817796, 'expenses': 46145, 'from_poi_to_this_person': 0, 'exercised_stock_options': 4452476, 'from_messages': 41, 'other': 426629, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 200000, 'shared_receipt_with_poi': 73, 'restricted_stock': 365320, 'director_fees': 'NaN'}
    data["FASTOW ANDREW S"]={'salary': 440698, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 2424083, 'loan_advances': 'NaN', 'bonus': 1300000, 'email_address': 'andrew.fastow@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -1386055, 'total_stock_value': 1794412, 'expenses': 55921, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 277464, 'from_this_person_to_poi': 'NaN', 'poi': True, 'long_term_incentive': 1736055, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 1794412, 'director_fees': 'NaN'}
    data["FOY JOE"]={'salary': 'NaN', 'to_messages': 57, 'deferral_payments': 181755, 'total_payments': 181755, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'tracy.foy@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 343434, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 343434, 'from_messages': 13, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 2, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["CALGER CHRISTOPHER F"]={'salary': 240189, 'to_messages': 2598, 'deferral_payments': 'NaN', 'total_payments': 1639297, 'loan_advances': 'NaN', 'bonus': 1250000, 'email_address': 'christopher.calger@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -262500, 'total_stock_value': 126027, 'expenses': 35818, 'from_poi_to_this_person': 199, 'exercised_stock_options': 'NaN', 'from_messages': 144, 'other': 486, 'from_this_person_to_poi': 25, 'poi': True, 'long_term_incentive': 375304, 'shared_receipt_with_poi': 2188, 'restricted_stock': 126027, 'director_fees': 'NaN'}
    data["RICE KENNETH D"]={'salary': 420636, 'to_messages': 905, 'deferral_payments': 'NaN', 'total_payments': 505050, 'loan_advances': 'NaN', 'bonus': 1750000, 'email_address': 'ken.rice@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -3504386, 'total_stock_value': 22542539, 'expenses': 46950, 'from_poi_to_this_person': 42, 'exercised_stock_options': 19794175, 'from_messages': 18, 'other': 174839, 'from_this_person_to_poi': 4, 'poi': True, 'long_term_incentive': 1617011, 'shared_receipt_with_poi': 864, 'restricted_stock': 2748364, 'director_fees': 'NaN'}
    data["KAMINSKI WINCENTY J"]={'salary': 275101, 'to_messages': 4607, 'deferral_payments': 'NaN', 'total_payments': 1086821, 'loan_advances': 'NaN', 'bonus': 400000, 'email_address': 'vince.kaminski@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 976037, 'expenses': 83585, 'from_poi_to_this_person': 41, 'exercised_stock_options': 850010, 'from_messages': 14368, 'other': 4669, 'from_this_person_to_poi': 171, 'poi': False, 'long_term_incentive': 323466, 'shared_receipt_with_poi': 583, 'restricted_stock': 126027, 'director_fees': 'NaN'}
    data["LOCKHART EUGENE E"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["COX DAVID"]={'salary': 314288, 'to_messages': 102, 'deferral_payments': 'NaN', 'total_payments': 1101393, 'loan_advances': 'NaN', 'bonus': 800000, 'email_address': 'chip.cox@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -41250, 'total_stock_value': 495633, 'expenses': 27861, 'from_poi_to_this_person': 0, 'exercised_stock_options': 117551, 'from_messages': 33, 'other': 494, 'from_this_person_to_poi': 4, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 71, 'restricted_stock': 378082, 'director_fees': 'NaN'}
    data["OVERDYKE JR JERE C"]={'salary': 94941, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 249787, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'jere.overdyke@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 7307594, 'expenses': 18834, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 5266578, 'from_messages': 'NaN', 'other': 176, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 135836, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 2041016, 'director_fees': 'NaN'}
    data["PEREIRA PAULO V. FERRAZ"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 27942, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -101250, 'total_stock_value': 'NaN', 'expenses': 27942, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 101250}
    data["STABLER FRANK"]={'salary': 239502, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 1112087, 'loan_advances': 'NaN', 'bonus': 500000, 'email_address': 'frank.stabler@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 511734, 'expenses': 16514, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 356071, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 511734, 'director_fees': 'NaN'}
    data["SKILLING JEFFREY K"]={'salary': 1111258, 'to_messages': 3627, 'deferral_payments': 'NaN', 'total_payments': 8682716, 'loan_advances': 'NaN', 'bonus': 5600000, 'email_address': 'jeff.skilling@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 26093672, 'expenses': 29336, 'from_poi_to_this_person': 88, 'exercised_stock_options': 19250000, 'from_messages': 108, 'other': 22122, 'from_this_person_to_poi': 30, 'poi': True, 'long_term_incentive': 1920000, 'shared_receipt_with_poi': 2042, 'restricted_stock': 6843672, 'director_fees': 'NaN'}
    data["BLAKE JR. NORMAN P"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 1279, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -113784, 'total_stock_value': 'NaN', 'expenses': 1279, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 113784}
    data["SHERRICK JEFFREY B"]={'salary': 'NaN', 'to_messages': 613, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'jeffrey.sherrick@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1832468, 'expenses': 'NaN', 'from_poi_to_this_person': 39, 'exercised_stock_options': 1426469, 'from_messages': 25, 'other': 'NaN', 'from_this_person_to_poi': 18, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 583, 'restricted_stock': 405999, 'director_fees': 'NaN'}
    data["PRENTICE JAMES"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 564348, 'total_payments': 564348, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'james.prentice@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1095040, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 886231, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 208809, 'director_fees': 'NaN'}
    data["GRAY RODNEY"]={'salary': 6615, 'to_messages': 'NaN', 'deferral_payments': 93585, 'total_payments': 1146658, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 680833, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 365625, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["THE TRAVEL AGENCY IN THE PARK"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 362096, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 362096, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["UMANOFF ADAM S"]={'salary': 288589, 'to_messages': 111, 'deferral_payments': 'NaN', 'total_payments': 1130461, 'loan_advances': 'NaN', 'bonus': 788750, 'email_address': 'adam.umanoff@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 53122, 'from_poi_to_this_person': 12, 'exercised_stock_options': 'NaN', 'from_messages': 18, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 41, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["KEAN STEVEN J"]={'salary': 404338, 'to_messages': 12754, 'deferral_payments': 'NaN', 'total_payments': 1747522, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'steven.kean@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 6153642, 'expenses': 41953, 'from_poi_to_this_person': 140, 'exercised_stock_options': 2022048, 'from_messages': 6759, 'other': 1231, 'from_this_person_to_poi': 387, 'poi': False, 'long_term_incentive': 300000, 'shared_receipt_with_poi': 3639, 'restricted_stock': 4131594, 'director_fees': 'NaN'}
    data["TOTAL"]={'salary': 26704229, 'to_messages': 'NaN', 'deferral_payments': 32083396, 'total_payments': 309886585, 'loan_advances': 83925000, 'bonus': 97343619, 'email_address': 'NaN', 'restricted_stock_deferred': -7576788, 'deferred_income': -27992891, 'total_stock_value': 434509511, 'expenses': 5235198, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 311764000, 'from_messages': 'NaN', 'other': 42667589, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 48521928, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 130322299, 'director_fees': 1398517}
    data["FOWLER PEGGY"]={'salary': 'NaN', 'to_messages': 517, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'kulvinder.fowler@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1884748, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 1324578, 'from_messages': 36, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 10, 'restricted_stock': 560170, 'director_fees': 'NaN'}
    data["WASAFF GEORGE"]={'salary': 259996, 'to_messages': 400, 'deferral_payments': 831299, 'total_payments': 1034395, 'loan_advances': 'NaN', 'bonus': 325000, 'email_address': 'george.wasaff@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -583325, 'total_stock_value': 2056427, 'expenses': 'NaN', 'from_poi_to_this_person': 22, 'exercised_stock_options': 1668260, 'from_messages': 30, 'other': 1425, 'from_this_person_to_poi': 7, 'poi': False, 'long_term_incentive': 200000, 'shared_receipt_with_poi': 337, 'restricted_stock': 388167, 'director_fees': 'NaN'}
    data["WHITE JR THOMAS E"]={'salary': 317543, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 1934359, 'loan_advances': 'NaN', 'bonus': 450000, 'email_address': 'thomas.white@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 15144123, 'expenses': 81353, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 1297049, 'from_messages': 'NaN', 'other': 1085463, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 13847074, 'director_fees': 'NaN'}
    data["CHRISTODOULOU DIOMEDES"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'diomedes.christodoulou@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 6077885, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 5127155, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 950730, 'director_fees': 'NaN'}
    data["ALLEN PHILLIP K"]={'salary': 201955, 'to_messages': 2902, 'deferral_payments': 2869717, 'total_payments': 4484442, 'loan_advances': 'NaN', 'bonus': 4175000, 'email_address': 'phillip.allen@enron.com', 'restricted_stock_deferred': -126027, 'deferred_income': -3081055, 'total_stock_value': 1729541, 'expenses': 13868, 'from_poi_to_this_person': 47, 'exercised_stock_options': 1729541, 'from_messages': 2195, 'other': 152, 'from_this_person_to_poi': 65, 'poi': False, 'long_term_incentive': 304805, 'shared_receipt_with_poi': 1407, 'restricted_stock': 126027, 'director_fees': 'NaN'}
    data["SHARP VICTORIA T"]={'salary': 248146, 'to_messages': 3136, 'deferral_payments': 187469, 'total_payments': 1576511, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'vicki.sharp@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 494136, 'expenses': 116337, 'from_poi_to_this_person': 24, 'exercised_stock_options': 281073, 'from_messages': 136, 'other': 2401, 'from_this_person_to_poi': 6, 'poi': False, 'long_term_incentive': 422158, 'shared_receipt_with_poi': 2477, 'restricted_stock': 213063, 'director_fees': 'NaN'}
    data["JAEDICKE ROBERT"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 83750, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': -44093, 'deferred_income': -25000, 'total_stock_value': 431750, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 431750, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 44093, 'director_fees': 108750}
    data["WINOKUR JR. HERBERT S"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 84992, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -25000, 'total_stock_value': 'NaN', 'expenses': 1413, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 108579}
    data["BROWN MICHAEL"]={'salary': 'NaN', 'to_messages': 1486, 'deferral_payments': 'NaN', 'total_payments': 49288, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'michael.brown@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 49288, 'from_poi_to_this_person': 13, 'exercised_stock_options': 'NaN', 'from_messages': 41, 'other': 'NaN', 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 761, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["MCCLELLAN GEORGE"]={'salary': 263413, 'to_messages': 1744, 'deferral_payments': 'NaN', 'total_payments': 1318763, 'loan_advances': 'NaN', 'bonus': 900000, 'email_address': 'george.mcclellan@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -125000, 'total_stock_value': 947861, 'expenses': 228763, 'from_poi_to_this_person': 52, 'exercised_stock_options': 506765, 'from_messages': 49, 'other': 51587, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 1469, 'restricted_stock': 441096, 'director_fees': 'NaN'}
    data["HUGHES JAMES A"]={'salary': 'NaN', 'to_messages': 719, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'james.hughes@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1118394, 'expenses': 'NaN', 'from_poi_to_this_person': 35, 'exercised_stock_options': 754966, 'from_messages': 34, 'other': 'NaN', 'from_this_person_to_poi': 5, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 589, 'restricted_stock': 363428, 'director_fees': 'NaN'}
    data["REYNOLDS LAWRENCE"]={'salary': 76399, 'to_messages': 'NaN', 'deferral_payments': 51365, 'total_payments': 394475, 'loan_advances': 'NaN', 'bonus': 100000, 'email_address': 'NaN', 'restricted_stock_deferred': -140264, 'deferred_income': -200000, 'total_stock_value': 4221891, 'expenses': 8409, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 4160672, 'from_messages': 'NaN', 'other': 202052, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 156250, 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 201483, 'director_fees': 'NaN'}
    data["PICKERING MARK R"]={'salary': 655037, 'to_messages': 898, 'deferral_payments': 'NaN', 'total_payments': 1386690, 'loan_advances': 400000, 'bonus': 300000, 'email_address': 'mark.pickering@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 28798, 'expenses': 31653, 'from_poi_to_this_person': 7, 'exercised_stock_options': 28798, 'from_messages': 67, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 728, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["BHATNAGAR SANJAY"]={'salary': 'NaN', 'to_messages': 523, 'deferral_payments': 'NaN', 'total_payments': 15456290, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'sanjay.bhatnagar@enron.com', 'restricted_stock_deferred': 15456290, 'deferred_income': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 2604490, 'from_messages': 29, 'other': 137864, 'from_this_person_to_poi': 1, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 463, 'restricted_stock': -2604490, 'director_fees': 137864}
    data["CARTER REBECCA C"]={'salary': 261809, 'to_messages': 312, 'deferral_payments': 'NaN', 'total_payments': 477557, 'loan_advances': 'NaN', 'bonus': 300000, 'email_address': 'rebecca.carter@enron.com', 'restricted_stock_deferred': -307301, 'deferred_income': -159792, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 29, 'exercised_stock_options': 'NaN', 'from_messages': 15, 'other': 540, 'from_this_person_to_poi': 7, 'poi': False, 'long_term_incentive': 75000, 'shared_receipt_with_poi': 196, 'restricted_stock': 307301, 'director_fees': 'NaN'}
    data["BUCHANAN HAROLD G"]={'salary': 248017, 'to_messages': 1088, 'deferral_payments': 'NaN', 'total_payments': 1054637, 'loan_advances': 'NaN', 'bonus': 500000, 'email_address': 'john.buchanan@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1014505, 'expenses': 600, 'from_poi_to_this_person': 0, 'exercised_stock_options': 825464, 'from_messages': 125, 'other': 1215, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 304805, 'shared_receipt_with_poi': 23, 'restricted_stock': 189041, 'director_fees': 'NaN'}
    data["YEAP SOON"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 55097, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 192758, 'expenses': 55097, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 192758, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["MURRAY JULIA H"]={'salary': 229284, 'to_messages': 2192, 'deferral_payments': 'NaN', 'total_payments': 812194, 'loan_advances': 'NaN', 'bonus': 400000, 'email_address': 'julia.murray@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 597461, 'expenses': 57580, 'from_poi_to_this_person': 11, 'exercised_stock_options': 400478, 'from_messages': 45, 'other': 330, 'from_this_person_to_poi': 2, 'poi': False, 'long_term_incentive': 125000, 'shared_receipt_with_poi': 395, 'restricted_stock': 196983, 'director_fees': 'NaN'}
    data["GARLAND C KEVIN"]={'salary': 231946, 'to_messages': 209, 'deferral_payments': 'NaN', 'total_payments': 1566469, 'loan_advances': 'NaN', 'bonus': 850000, 'email_address': 'kevin.garland@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 896153, 'expenses': 48405, 'from_poi_to_this_person': 10, 'exercised_stock_options': 636246, 'from_messages': 44, 'other': 60814, 'from_this_person_to_poi': 27, 'poi': False, 'long_term_incentive': 375304, 'shared_receipt_with_poi': 178, 'restricted_stock': 259907, 'director_fees': 'NaN'}
    data["DODSON KEITH"]={'salary': 221003, 'to_messages': 176, 'deferral_payments': 'NaN', 'total_payments': 319941, 'loan_advances': 'NaN', 'bonus': 70000, 'email_address': 'keith.dodson@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 28164, 'from_poi_to_this_person': 10, 'exercised_stock_options': 'NaN', 'from_messages': 14, 'other': 774, 'from_this_person_to_poi': 3, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 114, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["YEAGER F SCOTT"]={'salary': 158403, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 360300, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'scott.yeager@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 11884758, 'expenses': 53947, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 8308552, 'from_messages': 'NaN', 'other': 147950, 'from_this_person_to_poi': 'NaN', 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 3576206, 'director_fees': 'NaN'}
    data["HIRKO JOSEPH"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 10259, 'total_payments': 91093, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'joe.hirko@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 30766064, 'expenses': 77978, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 30766064, 'from_messages': 'NaN', 'other': 2856, 'from_this_person_to_poi': 'NaN', 'poi': True, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["DIETRICH JANET R"]={'salary': 250100, 'to_messages': 2572, 'deferral_payments': 'NaN', 'total_payments': 1410464, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'janet.dietrich@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1865087, 'expenses': 3475, 'from_poi_to_this_person': 305, 'exercised_stock_options': 1550019, 'from_messages': 63, 'other': 473, 'from_this_person_to_poi': 14, 'poi': False, 'long_term_incentive': 556416, 'shared_receipt_with_poi': 1902, 'restricted_stock': 315068, 'director_fees': 'NaN'}
    data["DERRICK JR. JAMES V"]={'salary': 492375, 'to_messages': 2181, 'deferral_payments': 'NaN', 'total_payments': 550981, 'loan_advances': 'NaN', 'bonus': 800000, 'email_address': 'james.derrick@enron.com', 'restricted_stock_deferred': -1787380, 'deferred_income': -1284000, 'total_stock_value': 8831913, 'expenses': 51124, 'from_poi_to_this_person': 64, 'exercised_stock_options': 8831913, 'from_messages': 909, 'other': 7482, 'from_this_person_to_poi': 20, 'poi': False, 'long_term_incentive': 484000, 'shared_receipt_with_poi': 1401, 'restricted_stock': 1787380, 'director_fees': 'NaN'}
    data["FREVERT MARK A"]={'salary': 1060932, 'to_messages': 3275, 'deferral_payments': 6426990, 'total_payments': 17252530, 'loan_advances': 2000000, 'bonus': 2000000, 'email_address': 'mark.frevert@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -3367011, 'total_stock_value': 14622185, 'expenses': 86987, 'from_poi_to_this_person': 242, 'exercised_stock_options': 10433518, 'from_messages': 21, 'other': 7427621, 'from_this_person_to_poi': 6, 'poi': False, 'long_term_incentive': 1617011, 'shared_receipt_with_poi': 2979, 'restricted_stock': 4188667, 'director_fees': 'NaN'}
    data["PAI LOU L"]={'salary': 261879, 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 3123383, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'lou.pai@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 23817930, 'expenses': 32047, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 15364167, 'from_messages': 'NaN', 'other': 1829457, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 8453763, 'director_fees': 'NaN'}
    data["HAYSLETT RODERICK J"]={'salary': 'NaN', 'to_messages': 2649, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'rod.hayslett@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 346663, 'expenses': 'NaN', 'from_poi_to_this_person': 35, 'exercised_stock_options': 'NaN', 'from_messages': 1061, 'other': 'NaN', 'from_this_person_to_poi': 38, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 571, 'restricted_stock': 346663, 'director_fees': 'NaN'}
    data["BAY FRANKLIN R"]={'salary': 239671, 'to_messages': 'NaN', 'deferral_payments': 260455, 'total_payments': 827696, 'loan_advances': 'NaN', 'bonus': 400000, 'email_address': 'frank.bay@enron.com', 'restricted_stock_deferred': -82782, 'deferred_income': -201641, 'total_stock_value': 63014, 'expenses': 129142, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 69, 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 145796, 'director_fees': 'NaN'}
    data["MCCARTY DANNY J"]={'salary': 'NaN', 'to_messages': 1433, 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'danny.mccarty@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 758931, 'expenses': 'NaN', 'from_poi_to_this_person': 25, 'exercised_stock_options': 664375, 'from_messages': 215, 'other': 'NaN', 'from_this_person_to_poi': 2, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 508, 'restricted_stock': 94556, 'director_fees': 'NaN'}
    data["FUGH JOHN L"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 50591, 'total_payments': 50591, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 176378, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 176378, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["SCRIMSHAW MATTHEW"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 'NaN', 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'matthew.scrimshaw@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 759557, 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 759557, 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["KOENIG MARK E"]={'salary': 309946, 'to_messages': 2374, 'deferral_payments': 'NaN', 'total_payments': 1587421, 'loan_advances': 'NaN', 'bonus': 700000, 'email_address': 'mark.koenig@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 1920055, 'expenses': 127017, 'from_poi_to_this_person': 53, 'exercised_stock_options': 671737, 'from_messages': 61, 'other': 150458, 'from_this_person_to_poi': 15, 'poi': True, 'long_term_incentive': 300000, 'shared_receipt_with_poi': 2271, 'restricted_stock': 1248318, 'director_fees': 'NaN'}
    data["SAVAGE FRANK"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 3750, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': -121284, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 125034}
    data["IZZO LAWRENCE L"]={'salary': 85274, 'to_messages': 496, 'deferral_payments': 'NaN', 'total_payments': 1979596, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'larry.izzo@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 5819980, 'expenses': 28093, 'from_poi_to_this_person': 28, 'exercised_stock_options': 2165172, 'from_messages': 19, 'other': 1553729, 'from_this_person_to_poi': 5, 'poi': False, 'long_term_incentive': 312500, 'shared_receipt_with_poi': 437, 'restricted_stock': 3654808, 'director_fees': 'NaN'}
    data["TILNEY ELIZABETH A"]={'salary': 247338, 'to_messages': 460, 'deferral_payments': 'NaN', 'total_payments': 399393, 'loan_advances': 'NaN', 'bonus': 300000, 'email_address': 'elizabeth.tilney@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -575000, 'total_stock_value': 1168042, 'expenses': 'NaN', 'from_poi_to_this_person': 10, 'exercised_stock_options': 591250, 'from_messages': 19, 'other': 152055, 'from_this_person_to_poi': 11, 'poi': False, 'long_term_incentive': 275000, 'shared_receipt_with_poi': 379, 'restricted_stock': 576792, 'director_fees': 'NaN'}
    data["MARTIN AMANDA K"]={'salary': 349487, 'to_messages': 1522, 'deferral_payments': 85430, 'total_payments': 8407016, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'a..martin@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 2070306, 'expenses': 8211, 'from_poi_to_this_person': 8, 'exercised_stock_options': 2070306, 'from_messages': 230, 'other': 2818454, 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 5145434, 'shared_receipt_with_poi': 477, 'restricted_stock': 'NaN', 'director_fees': 'NaN'}
    data["BUY RICHARD B"]={'salary': 330546, 'to_messages': 3523, 'deferral_payments': 649584, 'total_payments': 2355702, 'loan_advances': 'NaN', 'bonus': 900000, 'email_address': 'rick.buy@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -694862, 'total_stock_value': 3444470, 'expenses': 'NaN', 'from_poi_to_this_person': 156, 'exercised_stock_options': 2542813, 'from_messages': 1053, 'other': 400572, 'from_this_person_to_poi': 71, 'poi': False, 'long_term_incentive': 769862, 'shared_receipt_with_poi': 2333, 'restricted_stock': 901657, 'director_fees': 'NaN'}
    data["GRAMM WENDY L"]={'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 'NaN', 'total_payments': 119292, 'loan_advances': 'NaN', 'bonus': 'NaN', 'email_address': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 'NaN', 'expenses': 'NaN', 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 'NaN', 'restricted_stock': 'NaN', 'director_fees': 119292}
    data["CAUSEY RICHARD A"]={'salary': 415189, 'to_messages': 1892, 'deferral_payments': 'NaN', 'total_payments': 1868758, 'loan_advances': 'NaN', 'bonus': 1000000, 'email_address': 'richard.causey@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -235000, 'total_stock_value': 2502063, 'expenses': 30674, 'from_poi_to_this_person': 58, 'exercised_stock_options': 'NaN', 'from_messages': 49, 'other': 307895, 'from_this_person_to_poi': 12, 'poi': True, 'long_term_incentive': 350000, 'shared_receipt_with_poi': 1585, 'restricted_stock': 2502063, 'director_fees': 'NaN'}
    data["TAYLOR MITCHELL S"]={'salary': 265214, 'to_messages': 533, 'deferral_payments': 227449, 'total_payments': 1092663, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'mitchell.taylor@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 3745048, 'expenses': 'NaN', 'from_poi_to_this_person': 0, 'exercised_stock_options': 3181250, 'from_messages': 29, 'other': 'NaN', 'from_this_person_to_poi': 0, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 300, 'restricted_stock': 563798, 'director_fees': 'NaN'}
    data["DONAHUE JR JEFFREY M"]={'salary': 278601, 'to_messages': 865, 'deferral_payments': 'NaN', 'total_payments': 875760, 'loan_advances': 'NaN', 'bonus': 800000, 'email_address': 'jeff.donahue@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': -300000, 'total_stock_value': 1080988, 'expenses': 96268, 'from_poi_to_this_person': 188, 'exercised_stock_options': 765920, 'from_messages': 22, 'other': 891, 'from_this_person_to_poi': 11, 'poi': False, 'long_term_incentive': 'NaN', 'shared_receipt_with_poi': 772, 'restricted_stock': 315068, 'director_fees': 'NaN'}
    data["GLISAN JR BEN F"]={'salary': 274975, 'to_messages': 873, 'deferral_payments': 'NaN', 'total_payments': 1272284, 'loan_advances': 'NaN', 'bonus': 600000, 'email_address': 'ben.glisan@enron.com', 'restricted_stock_deferred': 'NaN', 'deferred_income': 'NaN', 'total_stock_value': 778546, 'expenses': 125978, 'from_poi_to_this_person': 52, 'exercised_stock_options': 384728, 'from_messages': 16, 'other': 200308, 'from_this_person_to_poi': 6, 'poi': True, 'long_term_incentive': 71023, 'shared_receipt_with_poi': 874, 'restricted_stock': 393818, 'director_fees': 'NaN'}

    return data
```
### studentCode.py
### if all_messages == "NaN":  分母不为NaN即可
### fraction = 0.   iint除int=0  加一个小数点，这也是一种转换为浮点型的方式，作用等于float(0)

```python
import pickle
from get_data import getData

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.

###################################################################################
    if all_messages == "NaN": 
        fraction = 0.  
    else:
        fraction = float(poi_messages)/float(all_messages)

    return fraction
###################################################################################

data_dict = getData() 

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    
    
#####################

def submitDict():
    return submit_dict

```
# 特征缩减
### sklearn 中有两大单变量特征选择工具：SelectPercentile 和 SelectKBest。 两者之间的区别从名字就可以看出：SelectPercentile 选择最强大的 X% 特征（X 是参数），而 SelectKBest 选择 K 个最强大的特征（K 是参数）
### 在email_preprocess.py 中,运用了两处特征缩减:
### 1    max_df将在50%文本中出现的单词删除.
### 2    SelectPercentile
### email_preprocess.py
```python
#!/usr/bin/python

import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif



def preprocess(words_file = "../tools/word_data.pkl", authors_file="../tools/email_authors.pkl"):
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "r")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)



    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=1)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print "no. of Chris training emails:", sum(labels_train)
    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test

```

# 过拟合决策树的准确率
### find_signature.py
```python
#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn import tree
from sklearn.metrics import accuracy_score
    ### create classifier
clf = tree.DecisionTreeClassifier()  

    ### fit the classifier on the training features and labels
clf.fit(features_train, labels_train) 

    ### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)  


print accuracy_score(pred, labels_test)


```

# 识别最强大特征
### 最重要特征的重要性是什么？该特征的数字是多少？题目给出阈值是0.2
### find_signature.py
```python
#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



##############################################################################################
### your code goes here
from sklearn.metrics import accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test, labels_test)
print accuracy_score(pred, labels_test)

key_index = 0

for index, feature in enumerate(clf.feature_importances_):
    if feature > 0.2:
        key_index = index
        print index, feature

# same result code:
#imp = clf.feature_importances_
#print imp.max()
#print imp.argmax()
```

# 使用 TfIdf 获得最重要的单词 
### find_signature.py
```python
#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]





from sklearn.metrics import accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test, labels_test)
print accuracy_score(pred, labels_test)

key_index = 0

for index, feature in enumerate(clf.feature_importances_):
    if feature > 0.2:
        key_index = index
        print index, feature
################################################################################
### your code goes here       
words_bags = vectorizer.get_feature_names()
print words_bags[key_index]

```

# 不断删除异常特征值并重复
### 从某种意义上说，这一单词看起来像是一个异常值，所以让我们在删除它之后重新拟合。 返回至 text_learning/vectorize_text.py，使用我们删除“sara”、“chris”等的方法，从邮件中删除此单词。 重新运行 vectorize_text.py，完成以后立即重新运行 find_signature.py。

# 每个主成分的可释方差
### 人脸识别
### eigenfaces.py
```python
"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""



print __doc__

from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print "Total dataset size:"
print "n_samples: %d" % n_samples
print "n_features: %d" % n_features
print "n_classes: %d" % n_classes


###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print "done in %0.3fs" % (time() - t0)

eigenfaces = pca.components_.reshape((n_components, h, w))

print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.3fs" % (time() - t0)



### your code goes here 
print pca.explained_variance_ratio_                                          ##





###############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_


###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0)

print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

pl.show()



```

# 先进行PCA主成分分析,再进行特征选择,除非特征非常多,而你知道此特征一定不相关.

# 在 Sklearn 中训练/测试分离  iris数据集
```python
#!/usr/bin/python

""" 
PLEASE NOTE:
The api of train_test_split changed and moved from sklearn.cross_validation to
sklearn.model_selection(version update from 0.17 to 0.18)

The correct documentation for this quiz is here: 
http://scikit-learn.org/0.17/modules/cross_validation.html
"""

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
features = iris.data
labels = iris.target

###############################################################
### YOUR CODE HERE
###############################################################

### import the relevant code and make your train/test split
### name the output datasets features_train, features_test,
### labels_train, and labels_test
# PLEASE NOTE: The import here changes depending on your version of sklearn
from sklearn import cross_validation # for version 0.17
# For version 0.18
# from sklearn.model_selection import train_test_split


### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

###############################################################
# DONT CHANGE ANYTHING HERE
clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)
##############################################################
def submitAcc():
    return clf.score(features_test, labels_test)
```

# K折必须打乱数据,否则会出现用sara特征预测chirs的情况.

# GridSearchCV 用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。它的好处是，只需增加几行代码，就能遍历多种组合。
##  下面是来自 sklearn 文档 的一个示例：
```python
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```
""" 
### 让我们逐行进行说明。
```python
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
```
### 参数字典以及他们可取的值。在这种情况下，他们在尝试找到 kernel（可能的选择为 'linear' 和 'rbf' ）和 C（可能的选择为1和10）的最佳组合。
### 这时，会自动生成一个不同（kernel、C）参数值组成的“网格”:
```python
"""
('rbf', 1)
('rbf', 10)
('linear', 1)
('linear', 10)
"""
```
### 各组合均用于训练 SVM，并使用交叉验证对表现进行评估。
```python
svr = svm.SVC() 
```
### 这与创建分类器有点类似，就如我们从第一节课一直在做的一样。但是请注意，“clf” 到下一行才会生成—这儿仅仅是在说采用哪种算法。另一种思考方法是，“分类器”在这种情况下不仅仅是一个算法，而是算法加参数值。请注意，这里不需对 kernel 或 C 做各种尝试；下一行才处理这个问题。
```python
clf = grid_search.GridSearchCV(svr, parameters) 
```
### 这是第一个不可思议之处，分类器创建好了。 我们传达算法 (svr) 和参数 (parameters) 字典来尝试，它生成一个网格的参数组合进行尝试。
```python
clf.fit(iris.data, iris.target)
``` 
### 第二个不可思议之处。 拟合函数现在尝试了所有的参数组合，并返回一个合适的分类器，自动调整至最佳参数组合。现在您便可通过 clf.best_params_ 来获得参数值。



# 过拟合的poi识别
### 构建最简单（未经过验证的）POI 识别符 未进行交叉验证,直接在所有数据集上使用决策树
### validate_poi.py
```python
#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
    ### create classifier
clf = tree.DecisionTreeClassifier()  

    ### fit the classifier on the training features and labels
clf.fit(features, labels) 

    ### use the trained classifier to predict labels for the test features
pred = clf.predict(features)  


print accuracy_score(pred, labels)


```

# 部署训练/测试机制
### 添加训练和测试, 使用 sklearn.cross_validation 中的 train_test_split 验证； 将 30% 的数据用于测试，并设置 random_state 参数为 42（random_state 控制哪些点进入训练集，哪些点用于测试；将其设置为 42 意味着我们确切地知道哪些事件在哪个集中； 并且可以检查你得到的结果）。更新后的准确率是多少？
### validate_poi.py
```python
#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
########################################################################################
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
#######################################################################################
### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
    ### create classifier
clf = tree.DecisionTreeClassifier()  

    ### fit the classifier on the training features and labels
clf.fit(features_train, labels_train) 

    ### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)  


print accuracy_score(pred, labels_test)


```


# 机器学习实战项目 安然数据集识别poi
## 首先是自己的代码
### 三个文件
### feature_format.py
```python
#!/usr/bin/python

""" 
    A general tool for converting data from the
    dictionary format to an (n x k) python list that's 
    ready for training an sklearn algorithm

    n--no. of key-value pairs in dictonary
    k--no. of features being extracted

    dictionary keys are names of persons in dataset
    dictionary values are dictionaries, where each
        key-value pair in the dict is the name
        of a feature, and its value for that person

    In addition to converting a dictionary to a numpy 
    array, you may want to separate the labels from the
    features--this is what targetFeatureSplit is for

    so, if you want to have the poi label as the target,
    and the features you want to use are the person's
    salary and bonus, here's what you would do:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = featureFormat( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    the line above (targetFeatureSplit) assumes that the
    label is the _first_ item in feature_list--very important
    that poi is listed first!
"""


import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print "error: key ", feature, " not present"
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features





```
### tester.py
```python
#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()

```
### poi_id.py
```python

# coding: utf-8

# # 进阶项目3：从安然公司邮件中发现欺诈证据

# In[1]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


# ## 数据探索

# In[2]:

### 载入数据集
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict


# In[3]:

# 数据点总数
print u'数据点总数:{}'.format(len(data_dict))
print '*'*100


# In[4]:

#类之间的分配（POI/非 POI)
def count_poi(data_dict):
    n = 0
    for name in data_dict:
        if data_dict[name]["poi"] == True:
            n += 1
    return n
p = count_poi(data_dict)
print u'类之间的分配（POI/非 POI）:{}/{}'.format(p,146-p)
print '*'*100


# In[5]:

#使用的特征数量
features_list=['poi','bonus','salary','to_messages', 'deferral_payments', 

               'expenses','deferred_income', 'long_term_incentive',

               'restricted_stock_deferred', 'shared_receipt_with_poi',

               'loan_advances', 'from_messages', 'other', 'director_fees', 

               'total_stock_value', 'from_poi_to_this_person',

               'from_this_person_to_poi', 'restricted_stock',  

               'total_payments','exercised_stock_options','email_address']
print u'使用的特征数量:{}'.format(len(features_list)-1)
print '*'*100


# In[6]:

#是否有哪些特征有很多缺失值
def count_nan(data_dict, features_list):
    null_value = {}
    for name in data_dict:
        for feature in features_list:
            if data_dict[name][feature] == 'NaN':
                if feature not in null_value:
                    null_value[feature] = 1 
                else:
                    null_value[feature] += 1
    return null_value
nan_fea = count_nan(data_dict, features_list)
nan_feature = sorted(nan_fea.items(), key=lambda x: x[1],reverse=True)
print u'以下特征含有缺失值:'
for i in nan_feature:
    print i
print '*'*100


# In[7]:

#是否有哪些人有很多缺失值
def count_nan_p(data_dict, features_list):
    null_value = {}
    for name in data_dict:
        for feature in features_list:
            if data_dict[name][feature] == 'NaN':
                if name not in null_value:
                    null_value[name] = 1 
                else:
                    null_value[name] += 1
    return null_value
nan_peo = count_nan_p(data_dict, features_list)
nan_people = sorted(nan_peo.items(), key=lambda x: x[1],reverse=True)
print u'有一个人所有特征全部为NaN，没有有用信息:'
print nan_people[:1]
print '*'*100


# ## 异常值调查

# In[8]:

#确定财务数据中的异常值，并解释如何消除或以其他方式处理它们。
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
data_dict.pop('LOCKHART EUGENE E',0)
print u'删除以下异常值:\nTOTAL\nTHE TRAVEL AGENCY IN THE PARK\nLOCKHART EUGENE E'
print '*'*100


# ## 创建新特征

# In[9]:

#书面回复中提供了选择该特征的理由，并测试了该特征对最终算法性能的影响。
# 总报酬+总股票 = 总收入
for keys,features in data_dict.items():
        if features['total_payments'] == "NaN" or features['total_stock_value'] == "NaN":
            features['total_net_worth'] = "NaN"
        else:
            features['total_net_worth'] = features['total_payments'] + features['total_stock_value']
features_list += ['total_net_worth']
len(features_list)
print u'添加一个新的财务特征:total_net_worth(总报酬+总股票 = 总收入)'
print '*'*100


# In[10]:

#查看新特征有多少缺失值
def count_nan_newfeature(data_dict, features_list):
    n = 0
    for name in data_dict:
        if data_dict[name]['total_net_worth'] == 'NaN':
            n += 1
    return n
nan_newfeature = count_nan_newfeature(data_dict, features_list)
print u'新特征total_net_worth缺失值为:{}'.format(nan_newfeature)
print '*'*100


# ## 选择算法

# In[11]:

#至少尝试了 2 种不同的算法并比较了它们的性能，最终分析中使用了性能较高的一个.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



#朴素贝叶斯
NB = GaussianNB()

#决策树
dt = DecisionTreeClassifier()

#k近邻
kn=KNeighborsClassifier()


# ## 验证策略

# In[12]:

# 定义测试函数
from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


#存储到我的数据集下便于导出
my_dataset = data_dict        


# ## 测试新特征

# In[13]:

#测试新特征
print u'测试新特征对最终算法性能的影响:'
features_new = ['poi', 'total_net_worth']
print u'朴素贝叶斯:'
test_classifier(NB,my_dataset,features_new,folds = 1000)

print u'决策树:'
test_classifier(dt,my_dataset,features_new,folds = 1000)

print u'k近邻:'
test_classifier(kn,my_dataset,features_new,folds = 1000)
print '*'*100


# ## 明智地选择特征

# In[14]:

#部署单变量或递归特征
#对于支持获取特征重要性（如：决策树）或特征得分（如：SelectKBest）的算法，进行记录.
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest,f_classif

#分析财务特征时删除邮件地址,str和float冲突
features_list.remove('email_address')

#提取特征和标签
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[15]:

print u'使用SelectKBest的算法选择10个得分最高的特征:'
#获取最重要10个特征
features_selected=[]
clf = SelectKBest(f_classif, k=10)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_10 = ['poi']+features_selected


#特征得分
features_score = zip(features_list_10[1:25], clf.scores_[:24])
features_score = sorted(features_score, key=lambda s: s[1], reverse=True)
for i in features_score:
    print i
print '*'*100


# In[55]:

print u'大多数都是财务特征,只有一个邮件特征排在第七位'
print '*'*100


# ### 使用SelectKBest尝试不同的特征组合(k=10~1)，并记录了每种组合的性能

# In[52]:

#测试选择6个最佳特征,各分类器得分
features_selected=[]
clf = SelectKBest(f_classif, k=6)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_6 = ['poi']+features_selected

print u'默认参数的朴素贝叶斯分类器最高得分为使用6个最佳特征:'
print u'朴素贝叶斯:'
test_classifier(NB,my_dataset,features_list_6,folds = 1000)

print '*'*100


# In[54]:

#测试选择3个最佳特征,各分类器得分
features_selected=[]
clf = SelectKBest(f_classif, k=3)
selected_features = clf.fit_transform(features, labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_list_3 = ['poi']+features_selected

print u'默认参数的决策树分类器最高得分为使用3个最佳特征:'
print u'决策树:'
test_classifier(dt,my_dataset,features_list_3,folds = 1000)

print '*'*100
print u'默认参数的k近邻分类器得分不理想:'
print u'k近邻:'
test_classifier(kn,my_dataset,features_list_3,folds = 1000)
print '*'*100


# In[53]:

print '*'*100
print u'默认参数下,精确度和召回率最高得分是朴素贝叶斯分类器,SelectKBest(k=6),精确度0.48,召回率0.36,f1=0.41'

#特征得分
features_score = zip(features_list_6[1:25], clf.scores_[:24])
features_score = sorted(features_score, key=lambda s: s[1], reverse=True)
print u'使分类器精确度和召回率最高的六个特征为:'
for i in features_score:
    print i
print '*'*100


# ## 调整算法

# In[60]:

#调整算法
print u'调整K近邻分类器参数,提高算法性能:'
knc=KNeighborsClassifier(n_neighbors=2, weights='distance', n_jobs=-1)
print u'正在计算,预计时间为2分钟..'
test_classifier(knc,my_dataset,features_list_3,folds = 1000)


# In[65]:

print u'***************最终算法*********************'
print u'调参后的KNeighborsClassifier性能最高'
print u'参数为:  n_neighbors=2, weights=distance'
print u'Precision: 0.49715   Recall: 0.39250   F1: 0.43867'


# In[14]:

from tester import dump_classifier_and_data
dump_classifier_and_data(knc, my_dataset, features_list_3)


```

## 然后是别人的代码 有主成分和数据管道,但是分类分数不高,且pca适用于非监督学习
### 前两个文件一样
### poi_id_test.py
```python
#!/usr/bin/python

import sys

import pickle

import matplotlib.pyplot as plt

sys.path.append("../tools/")

#%%

from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data,test_classifier

#%%

### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.

### The first feature must be "poi".

features_list=['poi','bonus','salary','to_messages', 'deferral_payments', 

               'expenses',

                'deferred_income', 

               'long_term_incentive',

               'restricted_stock_deferred', 

               'shared_receipt_with_poi', 'loan_advances',

               'from_messages', 'other', 'director_fees', 

               'total_stock_value', 'from_poi_to_this_person',

               'from_this_person_to_poi', 'restricted_stock',  

               'total_payments','exercised_stock_options','email_address']             

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:

    data_dict = pickle.load(data_file)

#缺失值和poi计数

key=list(data_dict.keys())

count=0

poi_count=0

for i in range(len(key)):

    for j in features_list:

        if data_dict[key[i]][j]=='NaN':

            count+=1

    if data_dict[key[i]]['poi']==1:

        poi_count+=1

print "Number of missing data points in the giving dataset :",count   

print "Number of employees found gulilty :",poi_count,"(out of 146)"         

#%%

## required Functions
#poi邮件比例


def computeFraction( poi_messages, all_messages ):

    if poi_messages=='NaN' or all_messages=='NaN':

        return 'NaN'

    else:

        return float(poi_messages)/all_messages

def data_points_of(feature):    

    data_points=[]

    names=list(my_dataset.keys())   

    for i in range(len(names)):

        if my_dataset[names[i]][feature] == 'NaN' :

            data_points.append(0)

        else:

            data_points.append(my_dataset[names[i]][feature])

    return data_points

#%%

### Task 2: Remove outliers

data_dict.pop("TOTAL",0)

data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

### Task 3: Create new feature(s)



#creating 4 new features 

for keys,features in data_dict.items():

    x=features['from_this_person_to_poi']

    y=features['to_messages']

    a=features['from_poi_to_this_person']

    b=features['from_messages']

    c=features['total_payments']

    d=features['total_stock_value']

    e=features['bonus']

    f=features['salary']

#fraction_from/to_poi is the fraction of poi_messages and from/to_messages
# 来自/发送poi邮件比例

    features['fraction_from_poi']=computeFraction(a,b)

    features['fraction_to_poi']=computeFraction(x,y) 

#bonus_salary_ratio is the ratio of bonus to salary
# 工资奖金比例

    if e=='NaN' or f=='NaN':

        features['bonus_salary_ratio']='NaN'

    else :

        features['bonus_salary_ratio']=float(e) /float(f)   

# total net worth is the sum of tatal payments and total stock value
#总资产是支付+股票, nan不能和int相加


    if c=='NaN' or d=='NaN':

        features['total_net_worth']='NaN'

    else:

        features['total_net_worth']=c+d   



#adding new features to features_list
#新特征加入特征列表

features_list+=['total_net_worth']+['bonus_salary_ratio']+\
['fraction_from_poi']+['fraction_to_poi']

# eleminating the feature 'email_address'
#删除邮件地址特征,特征计数

features_list.remove('email_address')

print "\ntotal number of features in including new features:"

print len(features_list),"\n"

#%%

### Store to my_dataset for easy export below.
#存储到我的数据集下便于导出。

my_dataset = data_dict

### Extract features and labels from dataset for local testing
# 提取特征和标签

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

#%%



## selecting 7 best features excluding 'poi'
# 找到7个最佳特征排除poi
#操作符函数
## get_support对应的是选取哪几个feature。如果被选中就是True，反之则为False。
## transform的输入是包含所有特征的矩阵，输出结果是只保留了被选择的七个特征的矩阵。
import operator

from sklearn.feature_selection import SelectKBest,f_classif

features_selected=[]

clf = SelectKBest(f_classif,k=7)

selected_features = clf.fit_transform(features,labels)

for i in clf.get_support(indices=True):

    features_selected.append(features_list[i+1])

features_score = zip(features_list[1:25],clf.scores_[:24])

features_score = sorted(features_score,key=operator.itemgetter(1),reverse=True)

#特征分数排序

features_list=['poi']+features_selected

print "Scores of the features :\n"

for i in features_score:

    print i

print " \n THE BEST 8 Features including 'poi' are :"

print features_list

#%%

### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.

### Note that if you want to do PCA or other multi-stage operations,

### you'll need to use Pipelines. For more info:

### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report,accuracy_score,recall_score,\
precision_score

#%%

# Provided to give you a starting point. Try a variety of classifiers.

print "\nGaussianNB classifier(Default) :"

NB = GaussianNB()

test_classifier(NB,my_dataset,features_list,folds = 1000)

#%%

print "\nDecission Tree classifier(default) :"

dt = DecisionTreeClassifier()

test_classifier(dt,my_dataset,features_list,folds = 1000)

#%%

print "\nKneighbour classifier(default)"

kn=KNeighborsClassifier()

test_classifier(kn,my_dataset,features_list,folds = 1000)

#%%

print "\n PCA with decisiontree \n"

param_grid = {

         'pca__n_components':[1,2,3,4,5,6],

         'tree__min_samples_split':[2,5,10,100],

         'tree__criterion':['gini'],

         'tree__splitter':['best']

          }

estimators = [('pca',PCA()),('tree',DecisionTreeClassifier())]

pipe = Pipeline(estimators)

gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1')

gs.fit(features,labels)

clf = gs.best_estimator_

test_classifier(clf,my_dataset,features_list,folds = 1000)

#%%

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### using our testing script. Check the tester.py script in the final project

### folder for details on the evaluation method, especially the test_classifier

### function. Because of the small size of the dataset, the script uses

### stratified shuffle split cross validation. For more info: 

### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



#%%

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

shuffle= StratifiedShuffleSplit(labels_train,n_iter = 25,test_size = 0.5,

                                random_state = 0)

print "\n Best classifier : PCA with GaussianNb \n"

param_grid = {

         'pca__n_components':[1,2,3,4,5,6]

          }

estimators = [('pca',PCA()),('gaussian',GaussianNB())]

pipe = Pipeline(estimators)

gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1',cv=shuffle)

gs.fit(features_train,labels_train)

pred=gs.predict(features_test)

clf = gs.best_estimator_

test_classifier(clf,my_dataset,features_list,folds = 1000)

print "\n\nbest parameters ",gs.best_params_

print '\n Accuracy:',accuracy_score(pred,labels_test),\

"\n Precision:",precision_score(pred,labels_test),\

"\nRecall",recall_score(pred,labels_test)

#%%

### Task 6: Dump your classifier, dataset, and features_list so anyone can

### check your results. You do not need to change anything below, but make sure

### that the version of poi_id.py that you submit can be run on its own and

### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(clf, my_dataset, features_list)
```
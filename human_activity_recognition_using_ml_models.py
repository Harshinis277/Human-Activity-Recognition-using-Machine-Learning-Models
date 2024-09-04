#!/usr/bin/env python
# coding: utf-8

# ### Project : 
# ## Human Activity Recognition : Predictions using ML Models

# In[1]:


# Importing necessary libraries

import numpy as np
import pandas as pd


# In[2]:


# Reading data from CSV file

train = pd.read_csv('UCI_HAR_Dataset/csv_files/train.csv')
test = pd.read_csv('UCI_HAR_Dataset/csv_files/test.csv')


# In[ ]:


# Checking the shape of train and test

print(train.shape)
print(test.shape)


# In[ ]:


# Displaying first 5 rows of training data

train.head(5)


# In[ ]:


# Displaying first 5 rows of testing data

test.head(5)


# In[ ]:


# Getting X_train and y_train from train data

X_train = train.drop(['Subject', 'Activity', 'Activity_Name'], axis=1)
y_train = train.Activity


# In[ ]:


# Getting X_test and y_test from test data

X_test = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
y_test = test.Activity


# In[ ]:


# Displaying the shape of training and testing data

print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))


# In[ ]:


# Let's use Linear discriminant analysis to find features that classifies the label well

# Importing libraries

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[ ]:


lda = LDA()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[ ]:


# Displaying the shape of training and testing data

print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))


# ## Let's define some generic functions to create ML models

# ### Function to plot Confusion Matrix

# In[ ]:


# Importing necessary libraries

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Creating a function to print text in Bold and in given color

from IPython.display import Markdown, display

def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


# In[ ]:


# Function to plot Confusion Matrix

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion Matrix',
                         cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


# In[ ]:


# Generic function to any model

from datetime import datetime
def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True,                  print_cm=True, cm_cmap=plt.cm.Greens):
    
    
    # Let's create an empty dictionary to be returned by the function
    results = dict()
    
    # Let's calculate & print the total training time
    
    train_start_time = datetime.now()
    model.fit(X_train, y_train)
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    printmd('Training_time(HH:MM:SS.ms) - {}'.format(results['training_time']), color='blue')
        
    # Let's calculate & print the test time
    
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    results['testing_time'] = test_end_time - test_start_time
    printmd('testing time(HH:MM:SS:ms) - {}'.format(results['testing_time']), color='blue')
    results['predicted'] = y_pred
   
    # Let's calculate the Accuracy of Model
    
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    results['accuracy'] = accuracy
    printmd('**Accuracy:**', color='blue')
    print('{}'.format(accuracy))
        
    # Let's get the Confusion Matrix
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Plotting Confusion Matrix
    
    printmd('**Confusion Matrix:**', color='blue')
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=labels, title='Normalized confusion matrix', cmap=plt.cm.YlGn, )
    plt.show()
        
    # Plotting Normalized Confusion Matrix
    
    printmd('**Normalized Confusion Matrix:**', color='blue')
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # PLotting classification report
    
    printmd('**Classifiction Report**', color='blue')
    classification_report = metrics.classification_report(y_test, y_pred)
    results['classification_report'] = classification_report
    print(classification_report)
    
    # Adding the trained model to the results
    
    results['model'] = model
    
    return results


# In[ ]:


def print_grid_search_attributes(model):
    
    # Let's print the best estimator that gave highest score
    
    printmd('**Best Estimator:**', color='blue')
    print('{}\n'.format(model.best_estimator_))


    # Let's print the best parameters that gave best results
    
    printmd('**Best parameters:**', color='blue')
    print('{}\n'.format(model.best_params_))


    #  Let's print the number of cross validation splits
    
    printmd('**Number of CrossValidation sets:**', color='blue')
    print('{}\n'.format(model.n_splits_))


    # Let's print the Best score of the best estimator
    
    printmd('**Best Score:**', color='blue')
    print('{}\n'.format(model.best_score_))


# ## Applying various Machine learning model with Grid-Search

# ### 1. Logistic Regression

# In[ ]:


# Importing necessary libraries

from sklearn import linear_model
from sklearn import metrics

from sklearn.model_selection import GridSearchCV


# In[ ]:


# Creating a list labels to be added to plots

labels=['Laying', 'Sitting','Standing','Walking','Walking_Downstairs','Walking_Upstairs']


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'C':[20, 25, 30, 35, 40], 'penalty':['l1', 'l2']}

# Let's initiate the model

log_reg = linear_model.LogisticRegression()
log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, verbose=1, n_jobs=-1)
log_reg_grid_results =  perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(log_reg_grid_results['model'])


# ### 2. Support Vector Classifier

# In[ ]:


# Importing Necessary libraries

from sklearn.svm import LinearSVC


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'C':[0.25, 0.5, 1, 2, 4, 8]}

# Let's initiate the model

lin_svc = LinearSVC()
lin_svc_grid = GridSearchCV(lin_svc, param_grid=parameters, verbose=1, n_jobs=-1)
lin_svc_grid_results = perform_model(lin_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(lin_svc_grid_results['model'])


# ### 3. Kernel SVM

# In[ ]:


# Importing Libraries

from sklearn.svm import SVC


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'C':[0.125, 0.25, 0.5, 1], 'gamma':[0.01, 0.1, 1, 2]}

# Let's initiate the model

rbf_svc = SVC(kernel='rbf')
rbf_svc_grid = GridSearchCV(rbf_svc, param_grid=parameters)
rbf_svc_grid_results = perform_model(rbf_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(rbf_svc_grid_results['model'])


# ### 4. Decision Tree

# In[ ]:


# Importing libraries

from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'max_depth':np.arange(4,10,1)}

# Let's initiate the model

dtree = DecisionTreeClassifier()
dtree_grid = GridSearchCV(dtree, param_grid=parameters, verbose=1, n_jobs=-1)
dtree_grid_results = perform_model(dtree_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(dtree_grid_results['model'])


# ### 5. Random Forest Classifier

# In[ ]:


# Importing libraries

from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(4,15,2)}

# Let's initiate the model

rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=parameters, n_jobs=-1)
rfc_grid_results = perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(rfc_grid_results['model'])


# ### 6. Gradient Boosted Decision Tree

# In[ ]:


# Importing Libraries

from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


# Let's define the parameters to be tuned

parameters = {'n_estimators': np.arange(120,150,10), 'max_depth':np.arange(3,7,1)}

# Let's initiate the model

gbdt = GradientBoostingClassifier()
gbdt_grid = GridSearchCV(gbdt, param_grid=parameters, n_jobs=-1)
gbdt_grid_results = perform_model(gbdt_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# Printing the best attributes of the model 

print_grid_search_attributes(gbdt_grid_results['model'])


# ### Let's compare all the models together

# In[ ]:


print('\n                     Accuracy     Error')
print('                     ----------   --------')
print('Logistic Regression : {:.04}%      {:.04}%'.format(log_reg_grid_results['accuracy'] * 100,                                                  100-(log_reg_grid_results['accuracy'] * 100)))

print('Linear SVC          : {:.04}%       {:.04}% '.format(lin_svc_grid_results['accuracy'] * 100,                                                        100-(lin_svc_grid_results['accuracy'] * 100)))

print('RBF SVM classifier  : {:.04}%      {:.04}% '.format(rbf_svc_grid_results['accuracy'] * 100,                                                          100-(rbf_svc_grid_results['accuracy'] * 100)))

print('Decision Tree       : {:.04}%      {:.04}% '.format(dtree_grid_results['accuracy'] * 100,                                                        100-(dtree_grid_results['accuracy'] * 100)))

print('Random Forest       : {:.04}%       {:.04}% '.format(rfc_grid_results['accuracy'] * 100,                                                           100-(rfc_grid_results['accuracy'] * 100)))
print('GradientBoosting    : {:.04}%      {:.04}% '.format(gbdt_grid_results['accuracy'] * 100,                                                        100-(gbdt_grid_results['accuracy'] * 100)))


# ## Conclusion

# The above table shows that Logistic Regression, Linear SVC, RBF SVM classifier & Random Forest has highest Accuracy with lowest Error value. We can use any of these three models for future predictions

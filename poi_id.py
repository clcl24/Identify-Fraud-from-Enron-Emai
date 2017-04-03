import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = ['poi']
features_financial = ["bonus", "deferral_payments", "deferred_income",
                      "director_fees", "exercised_stock_options",
                      "expenses", "loan_advances",
                      "long_term_incentive", "other",
                      "restricted_stock", "restricted_stock_deferred",
                      "salary", "total_payments", "total_stock_value"]
features_email = ['from_messages', 'from_poi_to_this_person',
                  'from_this_person_to_poi', 'shared_receipt_with_poi',
                  'to_messages']

features_list = poi_label + features_email + features_financial


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") )

# Fix a couple of records so check_total matches enron61702insiderpay.pdf
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 2: Remove outliers
# create a scatter plot with the salary and bonus features
def creat_scatter(outliers = False):
    data = featureFormat(data_dict, ['poi','salary', 'bonus'], sort_keys = True)
    poi=data[:,0]
    salary = data[:,1]
    bonus = data[:,2]
    plt.scatter(salary[poi==1],bonus[poi==1],c='red',s=50,label='poi')
    plt.scatter(salary[poi==0],bonus[poi==0],c='blue',s=50,label='not poi')
    plt.xlabel('Salary')
    plt.ylabel('Bonus')
    plt.legend(loc='lower right')
    plt.title('Bonus vs Salary')
    if outliers:
        name = 'Bonus vs Salary without outliers.png'
    else:
        name = 'Bonus vs Salary with outliers.png'
    plt.savefig(name)
    # plt.show()

creat_scatter()

# remove "TOTAL", 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'
# 'LOCKHART EUGENE E' all values are NaN
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for outlier in outliers:
    data_dict.pop(outlier, 0)

# create a scatter plot with the salary and bonus features witout outliers
creat_scatter(outliers = True)

# data exploration
print ('number of employees in the dataset: ', len(data_dict.keys()))
print ('number of features for each person: ', len(data_dict[list(data_dict.keys())[0]]))

poi_counter = 0
for person in data_dict.keys():
    if data_dict[person]['poi'] == True:
        poi_counter += 1
print ("number of POIs: ", poi_counter)
print ("Proportion of POIs in the dataset: ", poi_counter / len(data_dict.keys()))

nan_count = 0
features_all = features_list + ['email_address']
for person in data_dict.keys():
    for feat in features_all:
        if data_dict[person][feat] == "NaN":
            nan_count += 1

print ("number of NaN: ", nan_count)


### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """
    if poi_messages != "NaN" or all_messages != "NaN":
        fraction = float(poi_messages)/float(all_messages)
    else:
        fraction = 0.
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

features_new = ['fraction_from_poi', 'fraction_to_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = features_list + features_new

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Determine DecisionTree Feature Importances
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier()
clf_tree.fit(features, labels)
unsorted_pairs = zip(features_list[1:],clf_tree.feature_importances_)
sorted_dict = sorted(unsorted_pairs, key=lambda feature: feature[1], reverse = True)
tree_best_features = sorted_dict[:len(features_list)]
print (tree_best_features)

print ("---Sorted Features from Decision Tree Feature Importances---")
features_list = ['poi']
for item in tree_best_features:
    # print (item[0], item[1])
    if item[1] > 0.10:
        features_list.append(item[0])
print (features_list)

# Extract features and labels from dataset with new features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Determine K-best features
# from sklearn.feature_selection import SelectKBest
#
# anova_filter = SelectKBest()
# anova_filter.fit(features, labels)
# scores = anova_filter.scores_
# unsorted_pairs = zip(features_list[1:], scores)
# sorted_dict = sorted(unsorted_pairs, key=lambda feature: feature[1], reverse = True)
# anova_best_features = sorted_dict[:len(features_list)]
# print (anova_best_features )
#
# print ("---TOP 5 Features from SelectKBest---")
# features_5 = []
# for item in anova_best_features [0:5]:
#     features_5.append(item[0])
# print (features_5)

# # scale features via min-max
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)
#
# from sklearn.pipeline import Pipeline
#
# from sklearn.naive_bayes import GaussianNB
# clf_NB = Pipeline([('anova', anova_filter), ('clf', GaussianNB())])
#
# from sklearn.linear_model import LogisticRegression
# clf_log = Pipeline([('anova', anova_filter), ('clf', LogisticRegression())])
#
# from sklearn.ensemble import RandomForestClassifier
# clf_RF = Pipeline([('anova', anova_filter), ('clf', RandomForestClassifier())])


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

# DecisionTree
params_tree = { "min_samples_split":[2, 5, 10, 20],
                "criterion": ('gini', 'entropy') }

clf_tree = GridSearchCV(clf_tree, params_tree, cv=cv)
clf_tree.fit(features, labels)

print("--Decision Tree--")
clf_tree = clf_tree.best_estimator_
test_classifier(clf_tree, my_dataset, features_list)

# # Naive Bayes
# params_tree= {'anova__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]}
# clf_NB = GridSearchCV(clf_NB, params_tree, cv=cv)
# clf_NB.fit(features, labels)
#
# print("--Naive Bayes GaussianNB--")
#
# clf_NB = clf_NB.best_estimator_
# test_classifier(clf_NB, my_dataset, features_list)
#
# # LogisticRegression
# params_log = {'anova__k': [5,10,15,20,21],
#               'clf__C':[0.05, 0.5, 1, 10],
#               "clf__tol":[10**-1, 10**-5, 10**-10],
#               "clf__class_weight":['balanced']}
# clf_log = GridSearchCV(clf_log, params_log, cv=cv)
# clf_log.fit(features, labels)
#
# print("--LogisticRegression--")
#
# clf_log = clf_log.best_estimator_
# test_classifier(clf_log, my_dataset, features_list)
#
# # RandomForest
# params_RF = {'anova__k': [5,10,15,20,21],
#              'clf__n_estimators': [50,100,200],
#              'clf__criterion': ('gini', 'entropy')}
# clf_RF = GridSearchCV(clf_RF, params_RF, cv=cv)
# clf_RF.fit(features, labels)
#
# print("--RandomForest--")
#
# clf_RF = clf_RF.best_estimator_
# test_classifier(clf_RF, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_tree, my_dataset, features_list)

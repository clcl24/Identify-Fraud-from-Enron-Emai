## Enron Submission Free-Response Questions

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is using financial and email data from Enron corpus to identify Enron Employees who may have committed fraud by utilizing machine learning. These employees are known as a person of interest(POIs), which are defined as individuals who were indicted, reached a settlement, or testified in exchange for prosecution immunity.

Before implementing the machine learning, I created the scatter plot with the salary and bonus features, and find a outlier – “TOTAL”. What’s more, compared the dataset with data from enron61702insiderpay.pdf, I found that “THE TRAVEL AGENCY IN THE PARK” is another outlier, and the data of “BELFER ROBERT” and “BHATNAGAR SANJAY” are not matched between these two data sources. Also, I found all feature values for “LOCKHART EUGENE E” are NAN. All these data can be considered as outliers. I removed “TOTAL”, “THE TRAVEL AGENCY IN THE PARK” and “LOCKHART EUGENE E” and updated the data of “BELFER ROBERT” and “BHATNAGAR SANJAY”.

After removing/updating records, there are 143 employees in the data set remaining, and every employee has 21 features including exercised stock options, total stock value, bonus, salary, deferred income, etc. Of these, 18 individuals were labeled as a POI while the remaining 125 were not. The Proportion of POIs in the dataset is 12.59%. And the number of missing values in the updated dataset is 1314.

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

Features I ended up using are 'exercised_stock_options', 'shared_receipt_with_poi', 'fraction_to_poi', 'other' and 'expenses'.  I selected these features because their importance values from DecisionTreeClassfier are higher than 0.10, and Decision Tree is the final algorithm I selected. When I choose the threshold for importance, I tried 0.10, 0.15, and 0.20, the result showed that the threshold_0.1 gave the DecisionTreeClassfier the best performance, that is, Precision, Recall, F1 Score(the reasons why choose these three metrics will be demonstrated in the last question) were higher when 0.1 is selected.

importance   | 0.10    | 0.15    | 0.20
------------ |-------- | ------- | -------
Precision    | 0.56192 | 0.48164 | 0.48921
Recall       | 0.39250 | 0.40650 | 0.34000
F1           | 0.46217 | 0.44089 | 0.40118

Decision Tree           | importance
----------------------- | ---------------------
exercised_stock_options | 0.25704012789768194
shared_receipt_with_poi | 0.17833784461152871
expenses                | 0.17030977443609024
fraction_to_poi         | 0.13560272924517558
other                   | 0.13074285714285716

When testing other classifiers (Gaussian Naive Bayes, Random Forest, LogisticRegression), I did not select features mamnualy, but rather used SelectKBest with the entire feature list by the GridSearchCV pipeline, which allow the algorithm to select the K-best
features. The result was K=11 for Gaussian Naive Bayes, k=5 for LogisticRegression, k=5 for Random Forest.

Top 5 features are shown as below:

SelectKBest             | scores
----------------------- | --------------------
total_stock_value       | 22.510549090242055
exercised_stock_options | 22.348975407306217
bonus                   | 20.792252047181535
salary                  | 18.289684043404513
fraction_to_poi         | 16.409712548035792

Also, I used MinMaxScaler to transform the data into a 0 to 1 range for other three machine learning algorithms except Decision Tree. Different features have different range, and some of them have significantly wide range, so it is necessary to feature-scale for the features to be considered evenly.

Additional features I created are ‘fraction_from_poi’ and ‘fraction_to_poi’.

* fraction_from_poi = (from_poi_to_this_person / to_messages)
* fraction_to_poi = (from_this_person_to_poi / from_messages )

In order to test if these two new features have impact on my algorithms, I evaluated Precision, Recall and F1 Score obtained by DecisionTreeClassfier with the original dataset against the results obtained by the original dataset + the newly generated features. The result was that performance would be better if new features were added. I calculated them because I think if an individual sent/received a larger fraction of emails to/from a POI then it likely they too are a POI. Actually, ‘fraction_from_poi’ doesn't help a lot, ‘fraction_to_poi’ are on top 5 features when I used Decision Tree Importances.

metrics   | with new features | without new features
--------- |--------------------- | --------------------
Precision | 0.55327              | 0.36047
Recall    | 0.51150              | 0.21250
F1        | 0.53157              | 0.26738

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

The algorithm /I ended up use is Decision Tree, other three algorithms I used were Gaussian Naive Bayes, Random Forest, LogisticRegression. The Decision Tree Classifier with parameters (class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=20, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best') performed the best of all of the algorithms tested based on evaluation metrics used in this project (Precision: 0.55327	Recall: 0.51150	F1: 0.53157). The performance of Gaussian Naive Bayes(Accuracy: 0.83477	Precision: 0.44281	Recall: 0.28650	F1: 0.34791	F2: 0.30826), Random Forest(Accuracy: 0.83838	Precision: 0.44080	Recall: 0.18800	F1: 0.26358	F2: 0.21236), LogisticRegression(Accuracy: 0.22815	Precision: 0.16187	Recall: 0.96150	F1: 0.27709	F2: 0.48365) are not as good as Decision Tree's.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

Tuning the parameters of an algorithm means that adjust inputs of parameters for the algorithm to find the optimized combination that have best performance. If it not be well done, then the algorithm will not perform as well as it could.

For this project, I select the features for Decision Tree and other three algorithms(Gaussian Naive Bayes, Random Forest, Logistic Regression) through Decision Tree Importances or SelectKBest Scocres as I mentioned above. Then I tuned the parameters for different algorithms and find the best combination except Gaussian Naive Bayes for it does not have parameter to be tuned. The function I used is GridSearchCV, which is a way of systematically working through multiple combinations of parameter tunes. The details of what I did are shown below:

Decision Tree           | parameter value
----------------------- | --------------------
K                       | 1 to 21
min_samples_split       | [2, 5, 10, 20],
criterion               | ('gini', 'entropy')

Random Forest           | parameter value
----------------------- | --------------------
K                       | [5,10,15,20,21]
n_estimators            | [50,100,200]
criterion               | ('gini', 'entropy')

LogisticRegression      | parameter value
----------------------- | ---------------------------
K                       | [5,10,15,20,21]
C                       | [0.05, 0.5, 1, 10]
tol                     | [10**-1, 10**-5, 10**-10]
class_weight            | ['balanced']

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

Validation is using test data to test a machine learning algorithm to see how it performs. A classic mistake is training and testing using the same data to train and test.

In this project, used the function called test_classifier from tester.py to test algorithms. In this function, the StratifiedShuffleSplit function is implemented, it can split data into train and test set randomly for a pre-set times(1000 is pre-set in tester.py). Also, I used the StratifiedShuffleSplit function when I select the parameters in GridSearchCV. In this way, it can be sure that train and test data are randomly picked and different.

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

Actually, the results of test will give many metrics including Accuracy, Precision, Recall, F1 Score and F2 Score When I used test_classifier function from tester.py, but consider that there are few POIs in the dataset, I think Accuracy cannot represent the performance precisely, so I choose Precision, Recall and F1 Score ads my metrics.

* Precision = true positive / (true positive + false positive)
* Recall = true positive / (true positive + false negative)
* F1 = 2 * (precision * recall) / (precision + recall)

High precision value means POIs identified by the algorithm tended to be actual POIs. High recall means that the algorithm has good ability to identifying POIs if they are in the test set. The F1-score considers both precision and recall, so it can be considered a weighted average of the two.

The metric I got as show below:

classifier          | Precision | Recall  | F1
------------------- | --------- | ------- | --------
Decision Tree       | 0.56192	  | 0.39250 | 0.46217
Gaussian NB         | 0.40667	  | 0.30500	| 0.34857
Random Forest       | 0.51515	  | 0.14450 | 0.22569
Logistic Regression | 0.15072	  | 0.96200	| 0.26062

From the result, Decision Tree  and Gaussian NB met the threshold(precision and recall are both at least 0.3), but Decision Tree is better. Its Precision is 0.56, it means if this Decision Tree algorithm predicts 100 POIs, then the chance would be 56 people who are true POIs and the rest 44 are innocent. With a recall score of 0.39, this model can find about 39% of all real POIs in prediction.

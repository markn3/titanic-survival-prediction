import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(train.shape)
print(train.info())

# Feature engineering

# check for NULL values
print(train.isnull().sum())

# redundant feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

# replacing the missing values in 
# the Embarked feature with S
train = train.fillna({"Embarked": "S"})

# sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager',
          'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels)

# create a combined group of both datasets
combine = [train, test]

# extract a title for each Name in the 
# train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

# replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major',
                                                 'Rev', 'Jonkheer', 'Dona'],
                                                'Rare')

    dataset['Title'] = dataset['Title'].replace(
        ['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
                 "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

mr_age = train[train["Title"] == 1]["AgeGroup"].mode()  # Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  # Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()  # Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode()  # Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode()  # Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Adult

age_title_mapping = {1: "Young Adult", 2: "Student",
                     3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

train.loc[train["AgeGroup"] == "Unknown", "AgeGroup"] = train["Title"].map(age_title_mapping)
test.loc[test["AgeGroup"] == "Unknown", "AgeGroup"] = test["Title"].map(age_title_mapping)

# map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5, 'Adult': 6, 
               'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

# dropping the Age feature for now, might change
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)

train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


for x in range(len(test)):
    if pd.isnull(test.loc[x, "Fare"]):
        pclass = test.loc[x, "Pclass"]
        test.loc[x, "Fare"] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

# map Fare values into groups of 
# numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, 
                            labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, 
                           labels=[1, 2, 3, 4])

# drop Fare values
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)


from sklearn.model_selection import train_test_split

# Drop the Survived and PassengerId
# column from the trainset
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(
    predictors, target, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Random forest (bagging)
randomforest = RandomForestClassifier()

# Fit the training data along with its output
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)

# Find the accuracy score of the model
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(f"random forest: {acc_randomforest}")

# AdaBoost (boosting)
adaboost = AdaBoostClassifier()
# Fit the training data along with its output
adaboost.fit(x_train, y_train)
y_pred = adaboost.predict(x_val)

# find the accuracy of adaboost
acc_adaboost = round(accuracy_score(y_pred, y_val) * 100, 2)
print(f"Adaboost: {acc_adaboost}")

# Gradient Boosting Classifier (boosting)
gbclassifier = GradientBoostingClassifier()
# Fit the training data along with its output
gbclassifier.fit(x_train, y_train)
y_pred = gbclassifier.predict(x_val)

# find the accuracy of adaboost
acc_gbclassifier = round(accuracy_score(y_pred, y_val) * 100, 2)
print(f"Gradient Boosting Classifier: {acc_gbclassifier}")

# Stacking
# Define base learners (mixing bagging and boosting)
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),  # Bagging
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),  # Boosting
    ('ab', AdaBoostClassifier(n_estimators=100, random_state=42))  # Boosting
]

# Define the meta-model (Logistic Regression works well as a final estimator)
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train the model
stacking_model.fit(x_train, y_train)

# Predict and evaluate
y_pred = stacking_model.predict(x_val)
acc_stacking = round(accuracy_score(y_pred, y_val) * 100, 2)
print(f"stacking: {acc_stacking}")




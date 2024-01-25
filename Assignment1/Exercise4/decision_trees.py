from operator import le, ge
from sklearn.tree import DecisionTreeClassifier
import sklearn
import matplotlib.pyplot as plt
from helpers.helper import read_csv
GENDER = 8
AGE = 1
SURVIVED = 9
DATA='/home/josiah/repos/ECE457B-Fundamentals-of-Computational-Intelligence/Assignment1/Exercise4/A1Q3DecisionTrees.csv'

def calculate_impurity_of_only_gender_splitting_tree():
    data = read_csv(filename=DATA)

    correct_count = total_count = 0

    for person in data[1:]:
        total_count += 1
        if person[GENDER] == '0' and person[SURVIVED] == '1': # check if female and survived
            correct_count += 1
        elif person[GENDER] == '1' and person[SURVIVED] == '0': # male and not survived
            correct_count += 1
    # return wrong predictions / total predictions
    return round(1 - (correct_count / total_count), 3)# calculate impurity

def age_impurity(age=25, sign=le):
    data = read_csv(filename=DATA)
    correct_count = 0
    total_count = 0
    for person in data[1:]:
        total_count += 1
        if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # predict survive
            correct_count += 1
        elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
            correct_count += 1

    return round(1 - (correct_count / total_count), 3)

def compare_age_impurity():
    print(f"Impurity at boundary=25 is {age_impurity(25)}, impurity at boundary=65 is {age_impurity(65, sign=ge)}")
    return

def gender_age_impurity(age=25, sign=le):
    # gender split at first level
    # age split at over 25/65 second level
    data = read_csv(filename=DATA)
    correct_count = total_count = 0
    for person in data[1:]:
        total_count += 1
        if person[GENDER] == '0': # female
            if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # young, predict survive
                correct_count += 1
            elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
                correct_count += 1
    
        else: # male
            if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # young, predict survive
                correct_count += 1
            elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
                correct_count += 1

    return round(1 - correct_count / total_count, 3)

# split on age first
def age_gender_impurity(age=25, sign=le):
    data = read_csv(filename=DATA)
    correct_count = total_count = 0
    for person in data[1:]:
        total_count += 1
        if sign(int(person[AGE]), age): # less than 25
            if person[GENDER] == '0' and person[SURVIVED] == '1': # female, predict survive
                correct_count += 1
            elif person[GENDER] == '1' and person[SURVIVED] == '0':
                correct_count += 1
    
        else: # older than 25
            if person[GENDER] == '0' and person[SURVIVED] == '1': # female, predict survive
                correct_count += 1
            elif person[GENDER] == '1' and person[SURVIVED] == '0':
                correct_count += 1

    return round(1 - correct_count / total_count, 3)

def gini_index_gender():
    data = read_csv(filename=DATA)
    male_count = female_count = 0
    male_survived_count = female_survived_count = 0
    for person in data:
        if person[GENDER] == '0':
            male_count += 1

        else:
            female_count += 1
    total = male_count + female_count
    gini_index = 1 - (male_count / total) ** 2 - (female_count / total) ** 2 # wrong
    return  gini_index

def gini_index_age():
    data = read_csv(filename=DATA)
    

def sklearn_decision_tree(criterion='gini'):
    data = read_csv(filename=DATA)
    inputs = []
    outputs = []
    for person in data[1:]:
        inputs.append(person[:-1])
        outputs.append([person[SURVIVED]])
        
    clf = DecisionTreeClassifier(max_depth=3, random_state=123123, criterion=criterion, splitter='random')
    clf = clf.fit(inputs, outputs)

    plt.figure(figsize=(12,12))
    sklearn.tree.plot_tree(clf, fontsize=12)
    #plt.show() 
    plt.savefig('decision_tree')
    return 

sklearn_decision_tree(criterion='entropy')

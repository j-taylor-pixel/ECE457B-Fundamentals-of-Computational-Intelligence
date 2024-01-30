from operator import le, ge
from sklearn.tree import DecisionTreeClassifier
import sklearn
import matplotlib.pyplot as plt
from helpers.helper import read_csv, splice_data, split_rows
from math import log2

GENDER = 8
AGE = 1
SURVIVED = 9
DATA='/home/josiah/repos/ECE457B-Fundamentals-of-Computational-Intelligence/Assignment1/Exercise4/A1Q3DecisionTrees.csv'

def gender_impurity():
    female_survived_count, female_count, male_not_survived_count, male_count = gender_split_metrics()
    # return wrong predictions / total predictions
    purity = (female_survived_count + male_not_survived_count) / (female_count + male_count)

    return round(1 - purity, 3)# calculate impurity

def age_impurity(age=25, sign=le):
    under_survived_count, under_count, over_not_survived_count, over_count = age_split_metrics(age=age, sign=sign)
    purity = (under_survived_count + over_not_survived_count) / (under_count + over_count)
    return round(1 - purity, 3)

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

def age_gender_impurity(age=25, sign=le):# split on age first

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

def calc_gini_index(b1_l, b1_t, b2_l, b2_t):
    b1_ratio = b1_l / b1_t
    b2_ratio = b2_l / b2_t
    
    gini_index_b1 = 1 - b1_ratio ** 2 - (1 - b1_ratio) ** 2
    gini_index_b2 = 1 - b2_ratio ** 2 - (1 - b2_ratio) ** 2
    
    total = b1_t + b2_t
    gini_index = (b1_t * gini_index_b1 + b2_t * gini_index_b2) / total

    return round(gini_index, 3)

def gender_split_metrics():
    data = read_csv(filename=DATA)
    male_count, female_count, male_not_survived_count, female_survived_count = 0, 0, 0, 0
    for person in data:
        if person[GENDER] == '0':
            female_count += 1
            if person[SURVIVED] == '1':
                female_survived_count += 1
        else:
            male_count += 1
            if person[SURVIVED] == '0':
                male_not_survived_count += 1
    return female_survived_count, female_count, male_not_survived_count, male_count

def gini_index_gender():
    female_survived_count, female_count, male_not_survived_count, male_count = gender_split_metrics()
    return  calc_gini_index(female_survived_count, female_count, male_not_survived_count, male_count)

def age_split_metrics(age=25, sign=le):
    data = read_csv(filename=DATA)
    under_count, over_count, under_survived_count, over_not_survived_count = 0, 0, 0, 0
    for person in data[1:]:
        if sign(int(person[AGE]), age):
            under_count += 1
            if person[SURVIVED] == '1': # predict survive
                under_survived_count += 1
        elif not sign(int(person[AGE]), age):
            over_count += 1
            if person[SURVIVED] == '0':
                over_not_survived_count += 1
    return under_survived_count, under_count, over_not_survived_count, over_count

def gini_index_age(age=25, sign=le):
    under_survived_count, under_count, over_not_survived_count, over_count = age_split_metrics(age=age, sign=sign)
    return calc_gini_index(under_survived_count, under_count, over_not_survived_count, over_count)

def calc_shannon_entropy(b1_l, b1_t, b2_l, b2_t):
    b1_ratio = b1_l / b1_t
    b2_ratio = b2_l / b2_t

    b1_entropy = -1 * b1_ratio * log2(b1_ratio) - (1 - b1_ratio) * log2(1 - b1_ratio)
    b2_entropy = -1 * b2_ratio * log2(b2_ratio) - (1 - b2_ratio) * log2(1 - b2_ratio)

    shannon_entropy = (b1_t * b1_entropy + b2_t * b2_entropy) / (b1_t + b2_t)
    return round(shannon_entropy, 3)

def entropy_gender():
    female_survived_count, female_count, male_not_survived_count, male_count = gender_split_metrics()
    return calc_shannon_entropy(female_survived_count, female_count, male_not_survived_count, male_count)

def entropy_age(age=25, sign=le):
    under_survived_count, under_count, over_not_survived_count, over_count = age_split_metrics(age=age, sign=sign)
    return calc_shannon_entropy(under_survived_count, under_count, over_not_survived_count, over_count)

def sklearn_decision_tree(criterion='gini', splitter='best'): # criterion can also be 'entropy', used solely for question 2oo  
   data = read_csv(filename=DATA, trim_header=True)
   inputs = []
   outputs = []
   for person in data:
       inputs.append(person[:-1])
       outputs.append([person[SURVIVED]])
      
   clf = DecisionTreeClassifier(max_depth=3, random_state=123123, criterion=criterion, splitter=splitter)
   clf = clf.fit(inputs, outputs)

   plt.figure(figsize=(12,12))
   sklearn.tree.plot_tree(clf, fontsize=9)
   #plt.show() # doesnt work without interactive
   plt.savefig('Assignment1/Exercise4/decision_tree_q2')
   return

def perf_of_decision_tree(criterion='gini', train_size=80, val_size=10, test_size=10, max_depth=3):
    data = read_csv(filename=DATA, trim_header=True)
    train_count, val_count, test_count = splice_data(num_rows=len(data),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(data, train_count, val_count, test_count)
    
    train_inputs, train_outputs = [], []
    for person in train_rows:
       train_inputs.append(person[:-1])
       train_outputs.append([person[SURVIVED]])
      
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=123123, criterion=criterion)
    clf = clf.fit(train_inputs, train_outputs)

    val_correct, test_correct = 0, 0
    for person in val_rows:
        if clf.predict([person[:-1]]) == [person[-1]]:
            val_correct += 1
    for person in test_rows:
        if clf.predict([person[:-1]]) == [person[-1]]:
            test_correct += 1

    return round(val_correct / val_count, 3), round(test_correct / test_count, 3)


def varied_max_depth():
    data_splits = [[80,10,10]]
    max_depths = [1, 3, 5, 7, 9, 11]
    for max_depth in max_depths:
        for data_split in data_splits:
            val_perf, test_perf = perf_of_decision_tree(train_size=data_split[0], val_size=data_split[1],test_size=data_split[2], max_depth=max_depth)
            print(f"For {max_depth}, and data split {data_split}, val_perf is {val_perf}, test_perf is {test_perf}")

def data_split_decision_trees():
    criterions = ['gini', 'entropy']
    data_splits = [[80,10,10],[34,33,33],[25,25,50]]
    for criterion in criterions:
        for data_split in data_splits:
            val_perf, test_perf = perf_of_decision_tree(criterion=criterion, train_size=data_split[0], val_size=data_split[1],test_size=data_split[2])
            print(f"For {criterion}, and data split {data_split}, val_perf is {val_perf}, test_perf is {test_perf}")
 
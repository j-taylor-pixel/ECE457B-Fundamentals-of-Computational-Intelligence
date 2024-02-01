from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from helpers.helper import read_csv, splice_data, split_rows
import matplotlib.pyplot as plt
import numpy as np

DATA='/home/josiah/repos/ECE457B-Fundamentals-of-Computational-Intelligence/Assignment1/Exercise4/A1Q3DecisionTrees.csv'
DECISION_TREE_COUNT = 100
SURVIVED = 9

def make_hist(title="Validation", results=[]):
    plt.hist(results, bins=5, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel(f"{title} Performance")
    plt.ylabel('Frequency')
    plt.title(f"{title} of 100 Decision Trees")
 
    # Display the plot
    #plt.show()
    plt.savefig(f"Assignment1/Exercise5/{title} Performance")
    plt.close()
    return

def split_train_data(train_rows):
    train_inputs, train_outputs = [], []
    for person in train_rows:
       train_inputs.append(person[:-1])
       train_outputs.append([person[SURVIVED]])
    return train_inputs, train_outputs

def count_correct_entries(rows, clf):
    correct = 0
    for person in rows:
        if clf.predict([person[:-1]]) == [person[-1]]:
            correct += 1
    return correct

def ensemble_accuracy(train_inputs, train_outputs, rows, name):
    # check accuracy of total ensemble with majority voting
    num_correct = 0
    for person in rows:
        votes = 0
        for _ in range(DECISION_TREE_COUNT):
            # cast 100 votes,
            clf = DecisionTreeClassifier()
            clf = clf.fit(train_inputs, train_outputs)
            votes += int(clf.predict([person[:-1]])[0]) # increment 1 if survive or 0
        if round(votes / DECISION_TREE_COUNT)  == int(person[SURVIVED]): # try reverting and see if val increasesandrea
            num_correct += 1

    print(f"Accuracy of {name} is {round(num_correct / len(rows), 3)}")
    return 


def bagging_ensemble(train_size=80, val_size=10, test_size=10):
    data = read_csv(filename=DATA, trim_header=True)
    train_count, val_count, test_count = splice_data(num_rows=len(data),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(data, train_count, val_count, test_count)

    train_inputs, train_outputs = split_train_data(train_rows=train_rows)
    
    val_results, test_results = [], []
    for _ in range(DECISION_TREE_COUNT):
        clf = DecisionTreeClassifier()
        clf = clf.fit(train_inputs, train_outputs)

        val_correct = count_correct_entries(val_rows, clf)
        test_correct = count_correct_entries(test_rows, clf)

        val_results.append(round(val_correct / val_count, 8))
        test_results.append(round(test_correct / test_count, 8))
    # make histogram of accurracy of all classifiers
    make_hist(title="Validation", results=val_results)
    make_hist(title="Test", results=test_results)

    ensemble_accuracy(train_inputs=train_inputs, train_outputs=train_outputs, rows=val_rows, name="Validation")
    ensemble_accuracy(train_inputs=train_inputs, train_outputs=train_outputs, rows=test_rows, name="Test")
    return

def explore_various_hyperparameters():
    criterions = ['gini', 'entropy']
    

    return

def adaptive_boosting(train_size=80, val_size=10, test_size=10):
    data = read_csv(filename=DATA, trim_header=True)
    train_count, val_count, test_count = splice_data(num_rows=len(data),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(data, train_count, val_count, test_count)

    train_inputs, train_outputs = split_train_data(train_rows=train_rows)
    train_outputs = np.ravel(train_outputs)
    estimators = [None, DecisionTreeClassifier(), RandomForestClassifier()]
    for estimator in estimators:
        clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0, estimator=estimator)
        clf.fit(train_inputs, train_outputs)
        val_correct = count_correct_entries(val_rows, clf)
        test_correct = count_correct_entries(test_rows, clf)
        print(f"For: {estimator}, Val: {val_correct}, test: {test_correct}")

    return

def gradient_boosting():
    return

def xg_boost():
    return


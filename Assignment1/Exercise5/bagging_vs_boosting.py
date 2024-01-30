from sklearn.tree import DecisionTreeClassifier
from helpers.helper import read_csv, splice_data, split_rows
import matplotlib.pyplot as plt

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

def bagging_ensemble(train_size=80, val_size=10, test_size=10):
    data = read_csv(filename=DATA, trim_header=True)
    train_count, val_count, test_count = splice_data(num_rows=len(data),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(data, train_count, val_count, test_count)

    # split train data
    train_inputs, train_outputs = [], []
    for person in train_rows:
       train_inputs.append(person[:-1])
       train_outputs.append([person[SURVIVED]])
    
    val_results, test_results = [], []
    for _ in range(DECISION_TREE_COUNT):
        clf = DecisionTreeClassifier()
        clf = clf.fit(train_inputs, train_outputs)
        val_correct, test_correct = 0, 0
        for person in val_rows:
            if clf.predict([person[:-1]]) == [person[-1]]:
                val_correct += 1
        for person in test_rows:
            if clf.predict([person[:-1]]) == [person[-1]]:
                test_correct += 1
        val_results.append(round(val_correct / val_count, 8))
        test_results.append(round(test_correct / test_count, 8))
    # make histogram of accurracy of all classifiers
    make_hist(title="Validation", results=val_results)
    make_hist(title="Test", results=test_results)
    # take a vote somehow
        

    return # overall accurracy

def adaptive_boosting():
    return

def gradient_boosting():
    return

def xg_boost():
    return


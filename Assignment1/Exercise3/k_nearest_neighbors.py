from enum import Enum
from helpers.helper import read_csv, splice_data, split_rows
import matplotlib.pyplot as plt

DATA='/home/josiah/repos/ECE457B-Fundamentals-of-Computational-Intelligence/Assignment1/Exercise3/A1Q4NearestNeighbors.csv'
    # column attributes are the following
    #   1. Number of times pregnant
    #   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    #   3. Diastolic blood pressure (mm Hg)
    #   4. Classifier, where 1 is positive diabetes

class DistanceTypes(str, Enum):
    cosine = 'cosine'
    euclidian = 'euclidian'

def cosine_similarity(a,b):
    num = denA = denB = 0
    for i in range(len(a)):
        num += a[i] * b[i]
        denA += a[i] ** 2
        denB += b[i] ** 2
    den = (denA ** 0.5) * (denB ** 0.5)
    return num / den

def euclidian_distance(x,y):
    total = 0
    for i in range(len(x)):
        total += (x[i] - y[i]) ** 2

    return total ** 0.5

def string_to_float(rows):
    return [[float(y) for y in x] for x in rows]
    
def performance(k, test_rows, train_rows, metric):
    # loop thru test data
    correct_count = wrong_count = 0
    for test_row in test_rows:
        distances_classifiers = []
        for train_row in train_rows:
            # dont use last element which is the classifier
            cos_similiarity = cosine_similarity(test_row[:-1], train_row[:-1])
            distances_classifiers.append([cos_similiarity, train_row[-1]])
        sorted_data = sorted(distances_classifiers, key=lambda x: x[0]) # sort in ascending
        if metric == DistanceTypes.cosine:
            k_closest = sorted_data[-k:] # k highest closest points
        else: # euclidian
            k_closest = sorted_data[:k] # k lowest distances in euclidian
        # take majority vote of classifiers in k closest
        total = 0 
        
        for element in k_closest:
            total += element[1] # add up classifiers. total will be averaged to do majority vote
        # compare using majority vote with test data classifier
        if ((total / k) > 0.5 and test_row[3] == 1) or ((total / k) < 0.5 and test_row[3] == 0): 
            correct_count += 1
        else:
            wrong_count += 1
    return correct_count / (correct_count + wrong_count) # return accuracy

def k_nearest_neighbors(k=1, metric=DistanceTypes.cosine, train_size=80, val_size=10, test_size=10):

    rows = read_csv(filename=DATA)
    rows = string_to_float(rows)
    train_count, val_count, test_count = splice_data(num_rows=len(rows),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(rows, train_count, val_count, test_count)

    val_perf = performance(k=k,test_rows=val_rows,train_rows=train_rows, metric=metric)
    test_perf = performance(k=k,test_rows=test_rows,train_rows=train_rows, metric=metric)
    return round(val_perf, 3), round(test_perf, 3)

def exercise_3_question_2a_b_c():
    # for each metric
    metrics = [DistanceTypes.cosine, DistanceTypes.euclidian]
    data_splits = [[80,10,10],[34,33,33],[25,25,50]]
    k_values = [1]
    for metric in metrics:
        for data_split in data_splits:
            for k in k_values:
                val_perf, test_perf = k_nearest_neighbors(k=k, metric=metric, train_size=data_split[0], val_size=data_split[2], test_size=data_split[1])
                print(f"Results: metric: {metric}, data split: {data_split}, k: {k}, val_perf: {val_perf}, test_perf: {test_perf}")
    return   

def exercise_3_question_3():
    # for each metric
    metrics = [DistanceTypes.cosine, DistanceTypes.euclidian]
    #metrics = [DistanceTypes.cosine]
    data_splits = [[80,10,10],[34,33,33],[25,25,50]]
    k_values = [1,3,5,11]
    results_x = []
    results_y = []
    colors=['r','g','b','k', 'r','g','b','k', 'r','g','b','k']
    plt.figure()

    for metric in metrics:
        for data_split in data_splits:
            split_results_x = [] 
            split_results_y = []
            for k in k_values:
                val_perf, test_perf = k_nearest_neighbors(k=k, metric=metric, train_size=data_split[0], val_size=data_split[2], test_size=data_split[1])
                split_results_x.append(k)
                split_results_y.append(val_perf)
                #print(f"Results: metric: {metric}, data split: {data_split}, k: {k}, val_perf: {val_perf}, test_perf: {test_perf}")
            
            legend = f"{metric},{data_split}"
 
            plt.plot(split_results_x,split_results_y, label=legend)
            plt.legend()
            results_x.append(split_results_x)
            split_results_x = []
            results_y.append(split_results_y)
            split_results_y = []
    # plot performance on y axis
    # k should be on x axis        
    plt.savefig('Performance across multiple distance methods and k sizes')

    return   
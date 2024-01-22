import csv
import matplotlib.pyplot as plt
import math
from enum import Enum

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

def read_csv(filename='A1Q4NearestNeighbors.csv'):
    file = open(filename)
    #type(file)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        #rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        rows.append(row)
    file.close()
    return rows

def string_to_float(rows):
    return [[float(y) for y in x] for x in rows]

# consider making this return index counts
def splice_data(num_rows=700, train_size=80, val_size=10, test_size=10):
    train_count = math.ceil(num_rows * train_size / 100)
    val_count = math.ceil(num_rows * val_size / 100)
    test_count = num_rows - train_count - val_count
    return train_count, val_count, test_count
    
def split_rows(rows, train_count, val_count, test_count):
    train_rows = rows[:train_count]
    val_rows = rows[train_count:train_count + val_count]
    test_rows = rows[train_count + val_count:]
    return train_rows, val_rows, test_rows
    

def k_nearest_neighbors(k=1, metric=DistanceTypes.cosine, train_size=80, val_size=10, test_size=10):

    rows = read_csv()
    rows = string_to_float(rows)
    train_count, val_count, test_count = splice_data(num_rows=len(rows),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(rows, train_count, val_count, test_count)

    
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

print(k_nearest_neighbors(k=25, metric=DistanceTypes.euclidian))

# data split can start with 80/10/10


    # column attributes are the following
    #   1. Number of times pregnant
    #   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    #   3. Diastolic blood pressure (mm Hg)
    #   4. Classifier, where 1 is positive diabetes


import csv
import matplotlib.pyplot as plt
import math


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
        rows.append(row)
    file.close()
    return rows

# consider making this return index counts
def splice_data(num_rows=700, train_size=80, val_size=10, test_size=10):
    train_count = math.ceil(num_rows * train_size / 100)
    val_count = math.ceil(num_rows * val_size / 100)
    test_count = num_rows - train_count - val_count
    return train_count, val_count, test_count
    
def split_rows(rows, train_count, val_count, test_count): #todo: test this
    train_rows = rows[:train_count - 1]
    val_rows = rows[train_count:val_count-1]
    test_rows = rows[val_count:]
    return train_rows, val_rows, test_rows
    

def k_nearest_neighbors(k=1, metric="cosine", train_size=80, val_size=10, test_size=10):

    rows = read_csv()
    train_count, val_count, test_count = splice_data(num_rows=len(rows),train_size=train_size, val_size=val_size, test_size=test_size)
    train_rows, val_rows, test_rows = split_rows(rows, train_count, val_count, test_count)

    k_current_nearest_neighbors = [-1] # [[sim, classifier],[]] 2xk array 
    # should i make train_rows, test_rows, and val_rows? yes that sounds useful
    if metric == 'cosine': # look for maximized similiarities
        # loop thru test data
        for test_row in test_rows:
            for train_row in train_rows:
                # dont use last element which is the classifier
                cos_similiarity = cosine_similarity(test_row[:-1], train_row[:-1])
                #if cos_similiarity <

        # each test data loops thru train data and calcs cos similiarity
        # sort cos simliiarity and chose K highest
        # look at classifier of k highest, and take a majority vote
        # compare with test classifier
        # record error rate and return

    elif metric == 'euclidian': # look for minimized distances
        return

    return

# data split can start with 80/10/10

#KNearestNeighbors()

    # column attributes are the following
    #   1. Number of times pregnant
    #   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    #   3. Diastolic blood pressure (mm Hg)
    #   4. Classifier, where 1 is positive diabetes


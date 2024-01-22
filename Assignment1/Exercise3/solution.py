import csv
import matplotlib.pyplot as plt
import math


def CosineSimilarity(a,b):
    num = denA = denB = 0
    for i in range(len(a)):
        num += a[i] * b[i]
        denA += a[i] ** 2
        denB += b[i] ** 2
    den = (denA ** 0.5) * (denB ** 0.5)
    return num / den

def EuclidianDistance(x,y):
    total = 0
    for i in range(len(x)):
        total += (x[i] - y[i]) ** 2

    return total ** 0.5 # square root of ans

def readCSV():
    file = open('A1Q4NearestNeighbors.csv')
    #type(file)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows

# 
def SpliceData(num_rows=700, train_size=80, val_size=10, test_size=10):
    train_count = math.ceil(num_rows * train_size / 100)
    val_count = math.ceil(num_rows * val_size / 100)
    test_count = num_rows - train_count - val_count
    return train_count, val_count, test_count
    

def KNearestNeighbors(k=1, metric="cosine", train_size=80, val_size=10, test_size=10):
    
    # data is rows of ['114', '38.1', '21', '0'], where the last element is 1 for positive diabetes
    # other attributes are the following
    #   1. Number of times pregnant
    #   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    #   3. Diastolic blood pressure (mm Hg)

    # i should have a 3d plot, where each point plotted is either a 1 or a 0

    # how to select which points? just use first 80 for training?
    rows = readCSV()

    if metric == 'cosine': # look for maximized similiarities

    elif metric == 'euclidian': # look for minimized distances


    return

# data split can start with 80/10/10

#KNearestNeighbors()

print(CosineSimilarity([1,1],[1,1]))
import math
import csv

def read_csv(filename='Assignment1/Exercise3/A1Q4NearestNeighbors.csv'):
    file = open(filename)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        #rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        rows.append(row)
    file.close()
    return rows

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

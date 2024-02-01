from helpers.helper import read_csv, splice_data, split_rows

DATA = "/home/josiah/repos/ECE457B-Fundamentals-of-Computational-Intelligence/Assignment1/Exercise6/A1Q6RawData.csv"

def compare_accuracy_of_different_models():
    data = read_csv(filename=DATA)
    train_count, val_count, test_count = splice_data(num_rows=len(data), train_size=80, val_size=10, test_size=10)
    train_rows, val_rows, test_rows = split_rows(rows=data, train_count=train_count, val_count=val_count, test_count=test_count)


    return
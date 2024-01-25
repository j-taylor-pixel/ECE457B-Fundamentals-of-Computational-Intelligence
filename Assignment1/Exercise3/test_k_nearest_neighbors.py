from .k_nearest_neighbors import euclidian_distance, cosine_similarity
from helpers.helper import splice_data, read_csv, split_rows

# evoke this from home directory with python -m pytest
# since calling pytest directly just causes modulenotfounderror

def test_euclidian_distance():
    assert euclidian_distance([1,1,1,1],[2,2,2,2]) == 2

def test_cosine_similarity():
    assert abs(cosine_similarity([1,1],[0,1]) - 0.7071) < 0.01

def test_splice_data():
    train_count, val_count, test_count = splice_data(num_rows=768)
    assert train_count == 615 and val_count == 77 and test_count == 76


def test_read_csv():
    rows = read_csv()
    assert len(rows) == 768

def test_split_rows():
    rows = read_csv()
    train_count, val_count, test_count = splice_data(num_rows=len(rows))
    train_rows, val_rows, test_rows = split_rows(rows, train_count, val_count, test_count)
    assert len(train_rows) == train_count and len(val_rows) == val_count and len(test_rows) == test_count
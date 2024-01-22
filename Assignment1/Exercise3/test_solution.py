import solution

def test_euclidian_distance():
    # 
    assert solution.euclidian_distance([1,1,1,1],[2,2,2,2]) == 2

def test_cosine_similarity():
    assert abs(solution.cosine_similarity([1,1],[0,1]) - 0.7071) < 0.01

def test_splice_data():
    assert solution.splice_data()
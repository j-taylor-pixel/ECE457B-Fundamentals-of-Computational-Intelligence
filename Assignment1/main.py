from Exercise2.sample_plot import plot, manhattan_distance, euclidian_distance
from Exercise3.k_nearest_neighbors import exercise_3_question_2a_b_c, exercise_3_question_3
from Exercise4.decision_trees import gender_impurity, compare_age_impurity, gender_age_impurity, age_gender_impurity, gini_index_gender, gini_index_age, sklearn_decision_tree, entropy_gender, entropy_age, data_split_decision_trees, varied_max_depth
from operator import le, ge


plot(distance_measurement=euclidian_distance) # exercise 2
plot(distance_measurement=manhattan_distance) # exercise 2

#exercise_3_question_2a_b_c() 
#exercise_3_question_3()

#print(gender_impurity()) # exercise 4, question 1(a)
#compare_age_impurity() # exercise 4, question 1(c)
#print(gender_age_impurity(age=65, sign=ge)) # exercise 4, questions 1(d)
#print(age_gender_impurity(age=65, sign=ge)) # exercise 4, questions 1(d)
#print(gini_index_gender()) # exercise 4, question 1ei
#print(gini_index_age()) # exercise 4, question 1ei
#print(gini_index_age(age=65, sign=ge)) # question 1ei
#print(entropy_gender()) # exercise 4, question 1eii
#print(entropy_age(), entropy_age(age=65, sign=ge)) # exercise 4, question 1eii
#sklearn_decision_tree() # exercise 4, question 2a
#sklearn_decision_tree(criterion='entropy') # exercise 4, question 2b
#sklearn_decision_tree(splitter='random') # exercise 4, question 2c
#data_split_decision_trees() # exercise 4, question 3
#varied_max_depth()# exercise 4, question 3




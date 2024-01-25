import csv
from operator import le, ge
from sklearn.tree import DecisionTreeClassifier

# consider adding constants to refer to headers index
GENDER = 8
AGE = 1
SURVIVED = 9
DATA = './A1Q3DecisionTrees.csv'

def read_csv(filename=DATA):
    file = open(filename)
    #type(file)
    csvreader = csv.reader(file)
    persons = []
    for person in csvreader:
        persons.append(person)
    file.close()
    return persons

def calculate_impurity_of_only_gender_splitting_tree():
    data = read_csv()

    female_count = 0
    survived_female_count = 0
    male_count = 0
    deceased_male_count = 0

    for person in data[1:]:
        if person[GENDER] == '0': # check if female
            female_count += 1
            if person[SURVIVED] == '1': # check if survived
                survived_female_count += 1
        else:
            male_count += 1
            if person[SURVIVED] == '0':
                deceased_male_count += 1
    # return wrong predictions / total predictions
    return 1 - (survived_female_count + deceased_male_count) / (female_count + male_count) # calculate impurity

def age_impurity(age=25, sign=le):
    data = read_csv()
    correct_count = 0
    total_count = 0
    for person in data[1:]:
        total_count += 1
        if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # predict survive
            correct_count += 1
        elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
            correct_count += 1

    return 1 - (correct_count / total_count)

def compare_age_impurity():
    print(f"Impurity at boundary=25 is {age_impurity(25)}, impurity at boundary=65 is {age_impurity(65, sign=ge)}")
    return

def gender_age_impurity(age=25, sign=le):
    # gender split at first level
    # age split at over 25/65 second level
    data = read_csv()
    correct_count = total_count = 0
    for person in data[1:]:
        total_count += 1
        if person[GENDER] == '0': # female
            if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # young, predict survive
                correct_count += 1
            elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
                correct_count += 1
    
        else: # male
            if sign(int(person[AGE]), age) and person[SURVIVED] == '1': # young, predict survive
                correct_count += 1
            elif not sign(int(person[AGE]), age) and person[SURVIVED] == '0':
                correct_count += 1

    return 1 - correct_count / total_count

# split on age first
def age_gender_impurity(age=25, sign=le):
    data = read_csv()
    correct_count = total_count = 0
    for person in data[1:]:
        total_count += 1
        if sign(int(person[AGE]), age): # less than 25
            if person[GENDER] == '0' and person[SURVIVED] == '1': # female, predict survive
                correct_count += 1
            elif person[GENDER] == '1' and person[SURVIVED] == '0':
                correct_count += 1
    
        else: # older than 25
            if person[GENDER] == '0' and person[SURVIVED] == '1': # female, predict survive
                correct_count += 1
            elif person[GENDER] == '1' and person[SURVIVED] == '0':
                correct_count += 1


    return 1 - correct_count / total_count

def gini_index_gender():
    data = read_csv()
    male_count = female_count = 0
    male_survived_count = female_survived_count = 0
    for person in data:
        if person[GENDER] == '0':
            male_count += 1

        else:
            female_count += 1
    total = male_count + female_count
    gini_index = 1 - (male_count / total) ** 2 - (female_count / total) ** 2 # wrong
    return  gini_index

#print(gini_index_gender())



X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)

print(clf.predict([[2., 2.]]))

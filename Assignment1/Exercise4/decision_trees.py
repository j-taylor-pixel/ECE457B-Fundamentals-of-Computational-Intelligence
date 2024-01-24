import csv


def read_csv(filename='Assignment1/Exercise4/A1Q3DecisionTrees.csv'):
    file = open(filename)
    #type(file)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        #rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        rows.append(row)
    file.close()
    return rows

def impurity_of_gender_splitting_tree():
    data = read_csv(filename='./A1Q3DecisionTrees.csv')

    female_count = 0
    survived_female_count = 0
    for row in data:
        if row[8] == '0': #check if female
            female_count += 1
            if row[9] == '1': #cehck if survived
                survived_female_count += 1

    print(f"females survived is: {survived_female_count}")
    print(f"total females is: {female_count}")
    print(f"ratio is { survived_female_count / female_count}")
    return 1 - survived_female_count / female_count


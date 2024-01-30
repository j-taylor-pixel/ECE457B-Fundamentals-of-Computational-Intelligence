import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

NUM_POINTS = 100 # should be 100 but change to 5 for now

def generate_dims(): # [1,2,4,8 ... 1024]
    dims = [] 
    # generate dims programatically
    for i in range(0,11): # should be 0 to 11, but shorten for testing
        dims.append(2 ** i)
    return dims

def sample_hundred_points(dimensions=1):
    hundred_points = []
    for _ in range(NUM_POINTS): # add 100 d dimensional points
        d_dim_point = [] # d dimension point
        for _ in range(dimensions): # add a dim dimenseional point to current point
            d_dim_point.append(np.random.uniform(0.0,1.0)) 
            # now d_dim_point has 100 points sampled
            # after adding 100 points, we proccess now
        hundred_points.append(d_dim_point)
    return hundred_points

def euclidian_distance(x, y):
    tot = 0
    for dim in range(len(x)):
        tot += (x[dim] - y[dim]) ** 2
    return sqrt(tot)

def manhattan_distance(x, y):
    tot = 0
    for dim in range(len(x)):
        tot += abs(x[dim] - y[dim])
    return tot 

def measure_distance(hundred_points, distance_measurement=euclidian_distance):
    data = []
    for i_x, point_x in enumerate(hundred_points):
        # do (a) squared eculdian distance, (x-y)^2
        for i_y, point_y in enumerate(hundred_points): # loop thru each pair of points
            if i_x != i_y:
                tot = distance_measurement(x=point_x, y=point_y)
                data.append(tot)
    return data

def plot(distance_measurement=euclidian_distance): # metric can also be manhattan
    dims = generate_dims()

    mean_per_dim, var_per_dim = [], [] # metrics for slope
    all_averages, all_variance = [], []

    # generate data
    for d in dims: 
        hundred_points = sample_hundred_points(dimensions=d)
        distances_between_points = measure_distance(hundred_points=hundred_points, distance_measurement=distance_measurement)

        average = sum(distances_between_points) / len(distances_between_points)
        variance = np.var(distances_between_points)
        all_averages.append(average)
        all_variance.append(variance)
        mean_per_dim.append(average / d)
        var_per_dim.append(variance / d)

        #print(f"Average is: {average}, var is: {variance} with {d} dimensions")
    
    # plot average and standard deviation on y-axis vs dimensions on x-axis
    plt.figure()
    plt.plot(dims, all_averages, label='mean')
    plt.plot(dims, all_variance, label='variance')
    plt.legend()

    plt.savefig(f"Assignment1/Exercise2/Mean & var across different dimensions using {distance_measurement.__name__}")
    
    # by analysis, there is a linear relationship between mean and dimensions, and variance and dimensions for manhattan distance
    # calculate slope below
    print(f"For {distance_measurement.__name__}, mean slope: {np.average(mean_per_dim)}, Var slope: {np.average(var_per_dim)}")

    return
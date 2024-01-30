import numpy as np
import matplotlib.pyplot as plt


NUM_POINTS = 100 # should be 100 but change to 5 for now

def plot():
    dims = [] # [1,2,4,8 ... 1024]
    # create dims to loop thru
    # generate dims programatically
    for i in range(0,11): # should be 0 to 11, but shorten for testing
        dims.append(2 ** i)
    print(dims)

    mean_per_dim = []
    var_per_dim = []

    # generate data
    all_averages, all_variance = [], []
    for d in dims:
        hundred_points = []
        for point in range(NUM_POINTS): # add 100 d dimensional points
            d_dim_point = [] # d dimension point
            for dimension in range(d): # add a dim dimenseional point to current point
                d_dim_point.append(np.random.uniform(0.0,1.0)) 
        # now points has num_dim points sampled
            # after adding 100 points, we proccess now
            hundred_points.append(d_dim_point)
        #print(f"hundred points: {hundred_points}")
        # now we are in a dimension(2), with 100 points
        data = []
        for point_x in hundred_points:
            # do (a) squared eculdian distance, (x-y)^2
            for point_y in hundred_points: # loop thru each pair of points
                tot = 0
                for dim in range(len(point_x)):
                    tot += (point_x[dim] - point_y[dim]) ** 2
                if tot != 0: # zero tot if same point distance from itself, dont append these datas
                    data.append(tot)
        average = sum(data) / len(data)
        variance = np.var(data)
        all_averages.append(average)
        all_variance.append(variance)
        mean_per_dim.append(average / d)
        var_per_dim.append(variance / d)

        print(f"Average is: {average}, var is: {variance} with {d} dimensions")
    
    # plot average and standard deviation on y-axis vs dimensions on x-axis
    plt.figure()
    plt.plot(dims, all_averages, label='mean')
    plt.plot(dims, all_variance, label='variance')
    plt.legend()

    plt.savefig('Mean & variance across different dimensions')
    
    # by analysis, there is a linear relationship between mean and dimensions, and variance and dimensions
    # calculate slope below
    print(f"Mean slope: {np.average(mean_per_dim)}, Var slope: {np.average(var_per_dim)}")

    return
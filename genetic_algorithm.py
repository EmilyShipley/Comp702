import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import signal_estimator
import test_framework

# Define the parameters for the genetic algorithm
algorithm_param = {'max_num_iteration': 200,\
                   'population_size':100,\
                   'mutation_probability':0.4,\
                   'elit_ratio': 0.0,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':50}

# Load in a calibration process
PATH = 'C:/Users/Emily/Desktop/Uni/MSc Project/parsed_data/3942_20210120_1338_coarsecal.txt'
cal_process = np.loadtxt(PATH, dtype=float, delimiter= " ")

# Looking at the signals corresponding to the first phase, locate rows in need of improvement and store their indices in a list
# This is so that the genetic algorithm doesn't need to improve on all 2304 sets of coordinates - more computationally efficient
indices = []
for row in range(2304):
    if cal_process[row,6]<-3:
        indices.append(row)
# Take those rows out of the dataframe to run a genetic algorithm on 
df = cal_process[indices,:]
print(len(indices)) # for calibration process 3942_20210120_1338, 1084 sets of coordinates require imrovement - 1084*4=4336 variables - likely too many for a GA to be suitable
# test on the first few rows first
#df = df[:10,:]

# Print the error that the first phase has for these indices
print("Error for first phase coordinates:",test_framework.test(df[:,2:6],df,closeness='mean')) # this is 485.62 for the whole df, 564.96 for the first 10 rows

# Define a function for the genetic algorithm to optimise (This simply makes the genetic algorithm compute the error value for the coordinates
# it is considering, using the calibration data we loaded in with only the coordinates in need of improvement selected)
def F(c):
    coords = np.reshape(c,(-1,4))
    return test_framework.test(coords,df,closeness='mean')

# Set the bounds for each coordinate value to -10000 and 10000
varbound=np.array([[-10000,10000]]*4*len(indices))
# Set up and run the genetic algorithm 
model=ga(function=F,dimension=4*len(indices),variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param)
model.run()
sol = model.output_dict['variable']
print(sol)

# Error for the first 10 rows comes out at 4999.24 - very far from a good calibration
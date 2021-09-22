import numpy as np


def Euclidean(a,b):
    """Computes and returns the Euclidean distance between vectors a and b."""
    return np.linalg.norm(a-b)


def cosSim(a,b):
    """Computes and returns the cosine similarity between vectors a and b."""
    return np.dot(a,b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))


def test(alg_output,cal_process,phase=None,dist_function=Euclidean,closeness='max',error_function='max'):
    """ Takes a set of coordinates for an optical switch (two pairs of (x,y) coordinates for each input-output
        pair of ports in the switch) and compares these with the coordinates from a single calibration process to
        compute and return an error value for the given coordinates. This error measures the difference between the
        calibration process' coordinates and the coordinates being tested, so a large error may not necessarily
        indicate that the coordinates are 'bad'.

    PARAMETERS: 
            'alg_output' is the array containing the coordinates the algorithm has output.
            'cal_process' is the file containing the calibration data for the calibration process being used.
            'phase' allows the user to choose a specific phase of the calibration process for comparison.
            E.g. if phase=1 the function will compare the results with the coordinates from phase 1,
            if phase=None the function will compare the results with the coordinates that gave the best 
            signal throughout the entire calibration process.
            'dist_function' allows the user to specify which distance function will be used to measure the distance
            between each of the two pairs of (x,y) coordinates for a pair of ports with the two pairs of (x,y) 
            coordinates belonging to the gold standard.
            The default function is the Euclidean distance, but another option is cosine similarity 'cosSim'.
            'closeness' allows the user to specify how the two distances calculated will be combined into a single 
            measure of closeness. The default is to take the maximum, but the other options are 'sum' and 'mean'.
            'error_function' allows the user to specify the error function to be used. The default is to take the maximum but
            other option is the mean."""
    # Remove port labels
    if np.shape(cal_process)[1] %5  == 2:
        cal_process_ = cal_process[:,2:]
    else:
        cal_process_ = np.copy(cal_process)
    num_phases = int((np.shape(cal_process_)[1])/5)
    num_rows = int(np.shape(cal_process_)[0])

    # Locate the gold standard coordinates from the calibration data
    if phase!=None:
        gold_standard = cal_process_[:,phase*5-5:phase*5-1]

    else:
        # Add the signal from each phase of the calibration process to an array
        signals = cal_process_[:,4:num_phases*5:5]
        # Set the gold standard coordinates as the ones that produced these signals
        gold_standard = np.array([cal_process_[row,np.argmax(signals, axis=1)[row].item()*5:np.argmax(signals, axis=1)[row].item()*5+4] for row in range(num_rows)])

    # For each input-output pair, calculate the distance between the first pair of (x,y) coordinates from both the coordinates being tested 
    # and the gold standard, and the distance between the second pair of (x,y) coordinates from the same two sets of coordinates
    distances = np.array([[dist_function(alg_output[row,0:2],gold_standard[row,0:2]),dist_function(alg_output[row,2:4],gold_standard[row,2:4])] for row in range(num_rows)])

    # For each pair, combine the two distances into a measure of closeness
    if closeness=='max':
        c = np.max(distances,axis=0)
    elif closeness=='sum':
        c = np.sum(distances,axis=0)
    elif closeness=='mean':
        c = np.mean(distances,axis=0)
    # Use the closenesses to compute an overall error for the whole set of coordinates
    if error_function=='max':
        error = np.max(c)
    elif error_function=='mean':
        error = np.mean(c)
   
    return error



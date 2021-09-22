import numpy as np
import test_framework


def Euclidean(a,b):
    """Computes and returns the Euclidean distance between vectors a and b."""
    return np.linalg.norm(a-b)

def cosSim(a,b):
    """Computes and returns the cosine similarity between vectors a and b."""
    return np.dot(a,b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))


def signal_est(alg_output,cal_process,dist_function=Euclidean,closeness='max'):
    """Uses the coordinates and corresponding signals from an optical switch calibration process to estimate
        the signals produced by another set of coordinates.

    PARAMETERS: 
            'alg_output' is the array containing the coordinates the algorithm has output.
            'cal_process' is the file containing the calibration data for the calibration process being used.
            'dist_function' allows the user to specify which distance function will be used to measure the distance
            between each of the two pairs of (x,y) coordinates for a pair of ports with the two pairs of (x,y) 
            coordinates belonging to the gold standard.
            The default function is the Euclidean distance, but another option is cosine similarity 'cosSim'.
            'closeness' allows the user to specify how the two distances calculated will be combined into a single 
            measure of closeness. The default is to take the maximum, but the other options are 'sum' and 'mean'."""
    # Remove the port labels from the calibration process dataframe
    cal_process_ = cal_process[:,2:]

    num_phases = int((np.shape(cal_process_)[1])/5)
    num_rows = int(np.shape(cal_process_)[0])

    signals=np.empty(num_rows)
    distances = np.empty((num_rows,num_phases*2))

    for row in range(num_rows):
        # Calculate the distance between the two pairs of coordinates, and the pair of coordinates from each 
        # phase of the calibration process
        for phase in range(1,num_phases+1):
            distances[row,phase*2-2] = dist_function(alg_output[row,0:2],cal_process_[row,phase*5-5:phase*5-3])
            distances[row,phase*2-1] = dist_function(alg_output[row,2:4],cal_process_[row,phase*5-3:phase*5-1])

        # Use these to compute a measure of closeness between the coordinates and each phase
        if closeness=='max':
            c = np.array([[np.max(distances[row,2*k:2*k+2]) for k in range(num_phases)]])
        elif closeness=='sum':
            c = np.array([[np.sum(distances[row,2*k:2*k+2]) for k in range(num_phases)]])
        elif closeness=='mean':
            c = np.array([[np.mean(distances[row,2*k:2*k+2]) for k in range(num_phases)]])

        # Find the minimum closeness
        min_closeness = np.min(c)
        # Select the best signal corresponding to this closeness
        best_signal = np.max(np.array( [cal_process_[row,(k)*5+4] for k in np.where(c == min_closeness)[1]]))
        
        if min_closeness == 0:
            # If the minimum closeness is 0 simply take the exact signal
            signals[row] = best_signal
        else:
            # Otherwise estimate the signal by scaling that of the closest coordinates by both the distance to 
            # these coordinates, and the magnitude of the corresponding signal
            # (It is necessary to scale inversely proportional to the signal as a 'larger' signal will be more skewed
            # by the multiplicative factor than a 'smaller' one, so we want to scale those by a smaller amount)
            signals[row] = np.clip( best_signal * min_closeness**(1/(2.55*abs(best_signal))) ,  -100 , 0 )

    return signals


def signal_est_2(alg_output,cal_process,dist_function=Euclidean):
    """Uses the coordinates and corresponding signals from an optical switch calibration process to estimate
        the signals produced by another set of coordinates.

    PARAMETERS: 
            'alg_output' is the array containing the coordinates the algorithm has output.
            'cal_process' is the file containing the calibration data for the calibration process being used.
            'dist_function' allows the user to specify which distance function will be used to measure the distance
            between each of the two pairs of (x,y) coordinates for a pair of ports with the two pairs of (x,y) 
            coordinates belonging to the gold standard.
            The default function is the Euclidean distance, but another option is cosine similarity 'cosSim'.
            'closeness' allows the user to specify how the two distances calculated will be combined into a single 
            measure of closeness. The default is to take the maximum, but the other options are 'sum' and 'mean'."""
    # Remove the port labels from the calibration process dataframe
    cal_process_ = cal_process[:,2:]
    num_phases = int((np.shape(cal_process_)[1])/5)
    num_rows = int(np.shape(cal_process_)[0])

    signals=np.empty(num_rows)
    distances = np.empty((num_rows,num_phases*2))

    for row in range(num_rows):
        # Calculate the distance between the two pairs of coordinates, and the pair of coordinates from each phase of the calibration process
        for phase in range(1,num_phases+1):
            distances[row,phase*2-2] = dist_function(alg_output[row,0:2],cal_process_[row,phase*5-5:phase*5-3])
            distances[row,phase*2-1] = dist_function(alg_output[row,2:4],cal_process_[row,phase*5-3:phase*5-1])
        # Use these to compute a measure of closeness between the coordinates and each phase
        c = np.array([[np.max(distances[row,2*k:2*k+2]) for k in range(num_phases)]])

        # Find the minimum closeness 
        min_closeness = np.min(c)
        # Select the best signal corresponding to this closeness
        closest_signal = np.max(np.array( [cal_process_[row,(k)*5+4] for k in np.where(c == min_closeness)[1]]))
        
        if min_closeness == 0:
            # If the minimum closeness is 0 simply take the exact signal
            signals[row] = closest_signal
        else:
            # Otherwise locate the coordinates corresponding to the closest signal
            for phase in range(num_phases): 
                if cal_process_[row,phase*5+4] == closest_signal:
                    closest_coords = cal_process_[row,phase*5:phase*5+4]
                    break

            if test_framework.test(np.reshape(alg_output[row,:],(1,-1)),np.reshape(cal_process[row,:],(1,-1))) < test_framework.test(np.reshape(closest_coords,(1,-1)),np.reshape(cal_process[row,:],(1,-1))):
                # If the alg_output coords are CLOSER to the gold standard than the ones they're nearest to, take the estimate closer to 0
                signals[row] = np.clip( closest_signal + min_closeness*(0.02) ,  -100 , 0 )
            else:
                # If the alg_output coords are FURTHER from the gold standard than the ones they're nearest to, take the estimate closer to -100
                signals[row] = np.clip( closest_signal - min_closeness*(0.1) ,  -100 , 0 )

    return signals

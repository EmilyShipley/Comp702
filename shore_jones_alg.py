import test_framework
import numpy as np 
import signal_estimator
import hill_climb


def shore_jones_alg(cal_process,input_phase,num_iters):
    """PARAMETERS:
            cal_process is the whole (parsed) calibration process file to be used in the signal estimation step of the algorithm.
            Note: this needs to include the labels of the two ports at the beginning of each row.
            input_phase is the phase of this calibration process that the algorithm should use as a starting point."""
    num_phases = int((np.shape(cal_process)[1]-2)/5)
    cal_process_ = np.copy(cal_process)
    # 'Correcting' the sign of every other input port's x coordinate and every other output port's x coordinates
    # so that the shifts are calculated correctly
    even_input = np.where(cal_process_[:,0] % 2 == 0)
    odd_output = np.where(cal_process_[:,1] % 2 == 1)
    for phase in range(num_phases):
        cal_process_[even_input,phase*5+2] = cal_process_[even_input,phase*5+2] * -1
        cal_process_[odd_output,phase*5+4] = cal_process_[even_input,phase*5+4] * -1
    # Taking the desired starting phase
    current_coords = np.hstack((cal_process_[:,:2],cal_process_[:,(input_phase-1)*5+2:(input_phase-1)*5+7]))

    print("Number of pairs with no signal at input phase of calibration: ",np.shape(np.where(current_coords[:,-1] == -100))[1])
    for iter in range(num_iters):
        # Create a list of target nodes that have already been shifted to prevent a set of coordinates being shifted more than once
        shifted = []

        for a1 in range(1,49):
            # locating neighbours based on current port's position in the input grid
            if a1%12 == 1:
                adjacent = np.array([a1+1])
            elif a1%12 == 0:
                adjacent = np.array([a1-1])
            else:
                adjacent = np.array([a1-1,a1+1])
            
            # Iterating over each neighbour
            for a2 in adjacent:
                # Iterating over all opposite ports to find some b1 that both a1 and a2 have a connection with
                for b1 in range(49,97):
                    if current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b1)),6]>-100 and current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b1)),6]>-100:
                        # locating possible target nodes based on b1's position in the output grid - these need to be within 2 ports on the same row
                        if b1%12 == 1:
                            b1_adjacent = np.array([b1+1])
                        elif b1%12 == 0:
                            b1_adjacent = np.array([b1-1])
                        #elif b1%12 == 2:
                          #  b1_adjacent = np.array([b1-1,b1+1,b1+2])
                        #elif b1%12 == 11:
                         #   b1_adjacent = np.array([b1-2,b1-1,b1+1])
                        else:
                            b1_adjacent = np.array([b1-1,b1+1])

                        # Iterating over all possible target nodes to locate any where a1 has a connection and a2 does not
                        for b2 in b1_adjacent:
                        #for b2 in range(49,97):
                            if (a2,b2) not in shifted and current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b2)),6]>-100 and current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),6]<=-100:
                                # Calculating shift of b2 for a1 with respect to root node b1, and applying that shift to a2
                                
                                # x0 = a1's x coord for b1
                                x0 = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b1)),2]
                                # y0 = a1's y coord for b1
                                y0 = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b1)),3]
                                # shift x = a1's x coord for b2 - x0
                                shift_x = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b2)),2] - x0
                                # shift y = a1's y coord for b2 - y0
                                shift_y = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b2)),3] - y0
                                # a2's x coord for b2 = a2's x coord for b1 + shift_x
                                current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),2] = current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b1)),2] + shift_x
                                # a2's y coord for b2 = a2's y coord for b1 + shift_y
                                current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),3] = current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b1)),3] + shift_y

                                # Calculating the shift of a2 for b1, with respect to root node a1, and applying that shift to b2

                                # x0 = b1's x coord for a1
                                x0 = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b1)),4]
                                # y0 = b1's y coord for a1
                                y0 = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b1)),5]
                                # shift x = b1's x coord for a2 - x0
                                shift_x = current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),4] - x0
                                # shift y = b1's y coord for a2 - y0
                                shift_y = current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b1)),5] - x0
                                # b2's x coord for a2 = b2's x coord for a1 + shift_x 
                                current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),4] = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b2)),4] +shift_x
                                # b2's y coord for a2 = b2's y coord for a1 + shift_y 
                                current_coords[np.intersect1d(np.where(current_coords[:,0]==a2),np.where(current_coords[:,1]==b2)),5] = current_coords[np.intersect1d(np.where(current_coords[:,0]==a1),np.where(current_coords[:,1]==b2)),5] +shift_y
                      
                                shifted.append((a2,b2))
                     
        shifted.clear()
        # Estimating the signals produced by the current set of coordinates
        current_coords[:,-1] = signal_estimator.signal_est_2(current_coords[:,2:-1],cal_process_)
        print("Number of pairs estimated to have no signal after iteration",iter,": ",np.shape(np.where(current_coords[:,-1] == -100))[1])

    return current_coords

# load in a parsed file
df = np.loadtxt('C:/Users/Emily/Desktop/Uni/MSc Project/parsed_data/3943_20210119_1001_coarsecal.txt', dtype=float, delimiter= " ")

output = shore_jones_alg(df,1,1)

# correcting the signs in the dataframe so that the test framework can accurately measure the error between the df and the Shore-Jones output
corrected_df = np.copy(df)
num_phases = int((np.shape(corrected_df)[1]-2)/5)
even_input = np.where(corrected_df[:,0] % 2 == 0)
odd_output = np.where(corrected_df[:,1] % 2 == 1)
for phase in range(num_phases):
    corrected_df[even_input,phase*5+2] = corrected_df[even_input,phase*5+2] * -1
    corrected_df[odd_output,phase*5+4] = corrected_df[even_input,phase*5+4] * -1

print("Original error:",test_framework.test(corrected_df[:,2:6],corrected_df,closeness='mean')) # 569.93
print("Error after 1 iteration:",test_framework.test(output[:,2:-1],corrected_df,closeness='mean')) # 593.67
output = hill_climb.hill_climb_alg(output,corrected_df,1,100,10)
print("Error after 1 hill-climb iteration:",test_framework.test(output[0][:,2:-1],corrected_df,closeness='mean')) # 550
print("Phase 2's error against gold standard:",test_framework.test(corrected_df[:,7:11],corrected_df,closeness='mean')) # 546.63
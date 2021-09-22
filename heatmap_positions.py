import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

# The path to the file containing the data to be visualised.
PATH = "C:/Users/Emily/Desktop/Uni/MSc Project/parsed_data/3943_20210119_1001_coarsecal.txt"

# The array ("INPUT" or "OUTPUT") to visualise the data with respect to.
ARRAY = "INPUT"

# Create the heatmap for the given phase, in the given space (ax).
def create_heatmap(phase, ax):

    # Remove unneeded rows from other phases from the DataFrame.
    df_ = np.hstack((df[:,:2],df[:,(phase-1)*5+2:phase*5+2]))
    grid = []

    # Construct a nested array of values to populate the heatmap with.  This
    # heatmap includes every input-output pair for the current phase.
    for x in range(min_node, max_node):
        grid.append([])
        for y in range(min_node_2, max_node_2):
            df_i = df_[np.where(df_[:,0] == x)]
            df_o = df_i[np.where(df_i[:,1] == y)]

            grid[-1].append(df_o[:,-1].item())

    # Set the colour palette for the heatmap and create it.
    cmap = sns.diverging_palette(h_neg=10, h_pos=150, s=100, as_cmap=True)
    sns.heatmap(grid, ax=ax, cmap=cmap, vmin=-6, vmax=0, square=True,
                yticklabels=False,
                xticklabels=False, cbar=False)

    ax.set_title(f"Phase {phase}",fontsize='small')


# Get the bounds for the main array and its opposite array.
if ARRAY == "INPUT":
    min_node = 1
    min_node_2 = 49
elif ARRAY == "OUTPUT":
    min_node = 49
    min_node_2 = 1

max_node = min_node + 47
max_node_2 = min_node_2 + 47

# Read the data and get the total number of phases.
df = np.loadtxt(PATH, dtype=float, delimiter= " ")
phase_count = int((np.shape(df)[1] - 2)/5)

# Calculate the required rows and columns, create the subplots, and remove any
# spaces that are not required.
cols = math.ceil(math.sqrt(phase_count))
rows = math.ceil(phase_count / cols)
fig, axs = plt.subplots(nrows=rows, ncols=cols)
if phase_count % cols != 0:
    for c in range(cols - 1, phase_count % cols - 1, -1):
        fig.delaxes(axs[rows - 1][c] if rows > 1 else axs[c])


# Populate the subplots with heatmaps. The conditional statements check how
# many rows and columns there are, as this changes the loops needed.
phase = 1
if rows > 1:
    for row in axs:
        for col in row:
            if phase > phase_count:
                break
            create_heatmap(phase, col)
            phase += 1
elif cols > 1:
    for col in axs:
        if phase > phase_count:
            break
        create_heatmap(phase, col)
        phase += 1
else:
    create_heatmap(phase, axs)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.6)
plt.show()

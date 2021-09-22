import os

# The path to the directory containing the files to be parsed.
# These will all be parsed into a single file, so make sure they are all
# for the same calibration process.
PATH = "C:/Users/Emily/Desktop/Uni/MSc Project/raw_data/20210120_1337_Coarse_Cal/calibration data"

# Parse the relevant data in SPM and power_grid files and save it to a new file.
def parse(dir):
    files = os.listdir(dir)

    # Each phase of the calibration process has its own SPM and power_grid
    # file.  The SPM and power_grid files for a single phase of a single
    # calibration process have the same name ("prefix"), plus the corresponding
    # suffix.  Here, we get all of the unique prefixes by removing the suffixes
    # and taking every other filename (they are in alphabetical order, so
    # every other filename will be a duplicate after removing suffixes).
    prefixes = [x.replace("_SPM.txt", "").replace("_power_grid.txt", "")
                for x in files if files.index(x) % 2 == 0]

    # Construct the filename of the file to be output.
    filename = "_".join(files[0].split("_")[0:3]) + "_coarsecal.txt"
    data = []

    # For each phase, open the SPM and power_grid files, get the relevant
    # data, and add it to the data
    for prefix in prefixes:
        with open(f"{PATH}/{prefix}_SPM.txt", "r") as spm:
            # We're not interested in most of the data in this file, so we
            # get what we need and convert it into a list format.
            data.append([[int(y) for y in x.split(" ")[1:7]]
                         for x in list(spm)[145:2551] if x != "%\n" and x.split(" ")[0]=='%!Con'])


        with open(f"{PATH}/{prefix}_power_grid.txt", "r") as grid:
            # By default, power_grid data is stored as a 2D plaintext array.
            # We change this to a Python list of lists.
            grid_data = [x.strip("\n").split(",") for x in list(grid)[2:]]
            grid_data[0] = grid_data[0][1:]
            grid_data = [[float(y) for y in x] for x in grid_data]

            # Then, we add the data from this list of lists to the final
            # data list which will be saved.
            for y, row in enumerate(grid_data[1:]):
                for x, col in enumerate(row[1:]):
                    data[-1][y * 48 + x].append(col)

    save(data, filename)

# Convert data from a list to a single string and save that string as a file.
# Data from all phases for a single calibration process are stored in the same
# file (data from both SPM and power_grid files).
def save(data, filename):
    content = ""
    for i, row in enumerate(data[0]):
        # Add the node IDs as the start of each line.
        content += f"{row[0]} {row[1]} "

        # Add data for each phase for the current input-output pair.
        for j, phase in enumerate(data):
            if j < len(data) - 1:
                content += f"{phase[i][2]} {phase[i][3]} {phase[i][4]} {phase[i][5]} {phase[i][6]} "
            else:
                content += f"{phase[i][2]} {phase[i][3]} {phase[i][4]} {phase[i][5]} {phase[i][6]}\n"

    with open(f"{PATH}/{filename}", "a") as f:
        f.write(content)

parse(PATH)

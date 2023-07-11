import csv

def save_statistics(statistics_list, filename):
    # Define the header of the CSV file
    header = ['name_of_method', 'elapsed_time', 'loss_value', 'history2D', 'distance_list', 'num_oracle', 'flags', 'nb_parameters', 'stationary_measure']

    # Open the CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header to the CSV file
        writer.writeheader()

        # Write the statistics to the CSV file
        for stats_of_method in statistics_list:
            writer.writerow(stats_of_method)



# This script creates a new CSV file with the given filename and writes the statistics to it. The `csv.DictWriter` class is used to create a writer object that maps dictionaries onto output rows. The `fieldnames` parameter is a sequence of keys that identify the order in which values in the dictionary are written to the CSV file.
#
# You can use this function in your code like this:
#
# ```python
# statistics_list = [...]  # Your list of statistics
# save_statistics(statistics_list, 'statistics.csv')
# ```
#
# This will create a new file named `statistics.csv` in the current directory and write the statistics to it. You can then use any CSV reader to open and analyze the file.
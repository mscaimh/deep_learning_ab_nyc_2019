"""
Script to split data into 80:20 ratio for training and testing.
"""

import csv
import random

def split_csv(input_path, output_training_file_path, output_testing_file_path, training_ratio):
    # Read all rows
    with open(input_path, newline='', encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # If the first row looks like a header, keep it separate
    header = reader[0]
    data_rows = reader[1:]  # everything except header

    # Shuffle rows
    random.shuffle(data_rows)

    # Split index
    split_index = int(len(data_rows) * training_ratio)
    training_rows = data_rows[:split_index]
    testing_rows = data_rows[split_index:]

    # Write training set
    with open(output_training_file_path, 'w', newline='', encoding="utf-8") as f_train:
        writer = csv.writer(f_train)
        writer.writerow(header)
        writer.writerows(training_rows)

    # Write testing set
    with open(output_testing_file_path, 'w', newline='', encoding="utf-8") as f_test:
        writer = csv.writer(f_test)
        writer.writerow(header)
        writer.writerows(testing_rows)

    print(f"Training set saved to: {output_training_file_path}")
    print(f"Testing set saved to:  {output_testing_file_path}")

# Run the program with input
input_path = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019\input\AB_NYC_2019.csv"

output_training_file_path = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019\output\AB_NYC_2019_training.csv"
output_testing_file_path  = r"C:\Users\mahah\git\deep_learning_ab_nyc_2019\output\AB_NYC_2019_testing.csv"

training_ratio = 0.8  # 80% data goes under training set, 20% under testing set

split_csv(input_path, output_training_file_path, output_testing_file_path, training_ratio)

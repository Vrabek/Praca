import csv
from collections import defaultdict
import numpy as np
from scipy.stats import kurtosis, mode

data_csv = 'DATA.csv'
aggregated_data_csv = 'AGGREGATED_DATA.csv'

def calculate_column_average(csv_file, column_index, constraint_column_index, constraint_value, delimiter=',', has_header=True):
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            if has_header:
                next(reader)  # Skip the header row
            values = []
            for row in reader:
                try:
                    value = float(row[column_index])
                    if row[constraint_column_index] == constraint_value:
                        values.append(value)
                except (ValueError, IndexError):
                    pass
            if values:
                average = sum(values) / len(values)
                return average
            else:
                return None
    except IOError:
        print("Error: Failed to read the CSV file.")


def write_to_csv(csv_file='DATA.csv', data=None):
    try:   
        with open(csv_file, 'a') as file:
            writer_object = csv.writer(file)
            writer_object.writerow(data)
            file.close()
    except IOError:
        print("Error: Failed to read the CSV file.")


def calculate_average(csv_file='DATA.csv', filter_column='method', filter_value='random search', columns= ['function_value']):
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)

            data = {column: [] for column in columns}
            for row in reader:
                if row[filter_column] == filter_value:
                    for column in columns:
                        value = float(row[column])
                        data[column].append(value)

        averages = [sum(data[column]) / len(data[column]) for column in columns]

        data = [filter_value] + averages   
        write_to_csv(aggregated_data_csv, data)

        return averages
    except IOError:
        print("Error: Failed to read the CSV file.")
    


def calculate_stats(csv_file='DATA.csv', filter_column='method',calculate_column='function_value'):

    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            group_index = header.index(filter_column)
            value_index = header.index(calculate_column)

            data = defaultdict(list)
            for row in reader:
                if not row:
                    continue
                group = row[group_index]
                value = float(row[value_index])
                data[group].append(value)
    except IOError:
        print("Error: Failed to read the CSV file.")

    try:
        with open('STATS.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Method', 'Średnia Arytmetyczna', 'Odchylenie Standardowe', 'Wariancja', 'Mediana', 'Kurtoza', 'Moda', 'Min', 'Max'])

            for group, values in data.items():
                np_values = np.array(values)
                writer.writerow([
                    group, 
                    np.mean(np_values), 
                    np.std(np_values), 
                    np.var(np_values), 
                    np.median(np_values), 
                    kurtosis(np_values), 
                    mode(np_values).mode[0], 
                    np.min(np_values), 
                    np.max(np_values)
                ])
    except IOError:
        print("Error: Failed to read the CSV file.")


if __name__ == "__main__":
    calculate_stats(data_csv)
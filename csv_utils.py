import csv

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


def write_to_csv(csv_file, data):
    try:   
        with open(csv_file, 'a') as file:
            writer_object = csv.writer(file)
            writer_object.writerow(data)
            file.close()
    except IOError:
        print("Error: Failed to read the CSV file.")


def calculate_average(csv_file, filter_column, filter_value, columns):
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)

            data = {column: [] for column in columns}
            for row in reader:
                if row[filter_column] == filter_value:
                    for column in columns:
                        value = float(row[column])  # Assuming numeric values in the columns
                        data[column].append(value)

        averages = [sum(data[column]) / len(data[column]) for column in columns]

        data = [filter_value] + averages   
        write_to_csv(aggregated_data_csv, data)

        return averages
    except IOError:
        print("Error: Failed to read the CSV file.")
    
import matplotlib.pyplot as plt
import csv

file = 'DATA.csv'
agg_file = 'AGGREGATED_DATA.csv'

def data_to_dict(filename, group_column, value_column):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) 
        group_index = header.index(group_column)
        value_index = header.index(value_column)

        data = {}
        for row in reader:
            if not row:
                continue
            group = row[group_index]
            value = float(row[value_index])
            if group in data:
                data[group].append(value)
            else:
                data[group] = [value]
    
    return data

def draw_boxplot(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    values = list(data.values())
    labels = list(data.keys())
    ax.boxplot(values, labels=labels, vert=0)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    plt.title('search_space = [-10,+10], problem_size = 2, max_iter = 100')
    ax.text(0.5, 1.03, 'Wykres pudełkowy czasu wykonania dla wszystkich metod', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
    plt.yticks(rotation=45)
    plt.xlabel('Czas wykonania [s]')
    plt.show()

def draw_barplot(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=False))
    values_prep = list(sorted_data.values())
    values = [item[0] for item in values_prep]
    labels = list(sorted_data.keys())

    ax.barh(labels, values)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.title('search_space = [-10,+10], problem_size = 2, max_iter = 100')
    ax.text(0.5, 1.03, 'Wykres słupkowy średniego czasu wykoanania dla wszystkich metod', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
    plt.yticks(rotation=45)
    plt.xlabel('Czas wykonania [s]')
    plt.show()

if __name__ == "__main__":
    agg_data = data_to_dict(agg_file, 'method', 'average_time_duration')
    draw_barplot(agg_data)

    data = data_to_dict(file, 'method', 'time_duration')
    draw_boxplot(data)
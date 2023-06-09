import matplotlib.pyplot as plt
import csv
from collections import defaultdict

def read_csv(filename, group_column, value_column):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        group_index = header.index(group_column)
        value_index = header.index(value_column)
        data = defaultdict(list)
        for row in reader:
            if row:
                data[row[group_index]].append(float(row[value_index]))
    return data

filenames = ['AGGREGATED_DATA_PS_2.csv', 'AGGREGATED_DATA_PS_4.csv', 'AGGREGATED_DATA_PS_6.csv', 'AGGREGATED_DATA_PS_8.csv', 'AGGREGATED_DATA_PS_10.csv']
legend_labels = ['2 zmienne decyzyjne', '4 zmienne decyzyjne', '6 zmiennych decyzyjnych', '8 zmiennych decyzyjnych', '10 zmiennych decyzyjnych']

data = [read_csv(filename, 'method', 'average_time_duration') for filename in filenames]


fig, ax = plt.subplots()


width = 0.15
all_bars = []
labels = list(data[0].keys())
for i, (filename, legend_label) in enumerate(zip(filenames, legend_labels)):
    means = [sum(data[i][label])/len(data[i][label]) for label in labels]
    bars = ax.bar([j + i*width for j in range(len(labels))], means, width=width, label=legend_label)
    all_bars.append(bars)

middle_bars = all_bars[len(filenames) // 2]
for j, bar in enumerate(middle_bars):
    ax.text(bar.get_x() + bar.get_width() / 2, 0, labels[j], 
            ha='center', va='bottom', rotation='vertical')
ax.set_xticks([j + width*(len(filenames)-1)/2 for j in range(len(labels))])
ax.set_xticklabels([''] * len(labels))


ax.legend()

plt.xlabel('Algorytmy')
plt.ylabel('Czas wykonania [s]')
plt.title('Wykres słupkowy średniego czasu wykonania dla wszystkich metod')
plt.show()

import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

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

filenames = ['AGGREGATED_DATA_10_iter.csv', 'AGGREGATED_DATA_100_iter.csv', 'AGGREGATED_DATA_1000_iter.csv']
legend_labels = ['10 iteracji', '100 iteracji', '1000 iteracji']

data = [read_csv(filename, 'method', 'average_error') for filename in filenames]

fig, ax = plt.subplots()

width = 0.3
all_bars = []
labels = list(data[0].keys())
for i, (filename, legend_label) in enumerate(zip(filenames, legend_labels)):
    print([data[i][label] for label in labels])
    means = [sum(data[i][label])/len(data[i][label]) for label in labels]
    bars = ax.bar([j + i*width for j in range(len(labels))], means, width=width, label=legend_label)
    all_bars.append(bars)
    
# Labeling at the center bar
middle_bars = all_bars[len(filenames) // 2]
for j, bar in enumerate(middle_bars):
    ax.text(bar.get_x() + bar.get_width() / 2, 0, labels[j], 
            ha='center', va='bottom', rotation='vertical')

ax.set_xticks([j + width*(len(filenames)-1)/2 for j in range(len(labels))])
ax.set_xticklabels([''] * len(labels))


ax.legend()

plt.xlabel('Algorytmy')
plt.ylabel('Wartość Błędu')
plt.title('Wykres słupkowy średniej wartości błędu  dla wszystkich metod')
plt.show()

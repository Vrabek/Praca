import matplotlib.pyplot as plt
import csv

file = 'DATA.csv'

with open(file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader) 
    group_index = header.index('method')
    value_index = header.index('error')

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

# Tworzenie wykresu pudełkowego dla wszystkich grup na jednym wykresie
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
values = list(data.values())
labels = list(data.keys())
ax.boxplot(values, labels=labels, vert=0)
plt.title('Wykresy pudełkowe dla grup')
plt.yticks(rotation=45) # obracamy etykiety o 45 stopni
plt.show()
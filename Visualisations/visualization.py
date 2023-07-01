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

    plt.title('przestrzeń wyszukiwania = [-10,+10], wymiarowość problemu = 3, limit iteracji = 100')
    
    ax.text(0.5, 1.03, 'Wykres pudełkowy wartości błędu dla wszystkich metod', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
    plt.yticks(rotation=45)
    plt.xlabel('Wartość błędu')
    plt.show()

def draw_boxplot_mid(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    values = list(data.values())
    labels = list(data.keys())
    box_plots = ax.boxplot(values, labels=labels, vert=0)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    plt.title('przestrzeń wyszukiwania = [-10,+10], wymiarowość problemu = 3, limit iteracji = 100')

    max_x_value = max([item for sublist in values for item in sublist])  # find the maximum x value in the data
    center_x_value = max_x_value / 3  # find the center of the x axis

    for i, label in enumerate(labels, 1):  # indeksy w matplotlib zaczynają się od 1, nie od 0
        ax.text(center_x_value, i, label, va='center', ha='left', color='black', fontsize=10)

    ax.text(0.5, 1.03, 'Wykres pudełkowy wartości błędu dla wszystkich metod', 
            fontsize=15, ha='center', va='bottom', transform=ax.transAxes)


    plt.yticks([])  # Usuń etykiety osi y
    plt.xlabel('Wartość błędu')
    plt.ylabel('Algorytmy')
    plt.show()


def draw_barplot(data, vert=False):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    values_prep = list(sorted_data.values())
    values = [item[0] for item in values_prep]
    labels = list(sorted_data.keys())

    if vert is False:
        bars = ax.barh(labels, values)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.title('przestrzeń wyszukiwania = [-10,+10], wymiarowość problemu = 3, limit iteracji = 100')

        for bar, label in zip(bars, labels):
            ax.text(0, bar.get_y() + bar.get_height()/2, label, 
                    va='center', ha='left', color='black', fontsize=10)

        ax.text(0.5, 1.03, 'Wykres słupkowy średniej wartości błędu dla wszystkich metod', 
                fontsize=15, ha='center', va='bottom', transform=ax.transAxes)

        plt.yticks([])  # Usuń etykiety osi y
        plt.xlabel('Średnia wartość błędu')
        plt.ylabel('Algorytmy')
        plt.show()
    else:

        bars = ax.bar(labels, values)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.title('przestrzeń wyszukiwania = [-10,+10], wymiarowość problemu = 3, limit iteracji = 100')
        for bar, category in zip(bars, labels):
            plt.text(bar.get_x() + bar.get_width() / 2, 0, category, ha='center', va='bottom', rotation=90)

        ax.text(0.5, 1.03, 'Wykres słupkowy średniego zużycia pamięci dla wszystkich metod', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
        plt.xticks([])
        plt.ylabel('Pamięć [MB]')
        plt.xlabel('Algorytmy')
        plt.show()

def draw_boxplot_hig(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    values = list(data.values())
    labels = list(data.keys())
    box_plots = ax.boxplot(values, labels=labels, vert=0, patch_artist=True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    plt.title('przestrzeń wyszukiwania = [-10,+10], wymiarowość problemu = 3, limit iteracji = 100')

    max_x_value = max([item for sublist in values for item in sublist])
    center_x_value = max_x_value / 3

    for i, label in enumerate(labels, 1):
        ax.text(center_x_value, i, label, va='center', ha='left', color='black', fontsize=10)

    ax.text(0.5, 1.03, 'Wykres pudełkowy wartości błędu dla wszystkich metod', 
            fontsize=15, ha='center', va='bottom', transform=ax.transAxes)

    colors = ['pink', 'lightblue', 'lightgreen', 'red', 'purple', 'orange', 'yellow']
    for i, box in enumerate(box_plots['boxes']):
        box.set(facecolor=colors[i % len(colors)])

    plt.yticks([])  # Usuń etykiety osi y
    plt.xlabel('Wartość błędu')
    plt.ylabel('Algorytmy')
    plt.show()


if __name__ == "__main__":
    agg_data = data_to_dict(agg_file, 'method', 'average_error')
    draw_barplot(agg_data, False)

    data = data_to_dict(file, 'method', 'error')
    draw_boxplot(data)
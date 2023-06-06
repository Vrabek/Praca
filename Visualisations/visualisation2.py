import csv
import matplotlib.pyplot as plt

def boxplot_complete():
    # Słownik do przechowywania grup danych
    data = {}
    file = 'DATA.csv'

    # Otwórz plik CSV
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Przeskocz nagłówki

        for row in reader:
            if not row:  # Pomiń puste linie
                continue
            method, value = row[0], float(row[1])  # Zakładam, że wartości są w drugiej kolumnie
            if method not in data:
                data[method] = []
            data[method].append(value)


    fig, ax = plt.subplots()

    # Tworzenie danych do wykresu
    labels = []
    box_data = []
    for method, values in data.items():
        labels.append(method)
        box_data.append(values)

    # Tworzenie wykresu pudełkowego
    ax.boxplot(box_data, vert=True, patch_artist=True, labels=labels) 

    plt.show()

if __name__ == "__main__":
    boxplot_complete()
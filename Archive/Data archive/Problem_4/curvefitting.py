import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np

def read_csv(filename, group_column, value_column):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        group_index = header.index(group_column)
        value_index = header.index(value_column)
        data = defaultdict(list)
        for row in reader:
            if row and row[group_index] == 'Algorytm Harmonijnego Przeszukiwania':
                data[row[group_index]].append(float(row[value_index]))
    return data


def func(x, a, b, c):
    #return a*x+b
    return a*x**2+ x*b + c


#filenames = ['AGGREGATED_DATA_PS_2.csv', 'AGGREGATED_DATA_PS_4.csv', 'AGGREGATED_DATA_PS_6.csv', 'AGGREGATED_DATA_PS_8.csv', 'AGGREGATED_DATA_PS_10.csv', 'AGGREGATED_DATA_PS_50.csv']
filenames = ['AGGREGATED_DATA_PS_2.csv', 'AGGREGATED_DATA_PS_4.csv', 'AGGREGATED_DATA_PS_6.csv', 'AGGREGATED_DATA_PS_8.csv', 'AGGREGATED_DATA_PS_10.csv']
legend_labels = ['2 zmienne decyzyjne', '4 zmienne decyzyjne', '6 zmiennych decyzyjnych', '8 zmiennych decyzyjnych', '10 zmiennych decyzyjnych']

data = [read_csv(filename, 'method', 'average_error') for filename in filenames]

# Przykładowe dane
xdata = np.linspace(2, 10, 5)
x = np.linspace(2, 10, 100) # użyj większej liczby punktów
#xdata = np.append(xdata, np.array([50]))
print(xdata)
ydata  = [list(d.values())[0][0] for d in data]


# Dopasowanie krzywej do danych
popt, pcov = curve_fit(func, xdata, ydata)

# Wyświetlanie wyników
plt.figure()
plt.scatter(xdata, ydata, color='r', label='Dane z Obserwacji') # Dodane zaznaczanie punktów
plt.plot(x, func(x, *popt), label = 'Regresja Wielomianowa')
plt.xlabel('liczba zmiennych decyzyjnych')
plt.ylabel('średnia wartość błędu')
plt.title('Regresja wielomianowa średniej wartości błędu \ndla Algorytmu Harmonijnego Przeszukiwania')
plt.legend()
plt.show()
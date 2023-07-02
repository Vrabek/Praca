import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
import numpy as np

# Załaduj dane
df = pd.read_csv('DATA.csv')

# Wykonaj test Kruskala-Wallisa
groups = df.groupby('method')['error'].apply(list).values.tolist()
h_val, p_val = kruskal(*groups)

print('Wartość H:', h_val)
print('P-Wartość:', p_val)

# Jeśli wynik Kruskala-Wallisa jest istotny, wykonaj test Dunn'a
if p_val < 0.05:
    dunn_results = sp.posthoc_dunn(df, val_col='error', group_col='method')
    print(dunn_results)

    # Zastąp wartości p-wartość mniejsze od 0.05 wartością NaN
    dunn_results[dunn_results < 0.05] = np.nan

    # Stwórz mapę ciepła z większym rozmiarem i mniejszą czcionką dla anotacji
    plt.figure(figsize=(15, 10))  # Zwiększ rozmiar wykresu
    sns.heatmap(dunn_results, annot=True, cmap='coolwarm', annot_kws={"size": 7})  # Zmniejsz rozmiar czcionki do 8
    plt.title('Macierz p-wartości > 0.05 dla testu Dunna')
    plt.tight_layout()
    plt.show()

    # Zapisz wyniki do pliku CSV
    dunn_results.to_csv('dunn_results.csv')
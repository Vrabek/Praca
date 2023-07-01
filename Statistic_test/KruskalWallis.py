import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp

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

    # Stwórz mapę ciepła z mniejszą czcionką dla anotacji
    plt.figure(figsize=(10, 8))
    sns.heatmap(dunn_results, annot=True, cmap='coolwarm', annot_kws={"size": 10})  # zmniejsz rozmiar czcionki do 10
    plt.title('Heatmap p-values from Dunn\'s Test')
    plt.show()

    # Zapisz wyniki do pliku CSV
    dunn_results.to_csv('dunn_results.csv')

# Pandas is a python library for data analysis
import pandas as pd
import os.path  # questo è solo per fare una cosa aggiuntiva


# IMPORT DATA
data = pd.read_csv("pokemon_data.csv")  # carica il csv file
# Per dati da excel possiamo usare: pd.read_excel("pokemon_data.xlsx")
# Per dati divisi da tab usiamo: pd.read_csv("pokemon_data.txt", delimiter="\t") e similmente per altri delimiter

# ACCESS DATA IN SPECIFIED LOCATION
primi_3 = data.head(3)   # primi 3 dati
ultimi_3 = data.tail(3)  # ultimi 3 dati
print(data.columns)   # header delle colonne (attributi)
print(data[["Name", "Type 1"]][0:5])   # dei primi 5 oggetti prende "Name" e "Type 1" (sono due degli attributi); [[]] è necessario solo nel caso di una lista di colonne, altrimenti si usa []
print(data.iloc[0:3, 1])   # iloc accede alle posizioni specificate, con la stessa struttura degli array di NumPy
grass_pokemons = data.loc[data["Type 1"] == "Grass"]  # loc accede alle posizioni che rispettano la condizione data
data.loc[data["Type 1"] == "Fire", ["Legendary", "Generation"]] = [True, 5]  # posso modificare i dati a cui accedo
# Possiamo usare gli operatori logici, ma scritti come: &, |, ~ (not); in generale bisogna farlo se si lavora con liste
grass_big_pokemons = data.loc[(data["Type 1"] == "Grass") & (data["HP"] > 70)]
for index, row in data.iterrows():  # modo rapido per accedere a tutte le righe di un attributo
    if index in range(0, 6):
        print(index, row["Name"])


# REORDER AND MODIFY DATA
information = data.describe()  # riporta informazioni utili per ogni attributo: conteggi, media, dev std, min, max, ...
sorted_data = data.sort_values(["Type 1", "HP"], ascending=[False, True])  # riordina i dati in base agli attributi dati
# sort.values non modifica l'array su cui è applicato
# Possiamo aggiungere nuovi attributi o rimuoverne
data["Total"] = data["HP"] + data["Attack"] + data["Defense"] + data["Sp. Atk"] + data["Sp. Def"] + data["Speed"]
# Equivalentemente si poteva fare: data["Total"] = data.iloc[:, 4:10].sum(axis=1) dove axis=1 significa sommare colonne
dropped_data = data.drop(columns=["Total"])  # non modifica l'array (se non specifichiamo inplace=True)
# Possiamo anche riordinare le colonne; un modo è il seguente: usiamo la funzione list per creare una lista degli attributi; riordiniamo la lista
# Attenzione: singoli elementi vengono trasformati in stringhe, dunque bisogna ritrasformarle in liste chiudendole in []
lista_attributi = list(data.columns.values)
reordered_data = data[lista_attributi[0:2] + [lista_attributi[-1]] + lista_attributi[2:12]]  # non modifica l'array
print(reordered_data.head(3))
grass_big_pokemons_non_mega = grass_big_pokemons.loc[~ grass_big_pokemons["Name"].str.contains("Mega")]  # ~ = not
grass_big_pokemons_non_mega.reset_index(drop=False, inplace=True)  # per resettare l'indice; altrimenti quando scegliamo di tenere solo alcuni elementi, vengono mantenuti gli indici dell'array iniziale
# drop=False crea un nuovo attributo (in posizione 1) in cui tiene i vecchi indici; inplace=True modifica l'array
# Un altro importante metodo è groupby che raggruppa gli elementi che rispettano una certa condizione o che condividono lo stesso valore di un attributo
# Può essere utile combinarlo con i metodi count, mean, sum, std (deviazione standard)
data.groupby(["Type 1"]).mean().sort_values("Attack", ascending=False)
# Nel caso di dataset estramemente grandi si può voler caricare solamente un pezzo di dati (o un pezzo alla volta)
# Poi magari possiamo combinare i dati analizzati in un unico array finale
results = pd.DataFrame(columns=data.columns) #crea un nuovo dataframe
for chunk_of_data in pd.read_csv("pokemon_data.csv", chunksize=100):  # chunksize è il numero di righe
    chunk_of_results = chunk_of_data.groupby(["Type 1"]).sum()
    results = pd.concat([results, chunk_of_results])


# EXPORT DATA
if not os.path.isfile("reordered_pokemon_data.csv"):  # controlla che il file non esista già
    reordered_data.to_csv("reordered_pokemon_data.csv", index=False)  # esporta in formato csv (senza colonna d'indici)
if not os.path.isfile("reordered_pokemon_data.txt"):
    reordered_data.to_csv("reordered_pokemon_data.txt", index=False, sep="\t")  # esporta dati separati da tab
# if not os.path.isfile("reordered_pokemon_data.xlsx"):
#  reordered_data.to_excel("reordered_pokemon_data.xlsx", index=True)  # esporta in formato excel
# NumPy is a library for fixed type multidimensional rectangular arrays: being fixed type, operations are much faster
import numpy as np

lista2 = [
    [1.2, 2.3, 3.4],
    [11, 22, 33],
    [100, 200, 300],
    [12.3, 23.4, 34.5]
]
array2 = np.array(lista2)
print(array2)
print(array2.ndim)  # dimensione dell'array
print(array2.size)  # numero di elementi
print(array2.shape)  # forma dell'array, ovvero lunghezza di ogni dimensione (ricordare che sono rettangolari)
print(array2.dtype)
# datatypes in numpy are: (always call them with np.)
# bool_, byte (char), ubyte (unsigned char), short, ushort, intc, uintc, int_ (long), uint, longlong, ulonglong
# half = float16, single (float), double, longdouble, csingle (float complex), cdouble, clongdouble
# o anche tramite il numero di byte, come int8, int16, ..., float32, float64, ...

# Possiamo accedere a ogni elemento dell'array con dei for usando dummy variables in range dati dalle lunghezze
for i in range(array2.shape[0]):
    for j in range(array2.shape[1]):
        array2[i, j] += (i+1)+(j+1)  # notazione alternativa: array2[i][j]
# Attenzione, si può usare la notazione delle liste [i][j] fintanto che stiamo facendo operazioni lecite anche su liste
# Per operazioni specifiche degli array bisogna usare [i, j]
print(array2)
# Più in generale possiamo usare tutti i metodi introdotti per le liste:
# index, extend, append, insert, count, remove, pop, clear, sort, reverse, copy
# Ci sono anche molte operazioni in più e molte generalizzazioni
# Infatti la restrizione di essere rettangolari rende ben definite alcune operazioni che sulle liste non lo erano
print(array2[:, 0])  # posso anche accedere a una data colonna
# In generale la notazione è: startindex:endindex:stepsize
print(array2[1:4:2, 0])

array3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(array3)

zero_array = np.zeros((2, 3, 4))  # array con soli 0, delle dimensioni specificate
one_array = np.ones((2, 3, 4))  # array con soli 1, delle dimensioni specificate
n_array = np.full((3, 5), 3.14)   # array con soli 3.14, delle dimensioni specificate
random_array = np.random.uniform(0, 1, (2, 4))  # array con numeri casuali uniformi tra 0 e 1
random_int_array = np.random.randint(10, 101, (4, 2))  # array con numeri interi casuali tra 10 e 100
print(np.random.normal(50, 20, (4, 2)))  # array con numeri casuali gaussiani di media 50 e deviazione standard 20
# E in generale in np.random ci sono tutte le distribuzioni di probabilità
identity_matrix = np.identity(3)  # identity matrix

# Esistono molte operazioni che si possono fare element by element
a = np.array([1, 2, 3, 4, 5, 6])
print(np.cos(a)+2)  # similmente con tutte le altre operazioni (e tutte le funzioni matematiche)
b = np.array([1, 0, 1, 0, 1, 0])
print(a*b)  # similmente con tutte le altre operazioni

# Si possono poi fare tutte le operazioni tra matrici
random_square = np.matmul(random_array, random_int_array)
print(random_square)  # matrix multiplication
np.linalg.det(random_square)  # determinante
# Similmente si possono fare traccia, inversa, autovalori, norma, SVD, ...
# Per una lista completa visitare https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

np.min(random_int_array)  # valore minimo
np.max(random_int_array)  # valore massimo
np.sum(random_int_array)  # somma tutti i valori
np.sum(random_int_array, axis=0)  # somma tutti i valori su ogni colonna (passando da matrice a vettore riga)
np.sum(random_int_array, axis=1)  # somma tutti i valori su ogni riga (passando da matrice a vettore colonna)

a_reshaped = a.reshape(3, 2)  # cambia la forma di una matrice (riempiendola in ordine)
ab_stack = np.vstack([a, b, a, b])  # sovrappone verticalmente due o più array
random_stack = np.hstack([random_square, random_array])  # sovrappone orizzontalmente due o più array

# Possiamo caricare dati da file e storarli in un array
# Ci sono varie opzioni, sono riportate solo le più importanti
# Delimiter è il modo in cui sono divisi i dati; di default è un whitespace
data_array1 = np.genfromtxt('Input_file.txt', dtype=np.int16, delimiter='', skip_header=1)
print(data_array1)
# Esiste anche loadtxt, che è più veloce, ma ha meno opzioni
data_array2 = np.loadtxt('Input_file.txt', skiprows=1)
print(data_array2)

# Si possono mettere delle condizioni sugli elementi degli array
print(random_square > 50)  # restituisce un array di booleani
print(np.logical_and(random_square > 50, random_square < 100))  # si possono usare gli operatori logici di np
print(random_square[random_square > 50])  # restituisce solo i valori che rispettano la condizione
# Possiamo anche fornire una lista degli elementi che vogliamo considerare
# Per farlo negli array D-dimensionali bisogna fornire D liste con lo stesso numero n di elementi
# Questo identifica n D-plette che a loro volta identificano n posizioni nell'array
print(array2[[0, 1, 3], [0, 1, 2]])  # restituisce gli elementi in posizione (0,0), (1,1) e (3,2)
# Alternativamente si può fornire una lista per una delle dimensioni e magari un insieme per un'altra
print(array2[[0, 1, 3], 1:])  # restituisce gli elementi nelle righe 0, 1 e 3 delle colonne dall'1 in avanti

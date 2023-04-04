# I commenti sono fatti così
# Le spaziature, le indentazioni, etc... sono molto importanti in Python ed è ovviamente anche case sensitive
# Convenzionalmente le righe hanno una lunghezza massima (ad esempio 120 caratteri); all'interno di una parentesi si può andare a capo senza
# problemi, tramite indentazione, altrimenti si usa il \ a fine riga per segnalare che si continua sulla riga sotto
# In generale è anche ritenuto importante usare nomi descrittivi (tanto che molti IDE hanno il controllo ortografico)
# Per runnare alcune parti di codice (al fine di testarli) si può usare la console
# Se una variabile v2 dipende da un'altra variabile v1, la definizione di v1 deve venire prima di quella di v2
# Tuttavia se una funzione dipende da v1, la definizione di v1 deve venire prima della call della funzione, ma non necessariamente prima della sua definizione

import welcome
from math import *  #NOQA
# usando * ho importato tutti i moduli di math (come se fossero direttamente scritti su questo file)
# usando #NOQA sopprimo il warning che verrebbe displayato (dato che usare import * viene considerato pericoloso)
# alternativamente potrei usare:
    # import math
    # in questo modo math viene importata come libreria e tutte le sue funzioni sono accessibili chiamandole come math.sqrt() eccetera
    # questo secondo metodo è preferito

print("Hello World")    # si possono fare anche in line, lasciando almeno due spazi
character_name = "John"
character_age = 35.2
booleano1 = True


# STRINGS
print("Hello\nWorld")   # newline; \ è l'escape character, quindi si può usare per avere, ad esempio," come carattere
surname = "Smith"
# Ci sono varie funzioni che si possono applicare alle stringhe, accessibili, in genere, tramite stringa. (con il punto)
surname.upper()     # mette tutto in upper case
# Tali funzioni si possono anche concatenare
surname.upper().isupper()   # True
len(surname)  # lunghezza
print(surname[0])  # primo carattere
print(surname[len(surname)-1])  # ultimo carattere
print(surname[-1])  # ultimo carattere, più semplicemente; notare che contando dalla fine, il primo carattere è -1
surname.index("m")  # 1: index fornisce la posizione del carattere specificato
surname.replace("m", "c")  # sostituisce m con c
print(r"$\omega$")  # anteporre r fa diventare la stringa raw: in questo modo viene interpretata letteralmente
# Questo è utile per scrivere usando MathText, invocato tramite $$ e simile al LaTeX
# String concatenation
nome1 = "un nome"
print("dico " + nome1)
print("dico {}".format(nome1))
print(f"dico {nome1}")  #f-string: f"stringa {variabile}"
print("dico", nome1) # notare che in questo modo viene aggiunto automaticamente uno spazio



# NUMBERS
print(10.23)  # non possiamo concatenare stringhe e numeri, ma possiamo convertire numeri in stringhe o viceversa
print("In questo modo è possibile mostrare " + str(10.23) + " in mezzo a un print.")  # similmente con int() o float()
# Operazioni: +, -, *, /, % (mod); ci sono le shortcuts += n, -= n, *= n, /=n
abs(-4)  # absolute value
pow(2, 10)  # potenza, 2^10; esiste anche la shortcut **: 2**10
sqrt(36)
max(5/7, 3/5)  # similmente con min
round(4.6)  # arrotonda (4.5 -> 4)
floor(4.7)  # per difetto
ceil(4.3)  # per eccesso


# INPUT FROM USER
# user_name = input("Enter your name: ")
# user_age = input("Enter your age: ")
# user_age_number = float(user_age)
# Vediamo un esempio con vari cambi di data type e concatenazione di funzioni:
# print(user_name + " is " + user_age + ", almost " + str(floor(user_age_number+1)))


# LISTS
# Le liste sono un elemento fondamentale in Python, similmente alle liste in Mathematica
# Diversamente dagli array di C++, le liste in Python possono contenere elementi di diverso tipo e hanno molte funzioni
lista1 = ["el0", "el1", 2, True, "el4"]
lista2 = ["1el0", 11, "1el2", 11, False]
print(lista1[3])  # anche con numeri negativi, per iniziare dal fondo
print(len(lista1))
print(lista1.index(True))
print(lista1[1:3])  # prende gli elementi da 1 a 3, 1 incluso e 3 escluso (prende 3-1 elementi)
lista1.extend(lista2)  # aggiunge gli elementi di lista2 a lista1, dal fondo; modifica lista1
lista1.append(33)  # aggiunge l'elemento specificato al fondo della lista
lista1.insert(2, "nuovo elemento al posto 2")  # aggiunge nell'indice specificato (si parte da 0) l'elemento passato
print(lista1.count(11))  # conta il numero di volte che compare l'elemento specificato
lista1.remove(11)  # rimuove la prima istanza dell'elemento specificato
print(lista1.pop())    # stora l'ultimo elemento della lista e lo rimuove dalla lista
print(lista1)
lista1.clear()  # svuota la lista
# Vediamo in particolare la funzione sort, che serve per ordinare gli elementi di una lista
# Funziona solo per liste omogenee, ovvero con un solo tipo di data type
# Le stringhe vengono ordinate in ordine alfabetico, con i caratteri numerici davanti e in modo case sensitive
# I numeri vengono ordinati in modo crescente
lista3 = ["casa", "formaggio", "cane", "alfabeto", "Python", "s5", "11", "Alfabeto"]
lista3.sort()
print(lista3)
lista3.reverse()  # inverte l'ordine degli elementi di una lista
copia_lista3 = lista3.copy()  # crea una copia (indipendente) di una lista
# Attenzione: se usassimo copia_lista3 = lista3, le due copie sarebbero dipendenti (sarebbero sempre modificate insieme)


# TUPLES
# Le tuples sono insiemi di elementi, ma, diversamente dalle liste, non possono essere modificate in nessun modo
coordinates = ("tel 0", 3, False)


# SETS
# I sets sono insiemi di elementi non ordinati, modificabili e senza duplicati
set1 = {"set_el2", 24, False, "set_el0", False}
print(set1)
# Possiamo trasformare un diverso oggetto in un set attraverso il comando set
set2 = set("Hello")
print(set2)
set1.add(True)  # aggiunge un elemento
set1.remove(False)  # rimuove un elemento
# Anche se False sembra comparire due volte in set1, in realtà non ci possono essere duplicati
# Dunque rimuovere un elemento una volta è sufficiente
set1.discard(22)   # rimuove un elemento, se c'è; altrimenti non fa nulla (non dà errore)
set2.clear()   # svuota il set
set1.pop()  # stora e rimuove un elemento casuale del set (è l'ultimo elemento, ma l'ordine è casuale)
print(set1)
set3 = {1, 2, 3, 4, 24}
unione = set1.union(set3)  # unione d'insiemi (senza duplicazione)(simmetrica)
intersezione = set1.intersection(set3)  # intersezione d'insiemi (simmetrica)
differenza = set1.difference(set3)  # differenza d'insiemi (asimmetrica)
differenza_simmetrica = set1.symmetric_difference(set3)  # differenza d'insiemi (simmetrica): unione - intersezione
set1.update(set3)  # unione d'insiemi con modifica dell'insieme di partenza
set1.intersection_update(set3)  # intersezione d'insiemi con modifica dell'insieme di partenza
set1.difference_update(set3)  # differenza d'insiemi (asimmetrica) con modifica dell'insieme di partenza
set1.symmetric_difference_update(set3)  # differenza d'insiemi (simmetrica) con modifica dell'insieme di partenza
set3.issubset(set1)  # verifica se si ha una relazione di contenuto
set3.issuperset(set1)  # verifica se si ha una relazione di contenimento
set3.isdisjoint(set1)  # verifica se i due set sono disgiunti
set4 = set3.copy()  # crea una copia (indipendente)
frozen_set = frozenset(["el2", 34, True])  # crea un set non modificabile
# Esistono anche altri tipi di contenitori, definiti nella libreria collections


# FUNCTIONS
# Le funzioni sono definite tramite la keyword def; il body della funzione inizia dopo i:
# Il body della funzione è segnalato da un'indentazione; per chiudere il body basta smettere d'indentare
# Le funzioni devono essere spaziate da codice (e commenti) sovrastanti e sottostanti tramite 2 righe vuote
# Infine notiamo che non bisogna definire le tipologie degli argomenti passati
# Di conseguenza si possono definire operazioni illecite e questo non dà un errore subito, ma solo durante l'esecuzione
def funzione_prova(argument1, argument2):   # definizione della funzione
    print(argument1)
    print(argument2 + 23)
    return argument1 + str(argument2)   # se usiamo un return statement, questo chiude la funzione


stringa_unita = funzione_prova("stringa_passata", 77)     # call della funzione


# IF STATEMENTS
# Struttura:
# if condition:
#   body
# Similmente alle funzioni e a tutti gli oggetti con un body, il body dell'if inizia dopo i: e va indentato
# Logical operators
# and, or, not, <, <=, >, >=, ==, !=, in
booleano2 = False
if not (booleano1 or booleano2):
    print("I booleani sono falsi")
elif booleano1:
    print("booleano1 è vero")
else:
    print("Il booleano2 è vero")
# Possiamo usare if comparando dei numeri o dei booleani
# Possiamo anche comparare delle stringhe, ma in quel caso l'unico operatore logico sensato è == (e l'opposto !=)
small_integer = 4
if small_integer in [0, 1, 2, 3, 4, 5]:
    print(factorial(small_integer))
else:
    print("Not allowed")


def max_3numbers(n1, n2, n3):
    if n1 >= n2 and n1 >= n3:
        return n1
    elif n2 >= n1 and n2 >= n3:
        return n2
    else:
        return n3


print(max_3numbers(4, 6, 7))


# DICTIONARIES
# I dizionari sono degli insiemi di sostituzioni, ovvero un insieme di coppie chiave-valore (key-value pairs)
# Le coppie sono inserite tra graffe con indentazione e divise da virgole; si usano i: tra elementi delle coppie
day_conversion = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}
print(day_conversion[2])
day_conversion.get(3, "Default")  # come la riga precedente, ma si può anche settare un default per chiavi indefinite


# WHILE LOOP
# Struttura:
# while condition:
#   body
# Si usa per loopare fintanto che una condizione è verificata
i = 1
base = 2
constant = 10
while i < 6:
    base *= constant
    i += 1
print(base)


# FOR LOOP
# Struttura:
# for variable in collection:
#   body
# Si usa per loopare tramite una dummy variable all'interno di un set di valori possibili (collection)
# Sottolineiamo che la dummy variable può essere di qualsiasi data type
# O meglio, il data type è in realtà specificato dal tipo di collection che si utilizza
# Dunque più correttamente diciamo che la collection può essere di qualsiasi data type
# collection: stringa -> dummy: carattere
# collection: lista/tupla/set -> dummy: elemento
# collection: intervallo -> dummy: numero
# Gli intervalli si scrivono range(xmin, xmax), xmin è incluso, xmax escluso; se xmin = 0 si può omettere
for lettera in "Una frase":
    print(lettera)
for elemento in ["el1", 2, False]:
    print(elemento)
for indice in range(10, 20):
    print(indice)


# 2D LISTS AND NESTED LOOPS
# Si possono creare liste di liste; non c'è nessuna restrizione sulle dimensioni
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0]
]
grid_element_12 = grid[0][1]  # per accedere agli elementi della lista (si ricorda che la numerazione parte da 0)
grid_row1 = grid[0][:]  # per accedere a un'intera riga (non c'è un modo diretto per le colonne)
# Dato che possiamo loopare sulle liste, possiamo facilmente accedere ai vari elementi di una lista 2D
# Infatti non c'è bisogno di una dummy variable che funga da indice; la dummy variable può essere l'elemento della lista
for riga in grid:
    for colonna in riga:
        print(colonna)
# La costruzione si può fare anche in più dimensioni, tuttavia bisogna stare attenti alle dimensioni
# Le liste sono fatte per funzionare fino in D = 2; in D > 2 è possibile farle funzionare finchè sono rettangolari
# Non significa nulla loopare su un singolo elemento, dunque se la lista è "mista 2D-3D" non posso fare un loop 3D
# Non è nemmeno possibile usare una dummy variable, perchè anche la lunghezza di un numero non ha nessun significato
grid3 = [  # esempio di lista "mista 2D-3D"
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [["a", "b"], "c"]
]


# TRY EXCEPT
# Struttura:
# try:
#   body
# except error:
# alternative body
# try:
#    input_number1 = int(input("Enter a number: "))
#    input_number2 = int(input("Enter a number: "))
#    print(input_number1/input_number2)
# except ZeroDivisionError:
#    print("Division by zero")
# except ValueError as errore:
#    print("Invalid input")  # possiamo decidere di printare una frase personalizzata
#    print(errore)  # oppure di printare l'errore ricevuto


# READING FILES
# Argomento: relative path, absolute path o file name se nella stessa directory (è possibile crearlo da PyCharm)
# Modi: "r" (read), "w" (write), "a" (append), "r+" (read and write)
opened_file = open(r"C:\Users\Emanuele\Desktop\GitHub repositories\Python guides and examples\Input_file.txt", "r")
if opened_file.readable():
    # print(opened_file.read())  # read legge tutto il file
    print(opened_file.readline())  # readline legge una riga e si muove a quella successiva
    print(opened_file.readline())  # in modo che due readline successivi diano due righe successive
    # readlines invece legge il file a partire dalla riga corrente e crea una lista in cui ogni elemento è una riga
    # su tale lista si possono fare tutte le operazioni possibili su una qualsiasi lista
    # attenzione: se ho ad esempio usato qualche readline prima di readlines, allora readline non vede tutto il file
    read_list = opened_file.readlines()
    print(read_list[2])  # questo è in realtà l'elemento d'indice 4 del file, dato che i primi 2 non sono letti
else:
    print("Unreadable file")
opened_file.close()


# WRITING TO FILES
# Bisogna fare attenzione con l'append: eseguire lo script più volte continua ad appendere più volte la stessa scritta
# Un'idea per evitarlo è ad esempio la seguente:
# controllare se il testo che si vuole aggiungere è già presente nel txt, prima di appendere nulla
# se non c'è, usare un if e appendere solo se tale elemento continua a non essere presente
# se c'è, fare un procedimento simile, ma usando un if condizionato al numero di volte che tale elemento è presente
# quest'ultimo si può fare usando il comando count e verificando il count sia uguale a quello iniziale del file
opened_output_file = open("Output_file.txt", "a")
opened_output_file.write("\nAdditional text")
opened_output_file.close()
# Altrimenti possiamo usare "w" in modo da sovrascrivere il testo presente
# Se il file specificato non esiste, viene creato
new_output_file = open("New_output_file.txt", "w")
new_output_file.write("Some new text")
new_output_file.close()


# MODULES AND PIP
# Possiamo importare un file Python esterno per avere accesso a tutte le funzioni e variabili definite in esso
# Per importare si usa il comando import all'inizio del Python file
# Si possono anche importare solamente alcune parti di file usando:
#   from filename import object
# Per chiamare funzioni e variabili di un certo file, bisogna usare:
#   nomefile.nomevariabile o nomefile.nomefunzione()
# Le librerie sono già presenti nel progetto (sotto external libraries), dunque vale la stessa procedura
# Mentre si importa un file è anche possibile dargli un nome alternativo da usare dentro il file di lavoro, per comodità
# Un esempio classico è quello di usare:
#   import numpy as np oppure import matplotlib.pyplot as plt
# Possiamo anche importare moduli esterni (non built-in), previa installazione con pip (finiscono in /Lib/site-packages)
# di fatto pip è un package manager (preinstallato in python3) e si può usare semplicemente da terminale tramite:
#   pip install packagename
# L'unico caso in cui questo non funziona è quello in cui si vogliono importare pacchetti non nativi per Python
# pip funziona solo con pacchetti Python, per altri tipi di pacchetti bisogna usare conda (ma è un caso molto raro)
print(welcome.dice_roll(20))


# CLASSES, OBJECTS AND CLASS FUNCTIONS
# Struttura:
# class ClassName:
#   def __init__(self, arguments):
#       self.argument = argument
#   def class_function(self, argomenti):
#       body
# Similmente alle funzioni bisogna lasciare due righe vuote attorno a una classe
# Il nome della classe deve seguire la CamelCase convention
# In genere le funzioni definite all'interno di una classe lavorano anche con gli attributi dell'oggetto (self.variable)
# Si possono anche usare le funzioni per modificare gli attributi


class Student:

    def __init__(self, nome, major, media, is_male):    # è l'analogo della constructor function in C++
        self.nome = nome  # self.nome è il nome dell'oggetto, e lo si inizializza al valore passato in fase di creazione
        self.major = major
        self.media = media
        self.is_male = is_male

    def lode(self):
        if self.media >= 28:
            return True
        else:
            return False


student1 = Student("Mike", "Informatics", 27, True)
student2 = Student("Jennifer", "Mathematics", 28, False)
print(student1.media)
print(student2.lode())


# INHERITANCE
# Posso ereditare gli attributi e i metodi da un'altra classe, tramite ()
# Viene ereditata anche l'initialising function
# Possiamo anche sovrascrivere dei metodi


class PhysicsStudent(Student):

    def lab_report(self):
        print("I am " + self.nome + " and I am doing a lab report")

    def lode(self):
        if self.media >= 28.5:
            return True
        else:
            return False


physics_student1 = PhysicsStudent("Tom", "Physics", 28.2, True)
print(physics_student1.lode())

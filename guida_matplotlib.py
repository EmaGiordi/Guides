import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import os.path

# Vedremo le funzioni più interessanti, ma la lista completa si può trovare su:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
# Una guida con applicazioni (e github) si trova su: https://www.youtube.com/watch?v=cTJBJH8hacc&ab_channel=Mr.PSolver


x = np.array([0.5, 1, 2, 3, 4, 5])
y = np.array([0.3, 1, 4, 8, 18, 24])

plt.figure(figsize=(10, 6), dpi=100)  # comando da usare all'inizio che può specificare tantissimi attributi generali
# Iniziamo con i grafici
# Innanzitutto creiamo l'oggetto plot, che poi possiamo modificare e infine mostrare
plt.plot(x, y, color="green", linewidth=0.5, linestyle="",
         marker=".", markersize=6, markeredgecolor="red", label="data points")
# Esiste la shorthand notation: "color marker line", ad esempio "g.--"
# Possiamo creare altre curve e verranno aggiunte allo stesso grafico
plt.plot(x, pow(x, 2), color="blue", label="theoretical values")
plt.title("Titolo del grafico", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel("Asse x", fontdict={"fontsize": 14})
plt.ylabel("Asse y")
plt.xticks(range(0, 6, 1))
plt.yticks(range(0, 30, 5))
plt.xscale("log")  # tipo di scala
plt.xscale("linear")
plt.ylim(top=26)  # limiti del grafico
plt.legend(loc="upper left", fontsize=10, ncol=2)  # la legenda mostra il label specificato in plot
if not os.path.isfile("grafico.png"):
    plt.savefig("grafico.png", dpi=200)  # per salvare la figura
# Alternativamente si può salvare la figura dalla GUI che mostra il grafico
# plt.show()  # mostra il plot; se creo altri plot verranno ora messi in un nuovo grafico
# Alternativamente possiamo clearare la figura anche senza plottarla
# plt.clf()  # clear current figure
# plt.close()  # chiude la finestra della figura
# Ad ogni modo se si vuole fare un altro grafico la cosa migliore è semplicemente creare una nuova figure


# Vediamo adesso una bar chart
# Potremmo modificare il grafico similmente a quanto fatto prima
# Vediamo invece solo alcuni degli attributi specifici degli istogrammi

labels = ["A", "B", "C"]
values = [1, 4, 2]
plt.figure(figsize=(10, 6), dpi=100)
bars_plot = plt.bar(labels, values)
bars_plot[0].set_hatch("/")  # stile della barra
bars_plot[1].set_hatch("o")
bars_plot[2].set_hatch("*")


# Vediamo gli istogrammi
ist_data = np.random.normal(50, 20, 1000)
binning = range(-10, 110, 10)
plt.figure(figsize=(10, 6), dpi=100)
plt.hist(ist_data, bins=binning, color="#6d1fcc", density=True)   # density è per normalizzare
# possiamo prendere il colore tramite un color picker (ad es Chrome)


# Possiamo anche creare pie charts
plt.figure(figsize=(10, 6), dpi=100)
plt.pie([24, 36, 7, 14], labels=["A", "B", "C", "D"], explode=[0.1, 0.1, 0.3, 0.2], autopct="%.1f %%")
# autopct calcola la percentuale di ogni caso e la mette nel grafico come label
# .1f significa che si usa 1 cifra dopo la virgola; %% è il segno percentuale


# Possiamo creare molteplici grafici in un figura
fig, axes = plt.subplots(3, 2, figsize=(10, 5))

ax00 = axes[0][0]
ax00.plot(x, y, 'o--', color='r', lw=0.4, ms=3)
ax00.text(0.1, 0.1, 'text here', transform=ax00.transAxes)  # per aggiungere testo nel subplot
# transform=ax00.transAxes è per usare posizioni come intervalli relativi a (0,1)
ax00.set_title("Titolo 00")  # notiamo la notazione un po' diversa quando usiamo subplots
ax11 = axes[1][1]
ist_data2 = np.random.randn(1000)*0.2 + 0.4
ax11.hist(ist_data2, bins=30, density=True, histtype='step', label='ist 2')
ax11.set_xlabel("$\hat{E}-E$")
ax11.legend()
fig.tight_layout()  # aggiusta le distanze tra i subplots
fig.suptitle('Title of All Plots', y=0, fontsize=25)  # bisognerebbe un po' sistemare i parametri
ax11.tick_params(axis="both", labelsize=10)  # dimensioni dei numeri sugli assi


# Possiamo anche fare plot 2d
plt.figure(figsize=(10, 6), dpi=100)
dominio2 = np.linspace(-1, 1, 100)
x2, y2 = np.meshgrid(dominio2, dominio2)
# meshgrid serve per creare una matrice di coordinate combinando due vettori di coordinate
z2 = x2**2 + x2 * y2
contour_plot = plt.contourf(x2, y2, z2, levels=30, vmax=1.68, cmap="plasma")  # contour invece non riempie di colore
# vmax e vmin possono essere utili nel caso di funzioni divergenti; settano il max e il min della scala di colori
plt.colorbar(label="legenda colori")
# plt.clabel(contour_plot, fontsize=8)  # mette i valori all'interno del grafico, sulle linee
# in genere clabel si usa in contour, mentre colorbar si usa in contourf

# Alternativamente possiamo plottare la superficie in 3d
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
ax2.plot_surface(x2, y2, z2, cmap='coolwarm', linewidth=0, antialiased=False)
ax2.view_init(elev=10, azim=50)


# Possiamo plottare campi vettoriali (streamplot)
w = 3
dominio_stream = np.linspace(-3, 3, 100)
x_stream, y_stream = np.meshgrid(dominio_stream, dominio_stream)
U = -1 - x_stream**2 + y_stream
V = 1 + x_stream - y_stream**2
speed = np.sqrt(U**2 + V**2)

fig_stream, axes_stream = plt.subplots(2, 2, figsize=(5, 5))

ax_stream00 = axes_stream[0][0]  # classico, senza opzioni
ax_stream00.streamplot(x_stream, y_stream, U, V)

ax_stream01 = axes_stream[0][1]  # vettori colorati diversamente per velocità diverse
ax_stream01.streamplot(x_stream, y_stream, U, V, color=speed)

ax_stream10 = axes_stream[1][0]
lw = 5*speed / speed.max()  # vettore più spesso per velocità più elevate
ax_stream10.streamplot(x_stream, y_stream, U, V, linewidth=lw)

ax_stream11 = axes_stream[1][1]
seed_points = np.array([[0, 1], [1, 0]])   # considero solo il flow di due punti iniziali
ax_stream11.streamplot(x_stream, y_stream, U, V, color=U, linewidth=2, cmap='autumn', start_points=seed_points)
ax_stream11.grid()


# Possiamo creare delle animazioni
def function_an(x_an, t_an):  # creiamo una funzione dipendente dal tempo
    return np.sin(x_an-3*t_an)

dominio_x_an = np.linspace(0, 10*np.pi, 1000)
dominio_t_an = np.arange(0, 24, 1/60)
valori_x_an, valori_t_an = np.meshgrid(dominio_x_an, dominio_t_an)
function_an_values = function_an(valori_x_an, valori_t_an)  # valutiamo la funzione a diversi istanti (vicini)

fig_an, ax_an = plt.subplots(1, 1, figsize=(8, 4))  # creiamo un plot per l'animazione
ln1, = plt.plot([], [])  # creiamo un plot (ora vuoto)
time_text = ax_an.text(0.65, 0.95, '', fontsize=15, bbox=dict(facecolor='white', edgecolor='black'),
                       transform=ax_an.transAxes)  # testo che mostrerà lo scorrere del tempo
ax_an.set_xlim(0, 10 * np.pi)
ax_an.set_ylim(-1.5, 1.5)

def animate(i):  # definiamo una funzione che valuta funzione e testo a un dato istante
    ln1.set_data(dominio_x_an, function_an_values[i])
    time_text.set_text('t={:.2f}'.format(i / 60))

ani = animation.FuncAnimation(fig_an, animate, frames=240, interval=50)
if not os.path.isfile("animation.gif"):
    ani.save('animation.gif', writer='pillow', fps=50, dpi=100)  # bisogna usare questo metodo speciale per salvare gif

# L'animazione può essere utile per mostrare plot di superfici
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
ax2.plot_surface(x2, y2, z2, cmap='coolwarm', linewidth=0, antialiased=False)  # è la stessa superficie di prima
ax2.view_init(elev=10, azim=0)  # initial viewing angle

def animate_surface(i):  # questa volta la funziona cambia il viewing angle
    ax2.view_init(elev=10, azim=3 * i)

ani_surface = animation.FuncAnimation(fig2, animate_surface, frames=120, interval=50)
if not os.path.isfile("animation_surface.gif"):
    ani_surface.save('animation_surface.gif', writer='pillow', fps=30, dpi=100)

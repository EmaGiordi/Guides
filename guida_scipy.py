# SciPy uses NumPy arrays to build advanced mathematical tools
# Una guida con esempi si trova su https://www.youtube.com/watch?v=jmX4FOUEfgU&list=LL&index=2&ab_channel=Mr.PSolver
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy.special import jv
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import odeint  # c'è anche la libreria solveivp
from scipy.fft import fft, fftfreq
from scipy.stats import beta, multinomial
import scipy.stats as st


# MINIMIZATION
def f1(x):
    return (x - 3) ** 2


minimize(f1, np.array(4))  # 4 è l'initial guess
# Il risultato mostra: fun (valore della funzione nel minimo), x(punto di minimo) e altri attributi


def funzione2(x):
    return (x[0] - 3)**2 + (x[1] - 2.5)**2


def restrizione1(x):
    return x[0] - 2 * x[1] + 2


def restrizione2(x):
    return -x[0] - 2 * x[1] + 6


def restrizione3(x):
    return -x[0] + 2 * x[1] + 2


# Metto dei constraint alla funzione, sotto forma di tupla di dizionari, da fornire nella forma g(x)>=0
restrizioni = [{"type": "ineq", "fun": restrizione1},
               {"type": "ineq", "fun": restrizione2},
               {"type": "ineq", "fun": restrizione3}]
domini = ((0, None), (0, None))  # domini di x e y
print(minimize(funzione2, np.array([4, 2]), bounds=domini, constraints=restrizioni))  # è giusto, trascura l'errore


# INTERPOLATION: non si conosce la forma analitica (si vuole conoscere il valore della funzione tra i punti noti)
dati_x = np.array([0.5, 1, 2, 3, 4, 5])
dati_y = np.array([0.3, 1, 4, 8, 18, 24])
interpolation = interp1d(dati_x, dati_y, kind="quadratic")

plt.plot(dati_x, dati_y, color="green", linewidth=0.5, linestyle="",
         marker=".", markersize=6, label="data points")
dati_densi_x = np.arange(0.5, 5, 0.1)   # si può anche usare np.linspace(0.5, 5, 50) con 50 numero di punti
dati_densi_y_interpolati = interpolation(dati_densi_x)
plt.plot(dati_densi_x, dati_densi_y_interpolati, color="blue", label="theoretical values")
# plt.show()
plt.clf()
plt.close()


# CURVE FITTING: si conosce la forma analitica
def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


parametri_fittati, covarianza_parametri = curve_fit(parabola, dati_x, dati_y, p0=(1, 1, 1))
# L'output di curve fit è una lista di array con primo elemento i parametri, secondo elemento covarianza dei parametri
# Bisogna sempre controllare visivamente che il fit converga e cercare di dare parametri iniziali appropriati
print(parametri_fittati)
a_fit, b_fit, c_fit = parametri_fittati
err_a_fit, err_b_fit, err_c_fit = np.sqrt(np.diag(covarianza_parametri))
dati_densi_y_fittati = parabola(dati_densi_x, a_fit, b_fit, c_fit)

plt.plot(dati_x, dati_y, color="green", linewidth=0.5, linestyle="",
         marker=".", markersize=6, label="data points")
plt.plot(dati_densi_x, dati_densi_y_fittati, color="blue", label="theoretical values")
# plt.show()
plt.clf()
plt.close()

# SPECIAL FUNCTIONS
plt.plot(dati_densi_x / 4.5, legendre(6)(dati_densi_x / 4.5), color="green")  # polinomi di legendre
plt.plot(dati_densi_x * 2, jv(3, dati_densi_x * 2), color="red")  # funzioni di Bessel
# plt.show()
plt.clf()
plt.close()


# DIFFERENTIATION
def funzione3(x):
    return x ** 2 * np.sin(2 * x) * np.exp(-x)


print(derivative(funzione3, dati_x, dx=1e-6, n=2))  # n: order of derivative
# In teoria dovrebbe funzionare solo calcolata in un punto, ma funziona anche su un array di punti (anche se dà errore)


# INTEGRATION
[integrale, errore_integrale] = quad(funzione3, 0, 1)  # 0, 1 = xmin, xmax
print(integrale)
# Possiamo anche calcolare integrali 2d con estremi della prima integrazione dipendenti dall'altra variabile


def funzione4(y, x):  # attenzione: dobbiamo passare prima la variabile dell'integrale interno (y)
    return np.sin(x+y**2)


def lower_bound_y(x):
    return -x


def upper_bound_y(x):
    return x**2


[integrale_2d, errore_integrale_2d] = dblquad(funzione4, 0, 1, lower_bound_y, upper_bound_y)
print(integrale_2d)
# Similmente esiste nquad (sempre da scipy.integrate) per integrazioni n-dimensionali


# DIFFERENTIAL EQUATIONS
# Risolve solamente equazioni del primo ordine, ma può risolvere sistemi con qualsiasi numero di equazioni accoppiate
# Per risolvere equazioni del secondo ordine è possibile trasformarle in un sistema di due equazioni del primo ordine
def dy_dt(y, t):  # anche se l'equazione è autonoma, bisogna comunque specificare il parametro temporale t
    return 3*y**2 - 5 + 0 * t  # caduta libera con attrito


y0 = 0
tempi = np.linspace(0, 1, 100)
soluzione_ode1 = odeint(dy_dt, y0, tempi)  # il risultato è un array di array
print(np.transpose(soluzione_ode1)[0])  # per un'unica equazioni, c'è un unico elemento al posto 0
plt.plot(tempi, np.transpose(soluzione_ode1)[0])
# plt.show()
plt.clf()
plt.close()
# Si possono risolvere anche sistemi di equazioni differenziali accoppiate


def ds_dx(s, x):
    y1, y2 = s
    return np.array([y1+y2**2+3*x,
                     3*y1+y2**3-np.cos(x)])


y1_0 = 0
y2_0 = 0
s_0 = (y1_0, y2_0)
punti_x = np.linspace(0, 1, 100)
soluzione_ode2 = odeint(ds_dx, s_0, punti_x)  # è un array di array in cui ogni elemento è una coppia (y1(x_i), y2(x_i))
soluzione_y1 = np.transpose(soluzione_ode2)[0]
soluzione_y2 = np.transpose(soluzione_ode2)[1]
plt.plot(punti_x, soluzione_y1)
plt.plot(punti_x, soluzione_y2)
# plt.show()
plt.clf()
plt.close()


# DISCRETE FOURIER TRANSFORM
tempi_f = np.linspace(0, 10*np.pi, 100)
x_f = np.sin(2*np.pi*tempi_f) + np.sin(4*np.pi*tempi_f) + 0.1*np.random.randn(len(tempi_f))
N_f = len(x_f)
y_f = fft(x_f)[:N_f//2]   # fino a N_f//2 perchè serie reali sono simmetriche (// per usare N_f come float)
frequenze_f = fftfreq(N_f, np.diff(tempi_f)[0])[:N_f//2]  # il secondo argomento è il Deltat
# y_f sono i coefficienti di Fourier, frequenze_f sono le frequenze di Fourier
plt.plot(frequenze_f, np.abs(y_f))  # spettro di Fourier
# plt.show()
plt.clf()
plt.close()


# LINEAR ALGEBRA
# SciPy usa metodi più specifici ed efficienti di NumPy, dunque può essere utile per matrici molto grandi
# Ad esempio ci sono metodi per particolari tipi di matrici:
# x = solve_triangular(a, b, lower=True/False) per risolvere ax = b con a triangolare inferiore/superiore
# x = solve_toeplitz((c, r), b) per risolvere ax = b con a matrice di toeplitz
# E ci sono anche metodi per risolvere problemi agli autovalori con tali matrici
# Ci sono poi le varie decomposizioni LU, Choleski, ...


# STATISTICS
# In Scipy.stats ci sono tutte le pdf d'interesse
# norm(gaussiana), beta, multinomial, ...
a_beta, b_beta = 2.5, 3.1
mean_beta, var_beta, skew_beta, kurt_beta = beta.stats(a_beta, b_beta, moments="mvsk")
print([mean_beta, var_beta, skew_beta, kurt_beta])
beta_start = beta.ppf(0.01, a_beta, b_beta)  # ppf è l'inversa della cumulative distribution function (.cdf)
beta_finish = beta.ppf(0.99, a_beta, b_beta)  # si usa per settare intervalli in un plot (utile per domini illimitati)
beta_domain_points = np.linspace(beta_start, beta_finish, 100)
plt.plot(beta_domain_points, beta.pdf(beta_domain_points, a_beta, b_beta))
# plt.show()
plt.clf()
plt.close()

# Attenzione che per le distribuzioni discrete la nomenclatura corretta è pmf (probability mass function)
probability = np.ones(6)/6  # array di 1/6 per dado a 6 facce
print(multinomial.pmf([3, 2, 1, 0, 0, 0], n=6, p=probability))
# random variable sampling: genera numeri casuali dalla pdf/pmf data; genera size samples con n prova ciascuna
multinomial.rvs(n=100, p=probability, size=5)
# Si possono anche definire le proprie pdf:


class CustomDistribution(st.rv_continuous):  # eredito dalla classe generale delle distribuzioni continue
    # rv_continuous è costruita per accettare un qualsiasi numero di parametri, mentre noi ne vogliamo un numero fissato
    def _pdf(self, x, a1, a2, b1, b2):
        return 1/(2*(a1*b1+a2*b2))*(b1*np.exp(-np.sqrt(x/a1)) + b2*np.exp(-np.sqrt(x/a2)))


random_variable = CustomDistribution(a=0, b=np.inf)  # a e b sono i limiti del dominio
a1_rv, a2_rv, b1_rv, b2_rv = 2, 3, 1, 2
rv_points = np.linspace(random_variable.ppf(0.01, a1_rv, a2_rv, b1_rv, b2_rv),
                        random_variable.ppf(0.99, a1_rv, a2_rv, b1_rv, b2_rv), 100)
pdf_of_rv_points = random_variable.pdf(rv_points, a1_rv, a2_rv, b1_rv, b2_rv)
plt.plot(rv_points, pdf_of_rv_points)
plt.semilogy()
# plt.show()
plt.clf()
plt.close()
print(random_variable.rvs(a1_rv, a2_rv, b1_rv, b2_rv, size=100))

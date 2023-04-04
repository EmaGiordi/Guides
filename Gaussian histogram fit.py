import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#data = pd.read_excel(path, sheet_name=1, usecols="D,E")
s_data = np.random.normal(loc=0.77, scale=0.03, size=8000)
deltax = 0.0005
x_data = np.empty(8000)

for i in range(len(x_data)):
    x_data[i]=(i+1)*deltax
    
counts, bin_edges = np.histogram(s_data, bins=57, density=False, weights=None)

bin_width = bin_edges[1]-bin_edges[0]
bins = np.empty(len(bin_edges)-1)
for i in range(len(bins)):
    bins[i]=bin_edges[i]+bin_width/2
    
def Gauss (x, A, mu, sigma):
    return A/np.sqrt(2*np.pi*sigma**2)*np.e**(-(x-mu)**2/(2*sigma**2))

def multiwrite(outfile, string):
    outfile.write(string + "\n")
    print(string)
    
par, cov = curve_fit(Gauss, bins, counts, p0 = None)

fitted = Gauss(bins, par[0], par[1], par[2])

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(bins, counts, color="green", label="data points")
plt.plot(bins, fitted, color="red", label="fit")
plt.title("Titolo del grafico", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel("Asse x", fontdict={"fontsize": 14})
plt.ylabel("Asse y")
plt.legend(loc="upper left", fontsize=10, ncol=2)

with open("C:/Users/Emanuele/Desktop/Python/outputfile.txt", "w+") as outfile:
    multiwrite(outfile, "Fit:")
    multiwrite(outfile, "A = " + str(par[0]) + " with error " + str(np.sqrt(cov[0,0])))
    multiwrite(outfile, "mu = " + str(par[1]) + " with error " + str(np.sqrt(cov[1,1])))
    multiwrite(outfile, "sigma = " + str(par[2]) + " with error " + str(np.sqrt(cov[2,2])))
    multiwrite(outfile, "")

    
    
    
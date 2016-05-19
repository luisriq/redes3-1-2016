# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *

def random_array(n, zeros):
	a = np.random.randint(2, size=n)
	b = []
	j=0
	for i in range(len(a)):
		while j<zeros:
			b.append(0)
			j+=1
		j=0
		if(a[i]==0):
			b.append(-1)
		else:
			b.append(1)
	while j<zeros:
			b.append(0)
			j+=1
	return b


f, plots = plt.subplots(3)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc

T = 80
Fs = 1/T
print(Fs)
x = np.linspace(-T, T, (T/Fs))
signal = np.sinc(x)

plots[0].plot(x, signal, 'g')


# Fourier sinc
transformada = abs(fft(signal))
transformada = transformada*Fs*T
freqs = np.fft.fftfreq(len(x), Fs)
#freqs = np.fft.fftshift(freqs)
"""plots[1].plot(freqs, transformada/(T*0.5), 'r')
f.show()
a = []
print(x)"""
signal2 = random_array(10, T)
conv = np.convolve(signal, signal2)
plots[1].plot(signal2)
plots[0].grid(True)
plots[2].plot(conv)
plots[2].grid(True)
f.show()



input("Presione enter para salir:\n")


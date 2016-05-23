# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
import math
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

def fun_sinc(T):
	Fs = 1/T
	x = np.linspace(-T,T,(T/Fs))
	signal = np.sinc(x)
	return Fs, x, signal

def fun_prc(T, a):
	Fs, x, sinc = fun_sinc(T)
	signal = sinc * np.cos(a*np.pi*x)/(1-((4*(a**2)*(x**2))))
	print(sinc)
	print(signal)
	return Fs, x, signal

f, plots = plt.subplots(3)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc

#parte1
T=80
Fs, x, signal = fun_sinc(T)
Fs2, x2, signal_rc_25 = fun_prc(T, 0.25)
Fs2, x2, signal_rc_5 = fun_prc(T, 0.5)
Fs2, x2, signal_rc_1 = fun_prc(T, 1)

plots[0].plot(x, signal, 'b')
plots[0].plot(x2, signal_rc_25, 'g')
plots[0].plot(x2, signal_rc_5, 'r')
plots[0].plot(x2, signal_rc_1, 'c')
plots[0].grid(True)
#end parte 1


# Fourier sinc
transformada = abs(fft(signal))
transformada = transformada*Fs*T
freqs = np.fft.fftfreq(len(x), Fs)
signal2 = random_array(10, T)
conv = np.convolve(signal, signal2)
plots[1].plot(signal2)
plots[2].plot(conv)
plots[2].grid(True)

f.show()



input("Presione enter para salir:\n")


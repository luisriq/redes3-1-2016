# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *

"""
def random_array(n, zeros):
	a = np.random.randint(2, size=n)
	b = []
	zeros = zeros - 1
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
	return b"""
def random_array(n, zeros):
	z = np.zeros(zeros*(n+1))
	a = np.random.randint(2, size=n)
	for index, v in enumerate(a):
		if v == 0:
			z[zeros + index*zeros] = -1
		else:
			z[zeros + index*zeros] = 1
	return z
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
def diagramadeoho(signal, plott, T):
	offset = T
	for i in range(int(signal.size/T)):
		print("[%f:%f]"%(offset+i*T,offset+(i+1)*T))
		plott.plot(signal[T*0.5+i*T:T*0.5+(i+1)*T], 'b')

f, plots = plt.subplots(2)
a, plots2 = plt.subplots(3)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc

#parte1
T=81
Fs, x, signal = fun_sinc(T)
Fs2, x2, signal_rc_25 = fun_prc(T, 0.25)
Fs2, x2, signal_rc_5 = fun_prc(T, 0.5)
Fs2, x2, signal_rc_1 = fun_prc(T, 0.99)
#graficos en funcion del tiempo
plots[0].plot(x, signal, 'b')
plots[0].plot(x2, signal_rc_25, 'g')
plots[0].plot(x2, signal_rc_5, 'r')
plots[0].plot(x2, signal_rc_1, 'c')
plots[0].grid(True)

# Fourier sinc      		el /T*0.5 es para normalizar la transformada
transformada = abs(fft(signal))
transformada = transformada/(T*0.5)
transformada_rc_25 = abs(fft(signal_rc_25))
transformada_rc_25 = transformada_rc_25/(T*0.5)
transformada_rc_5 = abs(fft(signal_rc_5))
transformada_rc_5 = transformada_rc_5/(T*0.5)
transformada_rc_1 = abs(fft(signal_rc_1))
transformada_rc_1 = transformada_rc_1/(T*0.5)
freqs = np.fft.fftfreq(len(x), Fs)
#graficos en frecuencia
plots[1].plot(freqs, transformada, 'b')
plots[1].plot(freqs, transformada_rc_25, 'g')
plots[1].plot(freqs, transformada_rc_5, 'r')
plots[1].plot(freqs, transformada_rc_1, 'c')
plots[1].set_xlim([-3,3])	#xq asi da bonito
#end parte 1



#parte 2
cantidad_impulsos= 10
signal2 = random_array(cantidad_impulsos, T)
conv = np.convolve(signal, signal2)
plots2[0].plot(signal2)
plots2[1].plot(conv)
plots2[1].grid(True)
plots2[1].set_xlim([3200, 4200])
#
puntos=[ [T*0.5+i*T, conv[T*0.5+i*T]] for i in range(T+cantidad_impulsos)]
plots2[1].plot(*zip(*puntos), marker='o', color='r', ls='')
diagramadeoho(conv[3000:5000], plots2[2], T)
#f.show()

a.show()

input("Presione enter para salir:\n")


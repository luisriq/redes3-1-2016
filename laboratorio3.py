# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
import math
from nyq_criterion import *
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *
import matplotlib.patches as mpatches

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
def diagramadeoho(signal, plott, T,a, pulsos = 1):
	offset = -0.7
	lims = signal.size/(T*pulsos)
	print("%d,%d"%(T,signal.size/T))
	
	for i in range(int(lims)):
		plott.plot(signal[  (i-1)*T :(i+pulsos)*T],color='b')
def x_axis_conv(n,T): 
	"""Función para obtener el eje x del tren de pulsos y la convolución
	parametros:
		n: numero de pulsos
		T: periodo
	retorno: 
		z  : puntos del eje x del tren de pulsos
	"""
	z = np.linspace(0,n+1,T*(n+1))
	return z


f, plots = plt.subplots(2)
a, plots2 = plt.subplots(2)
b, plots3 = plt.subplots(4)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc

#parte1
T=80
Fs, x, signal = fun_sinc(2*T)
Fs2, x2, signal_rc_25 = fun_prc(2*T, 0.25)
Fs2, x2, signal_rc_5 = fun_prc(2*T, 0.5)
Fs2, x2, signal_rc_75 = fun_prc(2*T, 0.75)
Fs2, x2, signal_rc_1 = fun_prc(2*T, 0.99)
#graficos en funcion del tiempo
plots[0].plot(x, signal, 'b',  label='Line 2')
plots[0].plot(x2, signal_rc_25, 'g')
plots[0].plot(x2, signal_rc_5, 'r')
plots[0].plot(x2, signal_rc_75, 'c')
plots[0].plot(x2, signal_rc_1, 'm')
plots[0].grid(True)
l_signal = mpatches.Patch(color='b', label='Sinc')
l_signal_rc_25 = mpatches.Patch(color='g', label='RC a = 0.25')
l_signal_rc_5 = mpatches.Patch(color='r', label='RC a = 0.5')
l_signal_rc_75 = mpatches.Patch(color='c', label='RC a = 0.75')
l_signal_rc_1 = mpatches.Patch(color='m', label='RC a = 0.99')
plots[0].legend(handles=[l_signal,l_signal_rc_25,l_signal_rc_5,l_signal_rc_75,l_signal_rc_1],prop={'size':10})
plots[0].set_xlim([-T*.07, T*.07])
plots[0].set_xlabel("Tiempo")
plots[0].set_ylabel("Amplitud")
plots[0].set_title("Señales en dominio del tiempo")
# Fourier sinc      		el /T*0.5 es para normalizar la transformada
transformada = abs(fft(signal))
transformada = transformada/(T*0.5)
transformada_rc_25 = abs(fft(signal_rc_25))
transformada_rc_25 = transformada_rc_25/(T*0.5)
transformada_rc_5 = abs(fft(signal_rc_5))
transformada_rc_5 = transformada_rc_5/(T*0.5)
transformada_rc_75 = abs(fft(signal_rc_75))
transformada_rc_75 = transformada_rc_75/(T*0.5)
transformada_rc_1 = abs(fft(signal_rc_1))
transformada_rc_1 = transformada_rc_1/(T*0.5)
freqs = np.fft.fftfreq(len(x), Fs)
#graficos en frecuencia
plots[1].plot(freqs, transformada, 'b')
plots[1].plot(freqs, transformada_rc_25, 'g')
plots[1].plot(freqs, transformada_rc_5, 'r')
plots[1].plot(freqs, transformada_rc_75, 'c')
plots[1].plot(freqs, transformada_rc_1, 'm')
plots[1].grid(True)
plots[1].set_xlim([-3,3])
plots[1].legend(handles=[l_signal,l_signal_rc_25,l_signal_rc_5,l_signal_rc_75,l_signal_rc_1],prop={'size':10})
plots[1].set_xlabel("Frecuencia")
plots[1].set_ylabel("Amplitud")
plots[1].set_title("Señales en dominio de la frecuencia")
#end parte 1


#parte 2 a

cantidad_impulsos= 10
limitex = cantidad_impulsos+1
signal2 = random_array(cantidad_impulsos, T)
conv = np.convolve(signal_rc_75, signal2, 'same')
 
dm=(signal_rc_75.size-signal2.size)/2
conv = np.convolve(signal_rc_75[int(dm):int(signal_rc_75.size-dm)], signal2, 'same')
xx=x_axis_conv(cantidad_impulsos,T)


plots2[0].plot(signal2)
plots2[0].set_title("Tren de impulsos")
plots2[1].plot(xx, conv)
plots2[1].plot(xx, signal2,color='c')
plots2[1].grid(True)
plots2[1].set_xlim([0, limitex])
plots2[1].set_title("Señal resultante RC a=0.75")
plots2[1].set_xlabel("Tiempo")
plots2[1].set_ylabel("Amplitud")

# Parte 2 b
# Pulso RC a = 0.75

cantidad_impulsos= 10**3
signal2 = random_array(cantidad_impulsos, T)

dm=(signal_rc_75.size-signal2.size)/2
conv = np.convolve(signal_rc_75[int(dm):int(signal_rc_75.size-dm)], signal2, 'same')
xx=x_axis_conv(cantidad_impulsos,T)

#Ruido 
noise = np.random.normal(1, 0.1, conv.size)
noise_conv = conv*noise
"""
plots3[0].plot(xx, noise_conv)
plots3[0].plot(xx, signal2,color='c')
plots3[0].grid(True)
plots3[0].set_xlim([0, limitex])
plots3[0].set_title("Señal resultante RC a=0.75 con ruido")
plots3[0].set_xlabel("Tiempo")
plots3[0].set_ylabel("Amplitud")

"""


#
"""
puntos=[ [T*0.5+i*T, conv[T*0.5+i*T]] for i in range(T+cantidad_impulsos)]
plots2[1].plot(*zip(*puntos), marker='o', color='r', ls='')
puntos=[ [i*T, conv[i*T]] for i in range(T+cantidad_impulsos)]
plots2[1].plot(*zip(*puntos), marker='x', color='g', ls='')
"""
f.subplots_adjust( hspace=0.7 )
a.subplots_adjust( hspace=0.8 )
b.subplots_adjust( hspace=0.8 )
b.set_size_inches(7.5, 10.5, forward=True)


diagramadeoho(conv, plots3[0], T, a)
plots3[0].set_title("Diagrama de ojo RC a = 0.75")
diagramadeoho(noise_conv, plots3[1], T, a)
plots3[1].set_title("Diagrama de ojo RC a = 0.75 con ruido")

conv = np.convolve(signal[int(dm):int(signal.size-dm)], signal2, 'same')
#Ruido 
noise = np.random.normal(1, 0.1, conv.size)
noise_conv = conv*noise

diagramadeoho(conv, plots3[2], T, a)
plots3[2].set_title("Diagrama de ojo SINC")
diagramadeoho(noise_conv, plots3[3], T, a)
plots3[3].set_title("Diagrama de ojo SINC con ruido")

alphas = [.25, .50, .75, .99]
for al in alphas:
	nyq_criterion(al,2*T)
f.show()
a.show()
b.show()
f.savefig('parte1.eps')
a.savefig('parte2a.eps')
b.savefig('parte2b.eps')
input("Presione enter para salir:\n")


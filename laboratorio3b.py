# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *

def nyq_criterion(a,T,error=1e-12):
	if(a==.25):
		a=.25+1e-12
	for n in range(-T,T):
		prc_t =  np.sinc(n) * np.cos(a*np.pi*n)/(1-((4*(a**2)*(n**2))))
		if(n==0 and abs(prc_t-1)<=error):
			continue
		if(abs(prc_t) > error):
			print("No cumple con el criterio")
			return False

	print("Cumple con el criterio")
	return True
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

def parte_uno(T, plotear):
	f, plots = plt.subplots(2)
	#parte1
	Fs, x, signal = fun_sinc(2*T)
	Fs2, x2, signal_rc_25 = fun_prc(2*T, 0.25)
	Fs2, x2, signal_rc_5 = fun_prc(2*T, 0.5)
	Fs2, x2, signal_rc_75 = fun_prc(2*T, 0.75)
	Fs2, x2, signal_rc_1 = fun_prc(2*T, 0.99)
	#graficos en funcion del tiempo
	plots[0].plot(x, signal, 'b')
	plots[0].plot(x, signal_rc_25, 'g')
	plots[0].plot(x, signal_rc_5, 'r')
	plots[0].plot(x, signal_rc_1, 'c')
	plots[0].grid(True)
	plots[0].set_xlim([-5,5])	#xq asi da bonito
	plots[0].set_ylim([-0.3,1.2])	#xq asi da bonito

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
	if(plotear==1):
		f.show()

	return Fs, x, signal, signal_rc_25, signal_rc_5, signal_rc_75, signal_rc_1

a, plots2 = plt.subplots(3)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc

#parte1
T=80
Fs, x, signal, signal_rc_25, signal_rc_5, signal_rc_75, signal_rc_1 = parte_uno(T, 1)



#parte 2
#A
cantidad_impulsos_2= 10**3
signal2_1 = random_array(cantidad_impulsos_2, T)
dm=(signal_rc_1.size-signal2_1.size)/2
conv_2 = np.convolve(signal_rc_1[int(dm):int(signal_rc_1.size-dm)], signal2_1, 'same')
xx=x_axis_conv(cantidad_impulsos_2,T)
plots2[0].plot(xx,conv_2)
plots2[0].plot(xx,signal2_1,color='c')
#plots2[0].set_xlim(0, cantidad_impulsos_2+1)
plots2[0].grid(True)

# #B
cantidad_impulsos= 10**3
signal2 = random_array(cantidad_impulsos, T)
conv = np.convolve(signal_rc_1, signal2, 'same')
#plots2[0].plot(signal2_1)
plots2[1].plot(conv)
plots2[1].plot(signal2,color='c')
plots2[1].grid(True)
plots2[1].set_xlim([3200, 4200])
#

a.show()
diagramadeoho(conv, plots2[2], T, a)
a.savefig('out.png', transparent=False, bbox_inches='tight', pad_inches=0)
#nyq_criterion(.25,2*T)

input("Presione enter para salir:\n")


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



f, plots = plt.subplots(2)

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
Fs2, x2, signal_rc_75 = fun_prc(2*T, 0.75)
#graficos en funcion del tiempo
plots[0].plot(x, signal, 'b')
plots[0].plot(x+1, signal, 'g')
plots[0].plot(x+2, signal, 'r')
plots[0].plot(x-1, signal, 'c')
plots[0].plot(x-2, signal, 'm')
plots[0].grid(True)
plots[0].set_xlim([-5, 5])
plots[0].set_xlabel("Tiempo")
plots[0].set_ylabel("Amplitud")
plots[0].set_title("Varios pulsos sinc")
#
plots[1].plot(x2, signal_rc_75, 'b')
plots[1].plot(x2+1, signal_rc_75, 'g')
plots[1].plot(x2+2, signal_rc_75, 'r')
plots[1].plot(x2-1, signal_rc_75, 'c')
plots[1].plot(x2-2, signal_rc_75, 'm')
plots[1].grid(True)
plots[1].set_xlim([-5, 5])
plots[1].set_xlabel("Tiempo")
plots[1].set_ylabel("Amplitud")
plots[1].set_title("Varios pulsos RC con a = 0.75")


for xi in range(-2, 3):
	plots[0].axvline(x=xi,ymin=0.2857, ymax=1, color='black', linestyle='dashed')
	plots[1].axvline(x=xi,ymin=0.1666, ymax=1, color='black', linestyle='dashed')


f.subplots_adjust( hspace=0.7 )
f.savefig('test.eps')

f.show()

alphas = [.25, .50, .75, .99]
for al in alphas:
	nyq_criterion(al,2*T)
f.savefig('test.eps')
input("Presione enter para salir:\n")


# -*- coding: utf-8 -*-
#Luis Riquelme
#18.598.138-K
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *
# .-.-.-.-.-. Parametros para filtro FIR .-.-.-.-.-. #
# Rango de frecuencias no normalizadas
cutoff1 = 8000.0
cutoff2 = 22049.0
# Numero de coeficientes
orden = 50

def plotEspectrograma(datos, rate, plott, filtrar = False):
	plott.set_title("Espectrograma de la señal Original")
	#Filtrando la señal
	if(filtrar):
		# Frecuencia nyq por definicion rate/2
		nyq = rate / 2.
		# frecuencias cutoff normalizadas con nyq
		cutoff1_norm = cutoff1/nyq
		cutoff2_norm = cutoff2/nyq
		# Se crea el filtro con los coeficientes
		filtro = firwin(orden, [cutoff1_norm,cutoff2_norm], pass_zero=False)
		plott.set_title('Señal Original')
		#Filtrando la señal
		datos = (lfilter(filtro, 1, datos))
		plott.set_title('Escpectrograma Señal Filtrada')
	# Se genera el espectrograma
	plott.specgram(datos, Fs=rate, NFFT=256, )


def plotTiempoFIR(datos, rate, plott, filtrar = True):
	plott.set_title('Señal Original')
	#Filtrando la señal
	if(filtrar):
		# Frecuencia nyq por definicion rate/2
		nyq = rate / 2.
		# frecuencias cutoff normalizadas con nyq
		cutoff1_norm = cutoff1/nyq
		cutoff2_norm = cutoff2/nyq
		# Se crea el filtro con los coeficientes
		filtro = firwin(orden, [cutoff1_norm,cutoff2_norm], pass_zero=False)
		#Filtrando la señal
		datos = (lfilter(filtro, 1, datos))
		plott.set_title('Señal Filtrada')
		#Guardando la señal filtrada
		signal_guardar = np.asarray((datos), dtype=np.int16)
		write('beacon-filtrado.wav',rate , signal_guardar)
	#Graficar respecto al tiempo
	ar_tiempo = np.linspace(0, len(datos)/rate, len(datos))
	plott.plot(ar_tiempo, (datos), 'g')

f, plots = plt.subplots(2)

# Grafico debe estar normalizado 
# Eyediagram
# Graficar ocupando la convolucion
# pulsos (convolucionation *) filtro
# 

# Tiempo-> 5 periodos ??
# Sinc
T = 50
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
plots[1].plot(freqs, transformada/(T*0.5), 'r')
f.show()

print(x)

input("Presione enter para salir:\n")


# -*- coding: utf-8 -*-
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import matplotlib
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import *

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


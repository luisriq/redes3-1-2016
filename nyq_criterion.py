# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.fftpack import fft, fftfreq, ifft

def nyq_criterion(a,T,error=1e-12):
	for n in range(-T,T):
		if((1-((4*(a**2)*(n**2))))==0):
			a=a+1e-12
		prc_t =  np.sinc(n) * np.cos(a*np.pi*n)/(1-((4*(a**2)*(n**2))))
		if(n==0 and abs(prc_t-1)<=error):
			continue
		if(abs(prc_t) > error):
			print("RC con a=%f No cumple con el criterio"%a)
			return False

	print("RC con a=%f Cumple con el criterio"%a)
	return True


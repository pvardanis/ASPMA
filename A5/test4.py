import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "/home/pvardanis/sms-tools/software/models/"))
import dftModel as DFT 
import utilFunctions as UF
from scipy.fftpack import ifft
from scipy.signal import blackmanharris, triang

fs = 44100
Ns = 512 
hNs = int(Ns/2)
H = int(Ns/4)
ipfreq = np.array([4000.0]) # frequency of the sinusoid
ipmag = np.array([0.0])
ipphase = np.array([0.0])
Y = UF.genSpecSines_p(ipfreq, ipmag, ipphase, Ns, fs) # create the frequency spectrum of a blackmanharris window
y = np.real(ifft(Y))

sw = np.zeros(Ns) # create a triangular window for better hopping factor
ow = triang(Ns/2)
sw[hNs-H:hNs+H] = ow # center around the middle
bh = blackmanharris(Ns)
bh = bh / sum(bh)
sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

yw = np.zeros(Ns)
yw[:hNs-1] = y[hNs+1:]
yw[hNs-1:] = y[:hNs+1]
yw *= sw
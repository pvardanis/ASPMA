import numpy as numpy
import matplotlib.pyplot as pyplot
import os, sys
from scipy.signal import get_window
sys.path.append("/home/pvardanis/sms-tools/software/models/")
import utilFunctions as UF 
import stft as STFT

inputFile = "/home/pvardanis/sms-tools/sounds/flute-A4.wav"
window = 'hamming'
M = 801
N = 1024
H = 400

(fs, x) = UF.wavread(inputFile)

w = get_window(window, M)
mX, pX = STFT.stftAnal(x, w, N, H)
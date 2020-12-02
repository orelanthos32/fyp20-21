import pandas as pd
import numpy as np
from ADL import*
from scipy.io import loadmat
annots = loadmat('sea.mat')
import matlab.engine
eng = matlab.engine.start_matlab()

chunkSize = 500.0       
epoch = 1.0              
alpha_w = 0.0005       
alpha_d = 0.0001       
delta   = 0.05
I = 3.0

p,d = eng.ADL(annots['data'].tolist(),I,chunkSize,epoch,alpha_w,alpha_d,delta)
import numpy as np
import TimeSeriesCode as TS
import datetime as dt
from EGARCH import EGARCHKernelModel
from MarginalKernelModel import KernelEstimateCDMLE

# Filters only main session
openingtime = dt.time(9,30) #AAPL opens at 9:30
closingtime = dt.time(16,0) #AAPL closes at 16:00

# Data loading and preprocessing
dataTS = TS.TimeSeries('AAPL.txt', openingtime, closingtime)
dayindeces, timeindeces, returns = dataTS.getlowerfrequencyreturns(30)

# Init guess
bandwidthh, bandwidthg = np.array([0.39, 0.41])
EGARCH_initguess = np.array([0.88465807, -0.06568638,  0.29727846])


# Bandwidth selection via CV
bandwidthh, bandwidthg = KernelEstimateCDMLE(returns, timeindeces, initGuess = np.array([bandwidthh, bandwidthg])).bandwidths

# Model Estimation
modelEst = EGARCHKernelModel(returns, dayindeces, timeindeces, bandwithh = bandwidthh, bandwithg = bandwidthg, initguess = EGARCH_initguess, nbatches = 10,
                 ntrajectories = 1000, nperiods = 200, perror = 0.00001)

# Results
print("---Estimated marginal normal EGARCH---")
print("Model parameters [return bandwidth, time bandwidth, beta_1, beta_2, beta_3]")
print(modelEst.getCoefficients())

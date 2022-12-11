import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class KernelEstimate(object):
    """
    Class for kernel estimate of the marginal distribution.
    """
    def __init__(self, logreturns, intradayperiods, bandwithh = None, bandwithg = None):
        self.logreturns = logreturns
        self.intradayperiods = intradayperiods
        self.sigma = np.std(logreturns)
        self.n = self.logreturns.size
        self.nperiods = int(np.unique(intradayperiods).size)
        self.Stau = np.zeros(self.n)
        self.StauU = np.zeros(self.nperiods)

        for tau in np.arange(self.nperiods):
            self.StauU[tau] = np.std(logreturns[tau == intradayperiods])
            self.Stau[tau == intradayperiods] = self.StauU[tau]

        if bandwithh == None:
            self.h = logreturns.size ** (-1. / 6.)
        else:
            self.h = bandwithh
        if bandwithg == None:
            self.g = logreturns.size**(-1./6.)*np.sqrt(1./12.)
        else:
            self.g = bandwithg
        self.taumax = np.max(intradayperiods)
        self.taunorm = intradayperiods/self.taumax
        self.logreturnsstandardized = logreturns/self.Stau

    def cdf(self, logreturns, intradayperiods):
        inputtaus = np.array([intradayperiods/self.taumax] * self.n)
        inputStau = np.zeros(logreturns.size)
        for tau in np.arange(self.nperiods):
            inputStau[tau == intradayperiods] = self.StauU[tau]
        inputlogreturnsstandardized = np.array([logreturns/inputStau] * self.n)
        t = logreturns.size
        mlogreturns = np.array([self.logreturnsstandardized] * t).T
        mtaus = np.array([self.taunorm] * t).T
        taupdf = norm.pdf((inputtaus-mtaus)/self.g)
        denominator = np.mean(taupdf, axis = 0)
        numerator = np.mean(norm.cdf((inputlogreturnsstandardized-mlogreturns)/self.h)*taupdf, axis = 0)

        return numerator/denominator

    def cdfbatches(self, logreturns, intradayperiods, nbatches):
        t = logreturns.size
        ipoints = np.arange(0, t, int(t/nbatches))
        result = np.zeros(t)
        for i in ipoints[:-1]:
            result[i:i+int(t/nbatches)] = self.cdf(logreturns[i:i+int(t/nbatches)], intradayperiods[i:i+int(t/nbatches)])
        result[ipoints[-1]:] = self.cdf(logreturns[ipoints[-1]:], intradayperiods[ipoints[-1]:])

        return result

    def pdf(self, logreturns, intradayperiods):
        inputtaus = np.array([intradayperiods/self.taumax] * self.n)
        inputStau = np.zeros(logreturns.size)
        for tau in np.arange(self.nperiods):
            inputStau[tau == intradayperiods] = self.StauU[tau]
        inputlogreturnsstandardized = np.array([logreturns/inputStau] * self.n)
        inputStau = np.array([inputStau] * self.n)
        t = logreturns.size
        mlogreturns = np.array([self.logreturnsstandardized] * t).T
        mtaus = np.array([self.taunorm] * t).T
        taupdf = norm.pdf((inputtaus-mtaus)/self.g)
        denominator = np.mean(taupdf, axis = 0)
        numerator = np.mean(norm.pdf((inputlogreturnsstandardized-mlogreturns)/self.h)*taupdf/(inputStau*self.h), axis = 0)
        return numerator/denominator

class KernelEstimateCDMLE(object):
    """
    Object performing cross-validation on initialization.
    """
    def __init__(self, logreturns, intradayperiods, initGuess = None):
        self.logreturns = logreturns
        self.intradayperiods = intradayperiods
        self.sigma = np.std(logreturns)
        self.n = self.logreturns.size
        self.nperiods = int(np.unique(intradayperiods).size)
        self.Stau = np.zeros(self.n)
        self.StauU = np.zeros(self.nperiods)

        for tau in np.arange(self.nperiods):
            self.StauU[tau] = np.std(logreturns[tau == intradayperiods])
            self.Stau[tau == intradayperiods] = self.StauU[tau]

        self.logreturnsstandardized = logreturns / self.Stau
        self.taumax = np.max(intradayperiods)
        self.taunorm = intradayperiods / self.taumax

        #Preparing matrices
        tausm = np.array([self.taunorm] * self.n)
        inputlogreturnsstandardized = np.array([self.logreturnsstandardized] * self.n)
        self.inputStau = np.array([self.Stau] * self.n)
        mlogreturns = np.array([self.logreturnsstandardized] * self.n).T
        mtaus = np.array([self.taunorm] * self.n).T
        self.taudif = tausm-mtaus
        self.retdiff = inputlogreturnsstandardized - mlogreturns

        #bandwidth optimization
        if initGuess is None:
            initGuess = np.array([logreturns.size ** (-1. / 6.), logreturns.size ** (-1. / 6.) * np.sqrt(1. / 12.)])

        self.bandwidths = minimize(self.negativeloglikelihood, initGuess, method='Nelder-Mead',
                 options={'maxiter': initGuess.size * 2000, 'maxfev': initGuess.size * 2000}).x

    def pdf(self, h, g):
        taupdf = norm.pdf((self.taudif)/g)
        denominator = np.mean(taupdf, axis = 0)
        numerator = np.mean(norm.pdf((self.retdiff)/h)*taupdf/(self.inputStau*h), axis = 0)
        return numerator / denominator

    def pdfJK(self, h, g):
        taupdf = norm.pdf((self.taudif) / g)
        numerator = norm.pdf((self.retdiff) / h) * taupdf / (self.inputStau * h)
        np.fill_diagonal(taupdf, 0)
        np.fill_diagonal(numerator, 0)
        return np.mean(numerator,axis = 0)/np.mean(taupdf, axis = 0)

    def negativeloglikelihood(self, bandwidths):
        return -np.sum(np.log(self.pdfJK(bandwidths[0], bandwidths[1])))
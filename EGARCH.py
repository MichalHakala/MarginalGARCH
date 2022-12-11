import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from MarginalKernelModel import KernelEstimate

# Model wrapper
class EGARCHKernelModel(object):
    """
    Main class for estimating the Marginal EGARCH model.
    """
    def __init__(self, returns, dayindeces, timeindeces, bandwithh = None, bandwithg = None,
                 initguess = np.array([0.8, 0.05, -0.025]), nbatches = 10, ntrajectories = 1000, nperiods = 200, perror = 0.00001):
        self.returns = returns
        self.dayindeces = dayindeces
        self.daylist = np.unique(dayindeces)
        self.D = self.daylist.size
        self.timeindeces = timeindeces
        self.timelist = np.unique(timeindeces)
        self.max = np.max(np.abs(returns))

        self.kernelEst = KernelEstimate(returns, timeindeces, bandwithh, bandwithg)
        self.p = self.kernelEst.cdfbatches(returns, timeindeces, nbatches)

        self.GARCH = NEGARCHMarginalModel(self.p, initguess, ntrajectories, nperiods, perror)

    def onestepPredictionQ(self, q):
        return self.GARCH.onestepPredictionP(q)

    def onestepPredictionR(self, q, tau):
        rp = np.zeros(q.size)
        for i in np.arange(q.size):
            tempp = self.onestepPredictionQ(q[i])
            rp[i] = brentq(lambda x: tempp - self.kernelEst.cdf(np.array([x]),np.array([tau]))[0], -self.max*2.0, self.max*2.0)

        return rp

    def onestepPredictionCum(self, q, r, tau):
        p = self.kernelEst.cdf(np.array([r]), np.array([tau]))
        eps = self.GARCH.onestepPredictionResidual(p)
        ut = norm.cdf(eps)
        return (1.0/q)*(q-ut)*(ut <= q)

    def getCoefficients(self):
        return np.array([self.kernelEst.h, self.kernelEst.g, self.GARCH.par[0],self.GARCH.par[1], self.GARCH.par[2]])

    def tramsformLatentToReturn(self, x, tau):
        p = self.GARCH.marg.computeCDFPDF(x)[0]
        rp = np.zeros(p.size)
        for i in np.arange(p.size):
            rp[i] = brentq(lambda ret: p[i] - self.kernelEst.cdf(np.array([ret]),np.array([tau]))[0], -self.max*4.0, self.max*4.0)

        return rp

    def conditionalMoments(self, sigma, tau, egrid = np.arange(-5,5, 0.1)):
        deltae = egrid[1:] - egrid[:-1]
        emids = egrid[:-1]+deltae/2
        edensityrectangles = norm.pdf(emids)*deltae
        logreturns = self.tramsformLatentToReturn(emids*sigma, tau)

        mean = np.sum((logreturns)*edensityrectangles)
        variance = np.sum((logreturns**2.0)*edensityrectangles)-mean**2.0
        kurtosis = np.sum((((logreturns-mean)/np.sqrt(variance))**4.0)*edensityrectangles)

        return variance, kurtosis



# Dynamic Model
class NEGARCHMarginalModel(object):

    def __init__(self, p, initGuess, ntrajectories, nperiods, perror):
        self.marg = NEGARCHmarginal(ntrajectories, nperiods, p, perror)
        self.n = p.size
        self.p = p


        # prestimation
        self.par = minimize(self.loglikelihoodS, initGuess, method='Nelder-Mead').x

        """
        #self.parMiss = minimize(self.loglikelihoodMiss, initGuess, method='Nelder-Mead').x
        self.margF = symmetricNGARCHmarginal(ntrajectories, nperiods, p, perror)
        self.margF.simulateSigma(self.par)
        self.eSq = self.marg.quantileFun(self.p, self.par) ** 2.0
        """

    def loglikelihoodS(self, par):
        beta1, beta2, beta3 = par
        if beta1 >= 1.0:
            return np.inf

        self.marg.simulateSigma(np.array(par))
        self.marg.computeQuantiles()

        x = self.marg.quantileFun(self.p)
        error = np.zeros(self.n)
        error[0] = x[0]
        logsigmaSq = np.zeros(self.n)
        sigma = np.ones(self.n)

        sigma[0] = np.std(x)
        logsigmaSq[0] = np.log(sigma[0]**2.0)

        for t in np.arange(1, self.n):
            error_absolute = np.abs(error[t-1])
            logsigmaSq[t] = beta1 * logsigmaSq[t - 1] + beta2 * error[t-1] + beta3*error_absolute

            sigma[t] = np.exp(logsigmaSq[t]/2.0)
            error[t] = x[t] / sigma[t]

        self.sigma = sigma
        self.x = x
        lt = -np.log(sigma) - 0.5 * error**2.0 + np.log(self.marg.quantileFunDer(self.p))
        loglikelihood = np.sum(lt)

        print("-----Log-likelihood Maximization------")
        print("Loglikelihood: " + str(loglikelihood))
        print("[beta_1, beta_2, beta_3]")
        print(par)
        return -loglikelihood


    def onestepPredictionP(self, q):
        fsigmaSq = self.onestepPredictionSigmaSq()
        return self.margF.computeCDFPDF(norm.ppf(q, 0.0, np.sqrt(fsigmaSq)))[0]

    def onestepPredictionSigmaSq(self):
        beta1, beta2 = self.par
        beta0 = 1 - beta1 - beta2

        sigmaSq = np.zeros(self.n)
        sigmaSq[0] = 1.0

        for t in np.arange(1, self.n):
            sigmaSq[t] = beta0 + beta2 * self.eSq[t - 1] + beta1 * sigmaSq[t - 1]
        fsigmaSq = beta0 + beta2 * self.eSq[-1] + beta1 * sigmaSq[-1]
        return fsigmaSq

    def onestepPredictionResidual(self, p):
        return self.marg.quantileFun( np.array([p]), self.par)/np.sqrt(self.onestepPredictionSigmaSq())


# Marginal Distribution of EGARCH model
class NEGARCHmarginal(object):
    """
    Class for marginal distribution of normal EGARCH.
    ntrajectories represents number of simulated sigma over nperiods.
    Quantile function and its derivative are solved on a grid of p. The grid should match the empirical values of p_t,
    or significantly finer grid should be provided. Quantile function returns a point closest to the required p_t.
    Newton-Rhapson algorithm solving for the quantile function stops when
    |smallest p - CDF of smallest p| <= perror

    How to use the class:
    1. initlize an object with required numerical settings
    2. call method simulateSigma with required parameters
    3. call computeQuantiles to numerically solve for quantile function and its derivative


    Repeat 2. and 3. if update of parameters is required (e.g., in maximum likelihood estimation optimization).
    """

    def __init__(self, ntrajectories, nperiods, p, perror):
        """
        :param ntrajectories: int, number of simulated trajectories for sigma
        :param nobservations: int, number of observations from each trajectory
        :param p: numpy array float, grid of input parameters to a quantile function
        :param perror: float, convergence when smallest p
        """
        #arbitrary beta vector
        self.betas = np.empty(3)
        #initialize random numbers, matrix with variance trajectories and empty vector for observations
        self.trajectoriesRN = np.random.normal(0.0, 1.0, np.array([ntrajectories, nperiods+1]))
        self.trajectoriesRNabs = np.abs(self.trajectoriesRN)

        #save dimensions
        self.ntrajectories = ntrajectories
        self.nperiods = nperiods
        self.sortedp = np.sort(p)
        self.incrementP = self.sortedp[1:] - self.sortedp[:-1]
        self.perror = perror
        self.npoints = p.size

        self.odd = (p.size % 2 == 1)
        self.midindex = np.floor(p.size / 2).astype(int)
        self.quantile = np.empty(self.sortedp.size)
        self.quantileDer = np.empty(self.sortedp.size)

        self.generated_sigma = False
        self.generated_quantiles = False

    def simulateSigma(self, betas):
        """
        beta1 - lagged logvolatility
        beta2 - lagged innovation
        beta3 - absolute value of lagged innovation
        :param betas: [beta1, beta2, beta3]
        :return:
        """

        self.betas = betas
        self.logsigmaSq = np.zeros(self.ntrajectories)
        for t in np.arange(self.nperiods):
            self.logsigmaSq = self.betas[0] * self.logsigmaSq + self.betas[1] * self.trajectoriesRN[:, t] + \
                              self.betas[2] * self.trajectoriesRNabs[:, t]
        self.sigma = np.exp(self.logsigmaSq/2)

        self.generated_sigma = True
        self.generated_quantiles = False

    def computeCDFPDF(self, x):
        """
        Computes CDF and PDF of the marginal distribution.

        :param x: scalar or array with input argument to CDF and PDF
        :return: array, [CDF, PDF]
        """
        if not self.generated_sigma:
            ValueError("No sigmas available. Call simulateSigma first.")
        if isinstance(x, (list, tuple, np.ndarray)):
            xm = np.tile(x, [self.sigma.size,1])
            sigmam = np.tile(self.sigma, [x.size,1]).T
            fraction = xm/sigmam
            CDF = np.mean(norm.cdf(fraction), axis= 0)
            pdf = np.mean(norm.pdf(fraction)/sigmam, axis = 0)
            return CDF, pdf
        else:
            fraction = x/self.sigma
            CDF = np.mean(norm.cdf(fraction))
            pdf = np.mean(norm.pdf(fraction)/self.sigma)
            return CDF, pdf

    def computeQuantiles(self):

        if not self.generated_sigma:
            ValueError("No sigmas available. Call simulateSigma first.")

        self.quantile[self.midindex] = 0.0

        # Init point searched by Newton-Rhapson
        CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex])
        while ((CDF - self.sortedp[self.midindex]) > self.perror):
            CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex])
            self.quantile[self.midindex] = self.quantile[self.midindex] - (CDF - self.sortedp[self.midindex]) / pdf

        if self.odd:
            #Initial guess for the whole grid by linear approximation
            for i in np.arange(self.midindex)+1:
                CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex - i+1])
                self.quantile[self.midindex - i] = self.quantile[self.midindex - i+1] - (self.incrementP[self.midindex - i])/pdf
                CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex + i-1])
                self.quantile[self.midindex + i] = self.quantile[self.midindex + i - 1] + (self.incrementP[self.midindex + i - 1])/pdf

        else:
            #Initial guess for the whole grid by linear approximation
            for i in np.arange(self.midindex-1)+1:
                CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex - i+1])
                self.quantile[self.midindex - i] = self.quantile[self.midindex - i+1] - (self.incrementP[self.midindex - i])/pdf
                CDF, pdf = self.computeCDFPDF(self.quantile[self.midindex + i-1])
                self.quantile[self.midindex + i] = self.quantile[self.midindex + i - 1] + (self.incrementP[self.midindex + i - 1])/pdf
            #missing first value
                CDF, pdf = self.computeCDFPDF(self.quantile[1])
                self.quantile[0] = self.quantile[1] - (self.incrementP[0]) / pdf

        #Newton-Rhapson whole quantile function
        CDF, pdf = self.computeCDFPDF(self.quantile)
        self.quantile = self.quantile - (CDF - self.sortedp) / pdf
        while(np.abs(CDF[0]-self.sortedp[0])>self.perror):
            CDF, pdf = self.computeCDFPDF(self.quantile)
            self.quantile = self.quantile - (CDF - self.sortedp) / pdf
        self.quantileDer = 1/pdf

        self.generated_quantiles = True

    def quantileFun(self, p):
        if not self.generated_quantiles:
            ValueError("No quantiles available. Run computeQuantiles first.")
        return self.quantile[np.round(p*(self.npoints-1)).astype(int)]

    def quantileFunDer(self, p):
        if not self.generated_quantiles:
            ValueError("No quantile function derivatives available. Run computeQuantiles first.")
        return self.quantileDer[np.round(p*(self.npoints-1)).astype(int)]
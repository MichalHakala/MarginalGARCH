import numpy as np
import datetime as dt


class TimeSeries(object):

    def __init__(self, fname, openingtime, closingtime):
        """
        Loads data from file fname and creates TimeSeries object. No header. Each line represents one observation.
        Expected format: date,time, open, high, low, close, volume.
        Example of expected format: 10/29/2015,04:00,110.91,110.91,110.59,110.59,242
        :param fname: string with path to datafile.
        """
        lines = open(fname).read().splitlines()

        self.ntotal = len(lines) #number of observations

        #Main market session data
        self.datetime = []
        self.open, self.high, self.low, self.close, self.volume = [], [], [], [], []

        #Pre-market and after-market session data
        self.datetimeO = []
        self.openO, self.highO, self.lowO, self.closeO, self.volumeO = [], [], [], [], []

        for i in np.arange(self.ntotal):
            tempobs = lines[i].split(',')
            tempdatetime = dt.datetime.strptime(tempobs[0]+tempobs[1], '%m/%d/%Y%H:%M')

            if openingtime <= tempdatetime.time() < closingtime:
                #main market session
                self.datetime.append(tempdatetime)
                self.open.append(float(tempobs[2]))
                self.high.append(float(tempobs[3]))
                self.low.append(float(tempobs[4]))
                self.close.append(float(tempobs[5]))
                self.volume.append(int(tempobs[6]))
            else:
                #other market sessions
                self.datetimeO.append(tempdatetime)
                self.openO.append(float(tempobs[2]))
                self.highO.append(float(tempobs[3]))
                self.lowO.append(float(tempobs[4]))
                self.closeO.append(float(tempobs[5]))
                self.volumeO.append(int(tempobs[6]))

        self.n = len(self.datetime) #number of observations in main session
        self.nO = len(self.datetimeO) #number of observations in other sessions
        self.datetime, self.datetimeO = np.array(self.datetime), np.array(self.datetimeO)
        self.open, self.high, self.low, self.close, self.volume = np.array(self.open), np.array(self.high), \
                                                                  np.array(self.low), np.array(self.close), \
                                                                  np.array(self.volume)
        self.openO, self.highO, self.lowO, self.closeO, self.volumeO = np.array(self.openO), np.array(self.highO), \
                                                                  np.array(self.lowO), np.array(self.closeO), \
                                                                  np.array(self.volumeO)

    def gettimesindeces(self):
        """
        Transform array of datetimes to array of intraday indeces, e.g. [0,1,2,3, .., tau_max, 0,1,2,...]
        :return: numpy array
        """
        times = np.array([dt.time() for dt in self.datetime])
        uniquetimes = np.array(sorted(np.unique(times)))

        indeces = np.zeros(self.datetime.size)
        for tau in np.arange(uniquetimes.size):
            indeces[times == uniquetimes[tau]] = tau

        return indeces.astype(int)

    def getdayindeces(self):
        """
        Transform array of datetimes to array of daily indeces, e.g. [0,0, ..,0,1,1,1,1, .., 1,2,2,2,...]
        :return: numpy array
        """
        dates = np.array([dt.date() for dt in self.datetime])
        uniquedates = np.array(sorted(np.unique(dates)))

        indeces = np.zeros(self.datetime.size)
        for d in np.arange(uniquedates.size):
            indeces[dates == uniquedates[d]] = d

        return indeces.astype(int)

    def getlogreturns(self):
        """
        Returns raw array of log returns without any formatting or inputation of missing data.
        :return:
        """
        logreturns = np.log(self.close/self.open)

        return logreturns

    def getovernightreturns(self):
        """
        Coomputes array of overnight log returns => D-1 elements.
        E.g. 1st element is log( Close_{0, tau_max} / Open_{1, 0} ).
        :return: numpy array
        """

        dayindeces = self.getdayindeces()
        uniquedays = np.unique(dayindeces)
        overnightreturns = np.zeros(uniquedays.size-1)
        for i in np.arange(uniquedays.size-1):
            d = uniquedays[i]
            lastclose = self.close[dayindeces == d][-1]
            firstopen = self.open[dayindeces == d+1][0]
            overnightreturns[d] = np.log(firstopen/lastclose)

        return overnightreturns

    def getlowerfrequencyreturns(self, periodlength):
        """
        Compute logreturns for a given periodlength. Periodlength represents number of periods from the original raw data.
        The new return series has no missing values. When no observations in a given low frequency period are available, 0 is used instead.
        :param periodlength: integer number >= 1, number of original periods in one new period.
        :return: list [array of daily indeces, array of intraday indeces, logreturns]
        """
        newdayindeces, newperiodindeces, logreturns = [], [], []
        dayindeces = self.getdayindeces()
        uniquedays = np.unique(dayindeces)
        timesindeces = self.gettimesindeces()
        taumax = np.max(timesindeces)

        taus = np.arange(0, taumax+1, periodlength) #array of periods that should be sampled for new data

        for d in uniquedays:
            tempopen = self.open[dayindeces == d]
            tempclose = self.close[dayindeces == d]
            temptimesindeces = timesindeces[dayindeces == d]
            for i in np.arange(taus.size):
                #periods that should be sampled in an ideal case - there may be some periods missing
                tauopen = taus[i]
                tauclose = taus[i]+periodlength-1

                #Loop that will find first open in given low-frequency period
                for k in np.arange(taumax):
                    if tauopen > tauclose:
                        addopen = 1.0
                        break
                    if tauopen in temptimesindeces:
                        addopen = tempopen[np.where(tauopen == temptimesindeces)][0]
                        break
                    tauopen += 1

                # Loop that will find last close in given low-frequency period
                for k in np.arange(taumax):
                    if tauopen > tauclose:
                        addclose = 1.0
                        break
                    if tauclose in temptimesindeces:
                        addclose = tempclose[np.where(tauclose == temptimesindeces)][0]
                        break
                    tauclose -= 1

                logreturn = np.log(addclose/addopen)
                logreturns.append(logreturn)
                newdayindeces.append(d)
                newperiodindeces.append(i)

        return np.array(newdayindeces), np.array(newperiodindeces), np.array(logreturns)

    def getrealizedvariance(self, periodlength):
        """
        Computes realized variances of main trading session.
        Periodlength specify number of periods of raw data used for one observation.
        E.g., 1 minute raw data and periodlength=5 computes realized variance by squaring 5-minute returns.
        :param periodlength: integer number >= 1, number of original periods in one new period.
        :return: list of numpy array with realized variance by days and numpy array of day indeces.
        """
        dayindeces, periodindeces, logreturns = self.getlowerfrequencyreturns(periodlength)
        returnsSq = logreturns**2.0
        uniquedays = np.unique(dayindeces)
        realizedvariance = np.zeros(uniquedays.size)

        for d in np.arange(uniquedays.size):
            tempreturnsSq = returnsSq[dayindeces == uniquedays[d]]
            realizedvariance[d] = np.sum(tempreturnsSq)

        return realizedvariance, uniquedays

    def gettotalrealizedvariance(self, periodlength):
        """
        Computes realized variance with squared overnight return.
        Periodlength specify number of periods of raw data used for one observation.
        E.g., 1 minute raw data and periodlength=5 computes realized variance by squaring 5-minute returns.
        Note, this function computes D-1 values, where D is number of days.
        The overnight return for the last day is missing.
        :param periodlength: integer number >= 1, number of original periods in one new period.
        :return: list of numpy array with realized variance by days and numpy array of day indeces.
        """
        realizedvariance, dayindeces = self.getrealizedvariance(periodlength)
        overnightreturnsSq = self.getovernightreturns()**2.0
        totalrealizedvariance = realizedvariance[:-1]+overnightreturnsSq

        return totalrealizedvariance, dayindeces[:-1]

    def getstandardizedreturns(self, periodlength, RVperiodlength):
        """
        Computes returns and realized volatility for standardization (with overnight return) of the previous day.
        Data of the first day is dropped due to non-existing realized volatility.
        :param periodlength: int, period length for obtaining returns, see getlowerfrequencyreturns.
        :param RVperiodlength: int, period length for obtaining realized volatility, see gettotalrealizedvariance.
        :return:
        """
        totalRV, dayindecesRV = self.gettotalrealizedvariance(RVperiodlength)
        totalRvol = np.sqrt(totalRV)
        dayindeces, periodindeces, logreturns = self.getlowerfrequencyreturns(periodlength)
        returns = np.zeros(logreturns.size)

        for d in dayindecesRV.astype(int):
            returns[dayindeces == d+1]=logreturns[dayindeces == d+1]/totalRvol[d]
        return returns[dayindeces > 0], dayindeces[dayindeces > 0], periodindeces[dayindeces > 0]

    def getmovingrealizedvariance(self, periodlength):
        """
        Computes moving realized variance over the last day.
        Periodlength specify number of periods of raw data used for one observation.
        E.g., 1 minute raw data and periodlength=5 computes realized variance by squaring 5-minute returns.
        Returns number of periods*(number of days-1) values.
        The first number of periods observations is dropped, since there is no overnight return for the first day.
        Overnight returns are not included.
        :param periodlength: integer number >= 1, number of original periods in one new period.
        :return: list of numpy array with moving realized variance, numpy array of day indeces,
        and numpy array of intraday period indeces.
        """
        dayindeces, periodindeces, logreturns = self.getlowerfrequencyreturns(periodlength)
        returnsSq = logreturns ** 2.0
        nperiods = np.unique(periodindeces).size
        realizedvariance = np.zeros(returnsSq.size)

        for i in np.arange(nperiods,returnsSq.size):
            realizedvariance[i] = np.sum(returnsSq[i-nperiods:i])

        return realizedvariance[nperiods:], dayindeces[nperiods:], periodindeces[nperiods:]

    def gettotalmovingrealizedvariance(self, periodlength):
        """
        Computes moving realized variance over the last day.
        Periodlength specify number of periods of raw data used for one observation.
        E.g., 1 minute raw data and periodlength=5 computes realized variance by squaring 5-minute returns.
        Returns number of periods*(number of days-1) values.
        The first number of periods observations is dropped, since there is no overnight return for the first day.
        Overnight returns are included.
        :param periodlength: integer number >= 1, number of original periods in one new period.
        :return: list of numpy array with moving realized variance, numpy array of day indeces,
        and numpy array of intraday period indeces.
        """
        realizedvariance, dayindeces, periodindeces = self.getmovingrealizedvariance(periodlength)
        days = np.unique(dayindeces)
        overnightreturnsSq = self.getovernightreturns()**2.0

        for d in np.arange(np.min(days),np.max(days)):
            if d <= 0:
                continue
            else:
                realizedvariance[dayindeces == d] = realizedvariance[dayindeces == d]+overnightreturnsSq[d-1]

        return realizedvariance, dayindeces, periodindeces

    def getstandardizedreturnsEWMA(self, periodlength, RVperiodlength, EWMApersistence = 0.96):
        """
        Computes returns standardized by exponentialy weighted moving average of realized volatility of the previous day.
        Data of the first day is dropped due to non-existing realized volatility.
        :param periodlength: int, period length for obtaining returns, see getlowerfrequencyreturns.
        :param RVperiodlength: int, period length for obtaining realized volatility, see gettotalrealizedvariance.
        :return:
        """
        totalRV, dayindecesRV = self.gettotalrealizedvariance(RVperiodlength)
        totalRvol = np.sqrt(totalRV)
        dayindeces, periodindeces, logreturns = self.getlowerfrequencyreturns(periodlength)
        returns = np.zeros(logreturns.size)
        EWMA = np.mean(totalRvol)

        for d in dayindecesRV.astype(int):
            EWMA = EWMA*EWMApersistence+(1-EWMApersistence)*totalRvol[d]
            returns[dayindeces == d+1]=logreturns[dayindeces == d+1]/EWMA
        return returns[dayindeces > 0], dayindeces[dayindeces > 0], periodindeces[dayindeces > 0]

















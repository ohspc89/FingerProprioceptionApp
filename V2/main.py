import math, time, copy
import numpy as np
from random import randrange
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty, ListProperty, NumericProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.storage.jsonstore import JsonStore
from kivy.uix.checkbox import CheckBox
from kivy import platform
from kivy.graphics.stencil_instructions import *
from kivy.config import Config
import threading

#Window.fullscreen = 'auto'
#Config.set('graphics', 'fullscreen', 1)

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    -----------------
    arrays: list of array-like
        1-D arrays to form the cartesian product of.
    out: ndarray
        Array to place the cartesian product in.

    Returns
    -----------------
    out: ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype = dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:,n]]

    return out

"""
Copyright © 2016, N. Niehof, Radboud University Nijmegen

PsiMarginal is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PsiMarginal is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PsiMarginal. If not, see <http://www.gnu.org/licenses/>.
"""

def pf(parameters, psyfun='cGauss'):
    """Generate conditional probabilities from psychometric function.

    Arguments
    ---------
        parameters: ndarray (float64) containing parameters as columns
            mu   : threshold

            sigma    : slope

            gamma   : guessing rate (optional), default is 0.2

            lambda  : lapse rate (optional), default is 0.04

            x       : stimulus intensity

        psyfun  : type of psychometric function.
                'cGauss' cumulative Gaussian

                'Gumbel' Gumbel, aka log Weibull

    Returns
    -------
    1D-array of conditional probabilities p(response | mu,sigma,gamma,lambda,x)
    """

    # Unpack parameters
    if np.size(parameters, 1) == 5:
        [mu, sigma, gamma, llambda, x] = np.transpose(parameters)
    elif np.size(parameters, 1) == 4:
        [mu, sigma, llambda, x] = np.transpose(parameters)
        gamma = llambda
    elif np.size(parameters, 1) == 3:
        [mu, sigma, x] = np.transpose(parameters)
        gamma = 0.2
        llambda = 0.04
    else:  # insufficient number of parameters will give a flat line
        psyfun = None
        gamma = 0.2
        llambda = 0.04
    # Psychometric function
    ones = np.ones(np.shape(mu), dtype='f4')
    if psyfun == 'cGauss':
        # F(x; mu, sigma) = Normcdf(mu, sigma) = 1/2 * erfc(-sigma * (x-mu) /sqrt(2))
        z = np.divide(np.subtract(x, mu), sigma)
        p = 0.5 * np.array([math.erfc(-zi / np.sqrt(2)) for zi in z])
    elif psyfun == 'Gumbel':
        # F(x; mu, sigma) = 1 - exp(-10^(sigma(x-mu)))
        p = ones - np.exp(-np.power((np.multiply(ones, 10.0, dtype='f4')), (np.multiply(sigma, (np.subtract(x, mu)), dtype='f4'))))
    elif psyfun == 'Weibull':
        # F(x; mu, sigma)
        p = 1 - np.exp(-(np.divide(x, mu)) ** sigma)
    else:
        # flat line if no psychometric function is specified
        p = np.ones(np.shape(mu), dtype='f4')
    y = gamma + np.multiply((ones - gamma - llambda), p, dtype='f4')
    return y

class Psi:
    """Find the stimulus intensity with minimum expected entropy for each trial, to determine the psychometric function.

    Psi adaptive staircase procedure for use in psychophysics.

    Arguments
    ---------
        stimRange :
            range of possible stimulus intensities.

        Pfunction (str) : type of psychometric function to use.
            'cGauss' cumulative Gaussian

            'Gumbel' Gumbel, aka log Weibull

        nTrials :
            number of trials

        threshold :
            (alpha) range of possible threshold values to search

        thresholdPrior (tuple) : type of prior probability distribution to use.
            Also: slopePrior, guessPrior, lapsePrior.

            ('normal',0,1): normal distribution, mean and standard deviation.

            ('uniform',None) : uniform distribution, mean and standard deviation not defined.

        slope :
            (sigma) range of possible slope values to search

        slopePrior :
            see thresholdPrior

        guessRate :
            (gamma) range of possible guessing rate values to search

        guessPrior :
            see thresholdPrior

        lapseRate :
            (lambda) range of possible lapse rate values to search

        lapsePrior :
            see thresholdPrior

        marginalize (bool) :
            If True, marginalize out the lapse rate and guessing rate before finding the stimulus
            intensity of lowest expected entropy. This uses the Prins (2013) method to include the guessing and lapse rate
            into the probability disctribution. These rates are then marginalized out, and only the threshold and slope are included
            in selection of the stimulus intensity.

            If False, lapse rate and guess rate are included in the selection of stimulus intensity.

    How to use
    ----------
        Create a psi object instance with all relevant arguments. Selecting a correct search space for the threshold,
        slope, guessing rate and lapse rate is important for the psi procedure to function well. If an estimate for
        one of the parameters ends up at its (upper or lower) limit, the result is not reliable, and the procedure
        should be repeated with a larger search range for that parameter.

        Example:
            >>> s   = range(-5,5) # possible stimulus intensities
            obj = Psi(s) 
        The stimulus intensity to be used in the current trial can be found in the field xCurrent.

        Example:
            >>> stim = obj.xCurrent
        NOTE: if obj.xCurrent returns None, the calculation is not yet finished.
        This can be avoided by waiting until xCurrent has a numeric value, e.g.:
            >>> while obj.xCurrent == None:
                    pass # hang in this loop until the psi calculation has finished
                stim = obj.xCurrent

        After each trial, update the psi staircase with the subject response, by calling the addData method.

        Example:
            >>> obj.addData(resp)
    """

    def __init__(self, stimRange, Pfunction='cGauss', nTrials=50, threshold=None, thresholdPrior=('uniform', None),
                 slope=None, slopePrior=('uniform', None),
                 guessRate=None, guessPrior=('uniform', None), lapseRate=None, lapsePrior=('uniform', None),
                 marginalize=True, thread=True):

        # Psychometric function parameters
        self.stimRange = stimRange  # range of stimulus intensities
        self.version = 1.0
        self.threshold = np.arange(-10, 10, 0.1, dtype='f4')
        self.slope = np.arange(0.005, 20, 0.1, dtype='f4')
        self.guessRate = np.arange(0.0, 0.11, 0.05, dtype='f4')
        self.lapseRate = np.arange(0.0, 0.11, 0.05, dtype='f4')
        self.marginalize = marginalize  # marginalize out nuisance parameters gamma and lambda?
        self.psyfun = Pfunction
        self.thread = thread

        if threshold is not None:
            self.threshold = threshold
            if np.shape(self.threshold) == ():
                self.threshold = np.expand_dims(self.threshold, 0)
        if slope is not None:
            self.slope = slope
            if np.shape(self.slope) == ():
                self.slope = np.expand_dims(self.slope, 0)
        if guessRate is not None:
            self.guessRate = guessRate
            if np.shape(self.guessRate) == ():
                self.guessRate = np.expand_dims(self.guessRate, 0)
        if lapseRate is not None:
            self.lapseRate = lapseRate
            if np.shape(self.lapseRate) == ():
                self.lapseRate = np.expand_dims(self.lapseRate, 0)

        # Priors
        self.thresholdPrior = thresholdPrior
        self.slopePrior = slopePrior
        self.guessPrior = guessPrior
        self.lapsePrior = lapsePrior

        self.priorMu = self.__genprior(self.threshold, *thresholdPrior)
        self.priorSigma = self.__genprior(self.slope, *slopePrior)
        self.priorGamma = self.__genprior(self.guessRate, *guessPrior)
        self.priorLambda = self.__genprior(self.lapseRate, *lapsePrior)

        # if guess rate equals lapse rate, and they have equal priors,
        # then gamma can be left out, as the distributions will be the same
        self.gammaEQlambda = all((all(self.guessRate == self.lapseRate), all(self.priorGamma == self.priorLambda)))
        # likelihood: table of conditional probabilities p(response | alpha,sigma,gamma,lambda,x)
        # prior: prior probability over all parameters p_0(alpha,sigma,gamma,lambda)
        if self.gammaEQlambda:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.lapseRate), len(self.stimRange))
            self.likelihood = np.reshape(
                pf(cartesian((self.threshold, self.slope, self.lapseRate, self.stimRange)), psyfun=Pfunction), self.dimensions)
            # row-wise products of prior probabilities
            self.prior = np.reshape(
                np.prod(cartesian((self.priorMu, self.priorSigma, self.priorLambda)), axis=1), self.dimensions[:-1])
        else:
            self.dimensions = (len(self.threshold), len(self.slope), len(self.guessRate), len(self.lapseRate), len(self.stimRange))
            self.likelihood = np.reshape(
                pf(cartesian((self.threshold, self.slope, self.guessRate, self.lapseRate, self.stimRange)), psyfun=Pfunction), self.dimensions)
            # row-wise products of prior probabilities
            self.prior = np.reshape(
                np.prod(cartesian((self.priorMu, self.priorSigma, self.priorGamma, self.priorLambda)), axis=1), self.dimensions[:-1])

        # normalize prior
        self.prior = self.prior / np.sum(self.prior)

        # Set probability density function to prior
        self.pdf = np.copy(self.prior)

        # settings
        self.iTrial = 0
        self.nTrials = nTrials
        self.stop = 0
        self.response = []
        self.stim = []

        # Generate the first stimulus intensity
        self.minEntropyStim()

    def __genprior(self, x, distr='uniform', mu=0, sig=1):
        """Generate prior probability distribution for variable.

        Arguments
        ---------
            x   :  1D numpy array (float64)
                    points to evaluate the density at.

            distr :  string
                    Distribution to use a prior :
                        'uniform'   (default) discrete uniform distribution

                        'normal'   normal distribution

                        'gamma'    gamma distribution

                        'beta'     beta distribution

            mu :  scalar float
                first parameter of distr distribution (check scipy for parameterization)

            sig : scalar float
                second parameter of distr distribution

        Returns
        -------
        1D numpy array of prior probabilities (unnormalized)
        """
        if distr == 'uniform':
            nx = len(x)
            p = np.ones(nx, dtype='f4') / nx
        elif distr == 'normal':
            p = np.exp(-(x-mu)**2 / (2.0*(sig)**2)) / np.sqrt(2.0*np.pi*(sig)**2)
        elif distr == 'beta':
            OnePx = (sig - 1.0) * np.log1p(-x) + (mu - 1.0) * np.log(x)
            beta = math.gamma(mu) * math.gamma(sig) / math.gamma(mu + sig)
            OnePx -= np.log(np.abs(beta))
            p = np.exp(OnePx)
        elif distr == 'gamma':
            p = x ** (mu - 1) * (np.exp(-x)) / math.gamma(sig)
        else:
            nx = len(x)
            p = np.ones(nx, dtype='f4') / nx
        return p

    def meta_data(self):
        import time
        import sys
        metadata = {}
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
        metadata['date'] = date
        metadata['Version'] = self.version
        metadata['Python Version'] = sys.version
        metadata['Numpy Version'] = np.__version__
        metadata['Scipy Version '] = scipy.__version__
        metadata['psyFunction'] = self.psyfun
        metadata['thresholdGrid'] = self.threshold.tolist()
        metadata['thresholdPrior'] = self.thresholdPrior
        metadata['slopeGrid'] = self.slope.tolist()
        metadata['slopePrior'] = self.slopePrior
        metadata['gammaGrid'] = self.guessRate.tolist()
        metadata['gammaPrior'] = self.guessPrior
        metadata['lapseGrid'] = self.lapseRate.tolist()
        metadata['lapsePrior'] = self.lapsePrior
        return metadata

    def __entropy(self, pdf):
        """Calculate shannon entropy of posterior distribution.
        Arguments
        ---------
            pdf :   ndarray (float64)
                    posterior distribution of psychometric curve parameters for each stimuli


        Returns
        -------
        1D numpy array (float64) : Shannon entropy of posterior for each stimuli
        """
        # Marginalize out all nuisance parameters, i.e. all except alpha and sigma
        postDims = np.ndim(pdf)
        if self.marginalize == True:
            while postDims > 3:  # marginalize out second-to-last dimension, last dim is x
                pdf = np.sum(pdf, axis=-2)
                postDims -= 1
        # find expected entropy, suppress divide-by-zero and invalid value warnings
        # as this is handled by the NaN redefinition to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = np.multiply(pdf, np.log(pdf), dtype='f4')
        entropy[np.isnan(entropy)] = 0  # define 0*log(0) to equal 0
        dimSum = tuple(range(postDims - 1))  # dimensions to sum over. also a Chinese dish
        entropy = -(np.sum(entropy, axis=dimSum))
        return entropy

    def minEntropyStim(self):
        """Find the stimulus intensity based on the expected information gain.

        Minimum Shannon entropy is used as selection criterion for the stimulus intensity in the upcoming trial.
        """
        self.pdf = self.pdf
        self.nX = len(self.stimRange)
        self.nDims = np.ndim(self.pdf)

        # make pdf the same dims as conditional prob table likelihood
        self.pdfND = np.expand_dims(self.pdf, axis=self.nDims)  # append new axis
        self.pdfND = np.tile(self.pdfND, (self.nX))  # tile along new axis

        # Probabilities of response r (succes, failure) after presenting a stimulus
        # with stimulus intensity x at the next trial, multiplied with the prior (pdfND)
        self.pTplus1success = np.multiply(self.likelihood, self.pdfND, dtype='f4')
        self.pTplus1failure = self.pdfND - self.pTplus1success

        # Probability of success or failure given stimulus intensity x, p(r|x)
        self.sumAxes = tuple(range(self.nDims))  # sum over all axes except the stimulus intensity axis
        self.pSuccessGivenx = np.sum(self.pTplus1success, axis=self.sumAxes)
        self.pFailureGivenx = np.sum(self.pTplus1failure, axis=self.sumAxes)

        # Posterior probability of parameter values given stimulus intensity x and response r
        # p(alpha, sigma | x, r)
        self.posteriorTplus1success = self.pTplus1success / self.pSuccessGivenx
        self.posteriorTplus1failure = self.pTplus1failure / self.pFailureGivenx

        # Expected entropy for the next trial at intensity x, producing response r
        self.entropySuccess = self.__entropy(self.posteriorTplus1success)
        self.entropyFailure = self.__entropy(self.posteriorTplus1failure)
        self.expectEntropy = np.multiply(self.entropySuccess, self.pSuccessGivenx, dtype='f4') + np.multiply(self.entropyFailure, self.pFailureGivenx, dtype='f4')
        self.minEntropyInd = np.argmin(self.expectEntropy)  # index of smallest expected entropy
        self.xCurrent = self.stimRange[self.minEntropyInd]  # stim intensity at minimum expected entropy

        self.iTrial += 1
        if self.iTrial == (self.nTrials - 1):
            self.stop = 1

    def addData(self, response):
        """
        Add the most recent response to start calculating the next stimulus intensity

        Arguments
        ---------
            response: (int)
                1: correct/right

                0: incorrect/left
        """
        self.stim.append(self.xCurrent)
        self.response.append(response)

        self.xCurrent = None

        # Keep the posterior probability distribution that corresponds to the recorded response
        if response == 1:
            # select the posterior that corresponds to the stimulus intensity of lowest entropy
            self.pdf = self.posteriorTplus1success[Ellipsis, self.minEntropyInd]
        elif response == 0:
            self.pdf = self.posteriorTplus1failure[Ellipsis, self.minEntropyInd]

        # normalize the pdf
        self.pdf = self.pdf / np.sum(self.pdf)

        # Marginalized probabilities per parameter
        if self.gammaEQlambda:
            self.pThreshold = np.sum(self.pdf, axis=(1, 2))
            self.pSlope = np.sum(self.pdf, axis=(0, 2))
            self.pLapse = np.sum(self.pdf, axis=(0, 1))
            self.pGuess = self.pLapse
        else:
            self.pThreshold = np.sum(self.pdf, axis=(1, 2, 3))
            self.pSlope = np.sum(self.pdf, axis=(0, 2, 3))
            self.pLapse = np.sum(self.pdf, axis=(0, 1, 2))
            self.pGuess = np.sum(self.pdf, axis=(0, 1, 3))

        # Distribution means as expected values of parameters
        self.eThreshold = np.sum(np.multiply(self.threshold, self.pThreshold, dtype='f4'))
        self.eSlope = np.sum(np.multiply(self.slope, self.pSlope, dtype='f4'))
        self.eLapse = np.sum(np.multiply(self.lapseRate, self.pLapse, dtype='f4'))
        self.eGuess = np.sum(np.multiply(self.guessRate, self.pGuess, dtype='f4'))

        # Distribution std of parameters
        self.stdThreshold = np.sqrt(np.sum(np.multiply((self.threshold - self.eThreshold) ** 2, self.pThreshold, dtype='f4')))
        self.stdSlope = np.sqrt(np.sum(np.multiply((self.slope - self.eSlope) ** 2, self.pSlope, dtype='f4')))
        self.stdLapse = np.sqrt(np.sum(np.multiply((self.lapseRate - self.eLapse) ** 2, self.pLapse),dtype='f4'))
        self.stdGuess = np.sqrt(np.sum(np.multiply((self.guessRate - self.eGuess) ** 2, self.pGuess, dtype='f4')))

        # Start calculating the next minimum entropy stimulus

        if self.thread:
            threading.Thread(target=self.minEntropyStim).start()
        else:
            self.minEntropyStim()

# This works on ubuntu, not on Windows
timestamp = time.strftime("%Y%m%d_%H:%M:%S")

# If running on a Windows PC, run the following
# timestamp = time.strftime("%Y%m%d_%H_%M_%S")

# If running on an android device, set the right path to save the JSON file
if platform == 'android':
    from jnius import autoclass, cast, JavaException

    try:
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
    except JavaException:
        PythonActivity = autoclass('org.renpy.android.PythonActivity')

    Environment = autoclass('android.os.Environment')
    context = cast('android.content.Context', PythonActivity.mActivity)
    private_storage = context.getExternalFilesDir(Environment.getDataDirectory().getAbsolutePath()).getAbsolutePath()

    store = JsonStore(".".join([private_storage, timestamp, 'json']))

# Linux / Windows OS
else:
    store = JsonStore(".".join([timestamp, 'json']))

# Prepare dictionaries to save information
subj_info = {}
subj_anth = {}
subj_trial_info = {}

'''
These are Psi-Marginal Staircase related parameters
mu = threshold parameter
sigma = slope parameter
StimLevels = delta angle
'''

## testing purpose
## These parameter values would give the initial value of 30
ntrials = 25
mu = np.concatenate((np.arange(0.0, 15.2, 0.2), np.arange(15.25, 67.25, 0.25)))
mu = np.delete(mu, 25)
sigma = np.linspace(0.05, 1, 21)
lapse = np.arange(0, 0.1, 0.01)
guessRate = 0.5


stimLevels = np.concatenate((np.arange(0.0, 15.2, 0.2), np.arange(15.25, 67.25, 0.25)))
stimLevels = np.delete(stimLevels, 25)

################

#ntrials = 25
#mu = np.arange(0.1, 28.0, 0.1)
#mu = np.delete(mu, 49)
#sigma = np.linspace(0.05, 1, 21)
#lapse = np.arange(0.0, 0.1, 0.01)
#guessRate = 0.5
## 5.0 degrees deviation means the exact spot of the center of an index finger - skipped
#stimLevels = np.arange(0.1, 28.0, 0.1)
#stimLevels = np.delete(stimLevels, 49)
##slopePrior = ('gamma', 2, 20)

global psi_obj1, psi_obj2
# The first psi_obj.xCurrent = 20.0
psi_obj1 = Psi(stimLevels, Pfunction = 'Gumbel', nTrials = ntrials, threshold = mu, thresholdPrior = ('uniform', None), slope = sigma, slopePrior = ('uniform', None), guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = ('uniform', None), marginalize = True)

psi_obj2 = copy.copy(psi_obj1)

class CalibrationScreen(Screen):

    # Popup window
    def show_popup(self):

        the_popup = CalibPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
        the_popup.open()

class CalibPopup(Popup):
    pass

class ParamPopup(Popup):
    pass

class TrialPMc4Popup(Popup):
    pass

class GotoRNPopup(Popup):
    pass

class GotoASPopup(Popup):
    pass

class AreYouSurePopup(Popup):
    pass

class CustomDropDown(DropDown):
    pass

class ParamInputScreenOne(Screen):

    gender = ObjectProperty(None)
    staircase = ObjectProperty(None)
    handed_chk = ObjectProperty(False)
    dd_btn = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(ParamInputScreenOne, self).__init__(*args, **kwargs)

        # So we make an instance of the class: CustomDropDown
        self.drop_down = CustomDropDown()

        # These are the options
        notes = ["Psi-Marginal_7", "Psi-Marginal_10", "Psi-M(A)", "Psi-M(B)", "Randomized", "Staircase"]
        for note in notes:
            btn = Button(text = '%r' % note, size_hint_y=None, height=22)

            # each button has a callback that will call the select() method on the dropdown
            # pass the text of the button as the data of the selection
            btn.bind(on_release = lambda btn: self.drop_down.select(btn.text))

            # add the button inside the dropdown
            self.drop_down.add_widget(btn)

        self.drop_down.bind(on_release = self.drop_down.open)
        self.drop_down.bind(on_select = lambda instance, x:self.make_staircase_selection(x)) 

    def make_staircase_selection(self, x):

        psi_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
        setattr(self.dd_btn, 'text', x)
        self.staircase = x.strip("\'")
        if self.staircase != 'Staircase':
            self.parent.ids.paramsc.initd_text_input.disabled = True
        if self.staircase in ['Psi-Marginal_7', 'Psi-Marginal_10', 'Psi-M(A)', 'Psi-M(B)']:
            self.psi_nTrials = 25
            self.parent.ids.trialsc.psi_nTrials = self.psi_nTrials
            self.t = threading.Thread(target = self.initialize_psi, args = [25])
            self.t.start()
        psi_popup.argh.text = "%r selected" % self.staircase
        psi_popup.open()

    # Popup window to check if everything is saved properly
    def show_popup(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
        global subid
        global subj_info

        # Check if any of the parameter inputs is missing!  
        if hasattr(self, 't'):
            #print(self.t.isAlive())
            if self.t.is_alive():
                the_popup.argh.text = 'Threading in Progress'
                the_popup.open()
            else:
                if any([self.pid_text_input.text == "", self.age_text_input.text == "", self.gender == None, self.staircase == None, self.handed_chk == False]):
                    the_popup.argh.text = "Missing Values"
                    the_popup.open()
                else:
                    self.parent.ids.testsc_pm.delta_d = float(psi_obj1.xCurrent)
                    if self.staircase == 'Psi-M(B)':
                        self.parent.ids.testsc_pm.handedness.degree_dir = 1
                        self.parent.ids.trialsc.handedness.degree_dir = 1
                    else:
                        self.parent.ids.testsc_pm.handedness.degree_dir = -1
                        self.parent.ids.trialsc.handedness.degree_dir = -1
                    subid = "_".join(["SUBJ", self.pid_text_input.text])
                    subj_info = {'age' : self.age_text_input.text, 'gender' : self.gender, 'right_used' : self.ids.rightchk.active, 'Staircase used': self.staircase}
                    self.parent.current = "param_screen_two"

        else:
            if any([self.pid_text_input.text == "", self.age_text_input.text == "", self.gender == None, self.staircase == None, self.handed_chk == False]):
                the_popup.argh.text = "Missing Values"
                the_popup.open()
            else:
                subid = "_".join(["SUBJ", self.pid_text_input.text])
                subj_info = {'age' : self.age_text_input.text, 'gender' : self.gender, 'right_used' : self.ids.rightchk.active, 'Staircase used': self.staircase}
                self.parent.current = "param_screen_two"

    def initialize_psi(self, ntrial):
        global psi_obj1, psi_obj2
        psi_obj1 = Psi(stimLevels, Pfunction = 'Gumbel', nTrials = ntrial, threshold = mu, thresholdPrior = ('uniform', None), slope = sigma, slopePrior = ('uniform', None), guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = ('uniform', None), marginalize = True)
        psi_obj2 = copy.copy(psi_obj1)

    def Psimarginal_Yes(self, state):
        # A popup window to make sure that Psi-marginal is chosen
        psi_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        if state:
            self.psi_nTrials = 25
            self.t = threading.Thread(target = self.initialize_psi, args = [25])
            self.t.start()
            self.staircase = "Psi-Marginal"
            self.parent.ids.paramsc.initd_text_input.text = 'N/A'
            psi_popup.argh.text = "Psi-Marginal(Short) selected"
            psi_popup.open() 

    # Presenting randomized set stimuli to test if psi-marginal is restricting the individuals in their choices.
    def Randomized_Yes(self, state):

        psi_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        if state:
            self.staircase = "Randomized"
            self.parent.ids.paramsc.initd_text_input.text = 'N/A'
            psi_popup.argh.text = "Randomized Stimuli selected"
            psi_popup.open() 

    def Adaptive_Yes(self, state):

        psi_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        if state:
            self.staircase = "Adaptive-Staircase"       
            self.parent.ids.paramsc.initd_text_input.text = ''
            psi_popup.argh.text = "Adaptive Staircase selected"
            psi_popup.open()

    def if_active_m(self, state):
        if state:
            # Whill change the orientation of the testscreen's colorscreen
            self.gender = "M"

    def if_active_f(self, state):
        if state:
            self.gender = "F"

    def if_active_r(self, state):
        if state:
            # Will change the orientation of the testscreen's colorscreen
            self.parent.ids.testsc_pm.handedness.dir = 1
            self.parent.ids.trialsc.handedness.dir = 1
            self.parent.ids.testsc_as.handedness.dir = 1
            self.parent.ids.testsc_rn.handedness.dir = 1

            # Just for fool-proof
            self.handed_chk = True

    def if_active_l(self, state):
        if state:
            self.parent.ids.testsc_pm.handedness.dir = -1
            self.parent.ids.trialsc.handedness.dir = -1
            self.parent.ids.testsc_as.handedness.dir = -1
            self.parent.ids.testsc_rn.handedness.dir = -1

            # Just for fool-proof
            self.handed_chk = True 

class ParamInputScreenTwo(Screen):

    # Popup window to check if everything is entered
    def show_popup2(self):

        the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))

        # Check if any of the parameter inputs is missing!
        if any([self.flen_text_input.text == "", self.fwid_text_input.text == "", 
            (self.initd_text_input.disabled == False and self.initd_text_input.text == ""), self.mprad_text_input.text == ""]):
            the_popup.argh.text = "Missing Values"
            the_popup.open()
        else:
            global subj_anth
            subj_anth = {'flen' : self.flen_text_input.text, 'fwid' : self.fwid_text_input.text, 'init_step' : self.initd_text_input.text, 'MPJR' : self.mprad_text_input.text}

            if self.parent.ids.paramscone.staircase in ['Psi-Marginal_7', 'Psi-Marginal_10', 'Psi-M(A)', 'Psi-M(B)']:
                # Give the mp joint radius input to draw the test screen display
                # This shall also be done for the trial PM screen display
                self.parent.ids.testsc_pm.handedness.mprad = self.mprad_text_input.text
            elif self.parent.ids.paramscone.staircase == 'Adaptive-Staircase':
                self.parent.ids.testsc_as.handedness.mprad = self.mprad_text_input.text
            elif self.parent.ids.paramscone.staircase == 'Randomized':
                self.parent.ids.testsc_rn.handedness.mprad = self.mprad_text_input.text

            self.parent.ids.trialsc.handedness.mprad = self.mprad_text_input.text
            self.parent.current = "trial_screen"

class TestScreenAS(Screen):

    handedness = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(TestScreenAS, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        self.rgbindex = 0
        # counting the number of reverses to terminate the experiment
        self.rev_count = 0
        # checking if the reverse is happening
        self.prev_choice = list()
        # block number
        self.block_num = 0
        # check the trial number(within a block)
        self.trial_num = 0
        # Keep the record of total trials(regardless of block)
        self.trial_total = 0
        # JSON dictionary preparation
        self.subj_trial_info = {}
        # Trial Average
        self.stimuli = list()

        self.TrialScreen = TrialScreen()

    # changes the color of the buttons as well as the screen
    def change_col_setting(self):
        rgb_index = randrange(0, 3, 1)
        while rgb_index == self.rgbindex:
            rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before
        self.rgbindex = rgb_index

    # keep track of reversals
    def track_rev(self, response):
        if len(self.prev_choice) > 0:
            if self.prev_choice[-1] is not response:
                self.rev_count += 1
        self.prev_choice.append(response)

    def update_delta_d(self):
        self.delta_d = float(self.parent.ids.paramsc.initd_text_input.text) / (2.0 ** self.rev_count)

    def save_trial_data(self, rel_pos):

        ## Save the current degree
        degree_current = self.ids.cw.degree

        self.stimuli.append(np.abs(degree_current - 50.0))

        '''
        Check if the respons('on the left' or 'on the right') is correct.
        Get the current third x-coordinate, or the fourth point of the quadrilateral.
        Compare it with the true third x-coordinate of the quadrilateral
        If the current x-coordinate is greater than the true value, the correct answer should be "left"
        If the current x-coordinate is smaller than the true value, the correct answer should be "right"
        '''

        x_coord_current = self.ids.cw.quad_points[4]
        if x_coord_current > self.ids.cw.x_correct:
            correct_ans = "left"
        else:
            correct_ans = "right"

        # Compare if the response is correct
        self.right_or_wrong = int(rel_pos == correct_ans)

        # Based on the updated reversal count, calculate the delta_d
        self.update_delta_d()

        self.subj_trial_info["_".join(["TRIAL", str(self.trial_total)])] = {'trial_num': self.trial_num, 'block_num': self.block_num, 'rev_cnt': self.rev_count, 'Next_Step_size(deg)': self.delta_d, 'Visual Stimulus(deg)': 90.0 + degree_current, 'correct_ans': correct_ans, 'response': self.prev_choice[-1], 'response_correct': self.right_or_wrong}

    def where_is_your_finger(self, rel_pos):

        # Add the current choice, check if reversal is happening
        self.track_rev(rel_pos)
        # Save trial data
        self.save_trial_data(rel_pos) 

        # If you have reached the end...
        if self.trial_num == 19 or self.rev_count == 4:
            self.reset(self.block_num)
        else: 
            # next step deviation angle
            if rel_pos == 'left':
                temp = self.ids.cw.degree - self.delta_d

                # Set the left limit
                if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp)) < self.ids.cw.x):
                    self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.x - self.ids.cw.quad_points[6]) / self.ids.cw.height))
                else:
                    self.ids.cw.degree = temp

            elif rel_pos == 'right':
                temp = self.ids.cw.degree + self.delta_d

                # Set the right limit
                if (self.ids.cw.quad_points[6] + self.ids.cw.height*math.tan(math.radians(temp)) > self.ids.cw.right):
                    self.ids.cw.degree = math.degrees(math.atan((self.ids.cw.right - self.ids.cw.quad_points[6]) / self.ids.cw.height))
                else:
                    self.ids.cw.degree = temp

            self.trial_num += 1
            self.trial_total += 1

            ## change the colors of the screen
            self.change_col_setting()

    def reset(self, block_num):

        # Refresh the parameters
        self.rev_count = 0
        self.prev_choice = list()
        self.trial_num = 0

        if self.block_num == 0:
            # A new block begins
            self.block_num += 1
            # New display configuration
            self.ids.cw.degree = -1 * self.ids.cw.dir * 60
            # There's no turning back
            self.ids.layout.remove_widget(self.ids._backward)
            # The buttons would be disabled until an experimenter presses the 'resume' button
            self.ids._more_left.disabled = True
            self.ids._more_right.disabled = True
            # Don't forget to update the total trial number
            self.trial_total += 1

        else:
            # Dump everything to the store and move to the outcome screen
            store.put(subid, subj_info = subj_info, subj_anth = subj_anth, subj_trial_info = self.subj_trial_info)
            self.parent.ids.outsc.avg_performance = str(np.mean(np.array(self.stimuli)))
            self.block_num -= 1
            self.trial_total = 0
            self.subj_trial_info = {}
            # Bring back the "back" button
            self.ids.layout.add_widget(self.ids._backward)
            self.parent.current = "outcome_screen"


## This is a fool-proof trial session consisted of 4 trials.
## When the user is incorrect in any of the four trials,
## the reset button will be pressed.
## The user must pass four consecutive trials to proceed to the actual
## Psi-marginal trials
class TrialScreen(Screen):

    handedness = ObjectProperty(None)
    psi_nTrials = NumericProperty(None) 
    delta_d = NumericProperty()
    degree_dir = NumericProperty()

    def __init__(self, **kwargs):
        super(TrialScreen, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        self.rgbindex = 0
        # check the trial number(within a session)
        self.trial_num = 0
        self.delta_d = 13

    # changes the color of the buttons as well as the screen
    def change_col_setting(self, *largs):
        rgb_index = randrange(0, 3, 1)
        while rgb_index == self.rgbindex:
            rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before
        self.rgbindex = rgb_index

    # check if the response is correct or not
    def chk_if_corr(self, rel_pos): 

        x_coord_current = self.ids.cw.quad_points[4]
        if x_coord_current > self.ids.cw.x_correct:
            correct_ans = "left"
        else:
            correct_ans = "right"

        # Compare if the response is correct
        self.right_or_wrong = int(rel_pos == correct_ans)

        self.change_col_setting()

        if self.right_or_wrong == 1:
            self.trial_num += 1 
            # change the direction
            self.ids.cw.degree_dir = -1 * self.ids.cw.degree_dir
            # For third and fourth trials, the degree should be 25
            if self.trial_num == 2:
                self.ids.cw.degree = 10.0
            if self.trial_num == 4: 
                # If the staircase choice was 'Psi-Marginal', then proceed to the TestScreenPM
                if self.parent.ids.paramscone.staircase in ['Psi-Marginal_7', 'Psi-Marginal_10', 'Psi-M(A)', 'Psi-M(B)']:
                    the_popup = TrialPMc4Popup(title = "READ IT", size_hint = (None, None), size = (400, 400))
                    # Catch trials at trial_num = {12, 23, 34}
                    #if self.psi_nTrials == 25:
                    if self.parent.ids.testsc_pm.fucked_up_cnt > 0:
                        pass

                    else: 
                        if self.parent.ids.paramscone.staircase == 'Psi-Marginal_10':
                            list_a = [range(x, y) for x, y in zip([0, 21, 32,43],[10, 26, 37,48])]
                            catch = [20,31,42]
                        elif self.parent.ids.paramscone.staircase == 'Psi-Marginal_7':
                            list_a = [range(x, y) for x, y in zip([0, 13, 23, 34, 45], [7, 18, 28, 39, 48])]
                            catch = [12, 28, 44]

                        elif self.parent.ids.paramscone.staircase == 'Psi-M(A)':        #A 25 times first
                            list_a = [range(x, y) for x, y in zip([0, 11],[10,26])]
                            catch = [10, 31, 42] 
                        else:                                                           #B 25 times first
                            list_a = [range(x, y) for x, y in zip([26,37,43],[36,42,53])]
                            catch = [10, 36, 42]

                        psi_order = np.ones(int(self.psi_nTrials*2 + np.ceil(self.psi_nTrials/10)), dtype='f4')
                        psi_obj1_trials = [int(item) for sublist in list_a for item in sublist]
                        psi_order[psi_obj1_trials] = 0
                        psi_order[catch] = 2
                        self.parent.ids.testsc_pm.psi_order = psi_order
                        self.parent.ids.testsc_pm.catch = catch
                        self.parent.ids.testsc_pm.psi_nTrials = self.psi_nTrials 
                        # Tell when the B trials (Mid to Index) start
                        self.parent.ids.testsc_pm.fb_tr_n = int(np.where(psi_order==1)[0][0])

                # If the staircase choice was 'Randomized', then proceed to the TestScreenRN
                elif self.parent.ids.paramscone.staircase == 'Randomized':
                    the_popup = GotoRNPopup(title = 'READ IT', size_hint = (None, None), size = (400, 400)) 

                # If the staircase choice was 'Adaptive-Staircase', then proceed to the TestScreenAS
                elif self.parent.ids.paramscone.staircase == 'Adaptive-Staircase':
                    the_popup = GotoASPopup(title = 'READ IT', size_hint = (None, None), size = (400, 400))
                    self.parent.ids.testsc_pms.psi_order = [0]*25
                    self.parent.ids.testsc_pms.psi_nTrials = self.psi_nTrials
            
                # Reset the trial number so that if revisted, it could start normally again.
                self.trial_num = 0

                the_popup.argh.text = "Starting the actual test"
                the_popup.open()
            else:
                the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
                the_popup.argh.text = "You're correct"
                the_popup.open()
        else:
            the_popup = ParamPopup(title = "READ IT", size_hint = (None, None), size = (400, 400))
            the_popup.argh.text = "You're incorrect \n Try again."
            the_popup.open()
            # If you are wrong, everything resets
            self.trial_num = 0
            self.ids.cw.degree = 13.0
            self.ids.cw.degree_dir = -1 * self.ids.cw.degree_dir

## Actual Psi-marginal testing screen
class TestScreenPM(Screen):

    handedness = ObjectProperty(None)
    leftbutton = ObjectProperty()
    rightbutton = ObjectProperty()
    trialcount = ObjectProperty()
    psi_order = ListProperty()
    catch = ListProperty()
    psi_nTrials = NumericProperty()
    delta_d = NumericProperty()
    degree_dir = NumericProperty()
    fb_tr_n = NumericProperty()

    def __init__(self, **kwargs):
        super(TestScreenPM, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        self.rgbindex = 0
        # check the trial number(within a session)
        self.trial_num = 0
        global psi_obj1
        # delta_d will alternate between psi_obj and psi_obj2
        #self.delta_d = psi_obj1.xCurrent
        self.psi_stims = list()
        self.subj_trial_info = {}

        self.psi_type = ['A', 'B']
        self.first_few = {}
        # Just not calling defaultdict...
        self.first_few['A'] = list()
        self.first_few['B'] = list()
        # index of fucking up...
        self.fucked_up = False
        # fucked_up counts
        self.fucked_up_cnt = 0

    # changes the color of the buttons as well as the screen
    def change_col_setting(self, *largs):
        rgb_index = randrange(0, 3, 1)
        while rgb_index == self.rgbindex:
            rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before
        self.rgbindex = rgb_index

    ## n is either a integer or a list; msg is a string
    ## I assume it should be fine to not wait until the psi restart is completed,
    ## because you would proceed with another psi object that's running fine...
    def clearance(self, n, msg):
        if type(n) is list:
            for i in range(n[0], n[1]):
                del self.subj_trial_info["_".join(["TRIAL", str(i)])]
            if self.fucked_up_cnt == 0:
                global psi_obj2
                psi_obj2 = Psi(stimLevels, Pfunction = 'Gumbel', nTrials = self.psi_nTrials, threshold = mu, thresholdPrior = ('uniform', None), slope = sigma, slopePrior = ('uniform', None), guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = ('uniform', None), marginalize = True) 
                while psi_obj2.xCurrent is None:
                    pass 
        else: 
            for i in range(n):
                del self.subj_trial_info["_".join(["TRIAL", str(i)])]
            if self.fucked_up_cnt == 0:
                global psi_obj1
                psi_obj1 = Psi(stimLevels, Pfunction = 'Gumbel', nTrials = self.psi_nTrials, threshold = mu, thresholdPrior = ('uniform', None), slope = sigma, slopePrior = ('uniform', None), guessRate = guessRate, guessPrior = ('uniform', None), lapseRate = lapse, lapsePrior = ('uniform', None), marginalize = True)
            elif self.fucked_up_cnt == 0 and n == 11:
                while psi_obj1.xCurrent is None:
                    pass

        self.subj_trial_info["NOTE"] = msg

    def save_trial_data(self, rel_pos): 

        self.psi_stims.append(self.ids.cw.degree)

        x_coord_current = self.ids.cw.quad_points[4]
        if x_coord_current > self.ids.cw.x_correct:
            correct_ans = "left"
        else:
            correct_ans = "right"

        # Compare if the response is correct
        self.right_or_wrong = int(rel_pos == correct_ans)

        # saving the response for the current psi direction: 'A' or 'B'
        # Don't do this for the catch trials
        if int(self.psi_order[self.trial_num]) in [0,1]:
            self.first_few[self.psi_type[int(self.psi_order[self.trial_num])]].append(self.right_or_wrong) 


        if self.fucked_up_cnt == 1:
            ## If you have failed in the first 'A' block and fail again in the first 'B' block,
            ## erase all trial info and just stop it...
            ## You better quit if you have reached here... there's just no point continuing the task 
            if self.trial_num == 2 and any([x == 0 for x in self.first_few['B'][0:3]]):
                self.clearance(2, "The participant failed both in the first A and B sets so terminated")
                # YOU ARE FUCKED FOREVER
                self.fucked_up_cnt += 1
            ## If you have failed in the first 'A' block, passed the new 'B' block, 
            ## and again failed on the following 'A' block, you are done.
            elif self.trial_num == 5 and any([x == 0 for x in self.first_few['A'][0:3]]):
                self.clearance(5, "The participant kept failed on 'A' direction so terminated") 
                self.fucked_up_cnt += 1 
            ## If you passed the first 'A' block, failed on 'B', passed another 'A' block, and then
            ## fail on B again, you are done.
            elif (self.trial_num in [15, 17])  and any([x == 0 for x in self.first_few['B'][0:3]]):
                self.clearance(self.trial_num, "The participant kept failed on 'B' direction so terminated")
                self.fucked_up_cnt += 1

            else:
                ## If you see no issue, go save your trial info!
                if self.psi_order[self.trial_num] == 2:
                    self.subj_trial_info["_".join(["TRIAL", str(self.trial_num)])] = {'trial_num': self.trial_num, 'Visual_stimulus(deg)': self.ids.cw.false_ref + self.ids.cw.degree_dir*self.ids.cw.degree, 'correct_ans': correct_ans, 'response': rel_pos, 'response_correct': self.right_or_wrong}
                else:
                    self.subj_trial_info["_".join(["TRIAL", str(self.trial_num)])] = {'trial_num': self.trial_num, 'Psi_obj': self.psi_type[int(self.psi_order[self.trial_num])], 'Psi_stimulus(deg)': self.ids.cw.degree, 'Visual_stimulus(deg)': self.ids.cw.false_ref + self.ids.cw.degree_dir * self.ids.cw.degree, 'correct_ans': correct_ans, 'response': rel_pos, 'response_correct': self.right_or_wrong}

        elif self.fucked_up_cnt == 0:
            ## It is likely that if the user fails one of the first three trials, the user may never converge to the finger position.
            ## Therefore, if a user fails in any of the first 3 'A' trials,
            ## erase all those 3 trials and begin anew with 'B' trials
            if self.trial_num == 2 and any([x == 0 for x in self.first_few['A'][0:3]]):
                the_popup = AreYouSurePopup(title = "READ IT", size_hint = (None, None), size = (Window.width, 2*Window.height/3.0), pos_hint = {'x':0, 'y':0.4})
                the_popup.open()
                threading.Thread(target = self.clearance, args = (2, "The participant missed one of the first three A trials")).start()
                # Start again from B
                self.trial_num = 0
                self.fucked_up_cnt += 1
                if self.parent.ids.paramscone.staircase == 'Psi-Marginal_7':
                    list_a = [range(x, y) for x, y in zip([5, 18, 29, 39, 50], [12, 23, 34, 44, 53])]
                elif self.parent.ids.paramscone.staircase == 'Psi-Marginal_10':
                    list_a = [range(x, y) for x, y in zip([10, 26, 37, 48], [20, 31, 42, 53])]
                psi_obj1_trials = [int(item) for sublist in list_a for item in sublist]
                # Psi-order should be renewed
                psi_order = np.ones(int(self.psi_nTrials*2 + np.ceil(self.psi_nTrials/10)), dtype='f4')
                psi_order[psi_obj1_trials] = 0
                psi_order[self.catch] = 2
                self.psi_order = psi_order
                ## Fucked up status turns on
                self.fucked_up = True
                # Previous record removed
                self.first_few['A'] = []

            ## If you have passed the first 'A' trials, but make 3 consecutive fails in the first 3
            ## 'B' trials, then erase all those 5 'B' trials and continue with the 'A' trials
            elif self.trial_num == (self.fb_tr_n+2) and any([x == 0 for x in self.first_few['B'][0:3]]):
                the_popup = AreYouSurePopup(title = "READ IT", size_hint = (None, None), size = (Window.width, 2*Window.height/3.0), pos_hint = {'x':0, 'y':0.4})
                the_popup.open()
                threading.Thread(target = self.clearance, args = ([self.fb_tr_n,self.fb_tr_n+2], "The participant missed one of the first three B trials")).start()
                # Return to the point where the participant was still okay.
                self.trial_num = self.fb_tr_n
                self.fucked_up_cnt += 1
                if self.parent.ids.paramscone.staircase == 'Psi-Marginal_7':
                    list_a = [range(x, y) for x, y in zip([7, 18, 29, 45], [12, 23, 34, 48])]
                    psi_obj1_trials = [int(item) for sublist in list_a for item in sublist]
                    psi_order = np.concatenate((self.psi_order[0:7], np.ones(len(self.psi_order) - 7, dtype='f4')))
                elif self.parent.ids.paramscone.staircase == 'Psi-Marginal_10':
                    list_a = [range(x, y) for x, y in zip([10,26,43],[15,31,48])]
                    psi_obj1_trials = [int(item) for sublist in list_a for item in sublist]
                    psi_order = np.concatenate((self.psi_order[0:10], np.ones(len(self.psi_order) - 10, dtype='f4')))
                psi_order[psi_obj1_trials] = 0
                psi_order[self.catch] = 2
                self.psi_order = psi_order
                self.fucked_up = True
                self.first_few['B'] = []

            ## If you see no issue, go save your trial info!
            if self.psi_order[self.trial_num] == 2:
                self.subj_trial_info["_".join(["TRIAL", str(self.trial_num)])] = {'trial_num': self.trial_num, 'Visual_stimulus(deg)': self.ids.cw.false_ref + self.ids.cw.degree_dir*self.ids.cw.degree, 'correct_ans': correct_ans, 'response': rel_pos, 'response_correct': self.right_or_wrong}
            else: 
                self.subj_trial_info["_".join(["TRIAL", str(self.trial_num)])] = {'trial_num': self.trial_num, 'Psi_obj': self.psi_type[int(self.psi_order[self.trial_num])], 'Psi_stimulus(deg)': self.ids.cw.degree, 'Visual_stimulus(deg)': self.ids.cw.false_ref + self.ids.cw.degree_dir * self.ids.cw.degree, 'correct_ans': correct_ans, 'response': rel_pos, 'response_correct': self.right_or_wrong} 


    def reactivate_leftbutton(self, *largs):
        self.leftbutton.disabled = False
        self.rightbutton.disabled = False
        self.trialcount.text = str(self.trial_num + 1)

    def screen_blackout(self, *largs): 
        self.ids.cw.bg_color_before = (0, 0, 0, 1)
        self.ids.cw.bg_color_after = (0, 0, 0, 1)
        self.trialcount.text = ""

    def where_is_your_finger(self, rel_pos):

        # buttons are disabled for a short amount of time
        self.leftbutton.disabled = True
        self.rightbutton.disabled = True

        # Screen blacks out and the count is momentarily wiped out too.
        self.screen_blackout()

        self.save_trial_data(rel_pos)

        print('status?', self.trial_num)

        # If You are on the final trial, reset everything
        # Short version: self.psi_nTrials*2 + 3(catch trials)
        # Long version: self.psi_nTrials*2 + 5(catchtrials) 
        if self.trial_num == len(self.psi_order)-1 or self.fucked_up_cnt == 2:
            self.reset()
        elif self.fucked_up:
            if len(self.first_few['A']) == 0:
                self.delta_d = float(psi_obj2.xCurrent)
                self.ids.cw.false_ref = 45
                self.ids.cw.degree_dir = 1
                self.ids.cw.degree = float(self.delta_d)
            elif len(self.first_few['B']) == 0:
                self.delta_d = float(psi_obj1.xCurrent)
                self.ids.cw.false_ref = 55
                self.ids.cw.degree_dir = -1
                self.ids.cw.degree = float(self.delta_d)
            # Fucked up status checked
            self.fucked_up = not(self.fucked_up)
            # The buttons will be reactivated after 1.2s
            Clock.schedule_once(self.reactivate_leftbutton, 2)
        else:
            if self.psi_order[self.trial_num] == 2:
                if self.psi_order[self.trial_num + 1] == 1:
                    self.delta_d = float(psi_obj2.xCurrent)
                    self.ids.cw.false_ref = 45
                    self.ids.cw.degree_dir = 1
                elif self.psi_order[self.trial_num + 1] == 0:
                    self.delta_d = float(psi_obj1.xCurrent)
                    self.ids.cw.false_ref = 55
                    self.ids.cw.degree_dir = -1

            elif self.psi_order[self.trial_num] == 0:
                psi_obj1.addData(self.right_or_wrong)
                while psi_obj1.xCurrent is None: # Wait until calculation is complete
                    pass
                if self.psi_order[self.trial_num + 1] == 2:
                    self.delta_d = float(35)
                elif self.psi_order[self.trial_num + 1] == 1:
                    self.delta_d = float(psi_obj2.xCurrent)
                    self.ids.cw.false_ref = 45
                    self.ids.cw.degree_dir = 1
                else:
                    self.delta_d = float(psi_obj1.xCurrent)
            else:
                psi_obj2.addData(self.right_or_wrong)
                while psi_obj2.xCurrent is None:
                    pass
                if self.psi_order[self.trial_num + 1] == 2:
                    self.delta_d = float(35)
                elif self.psi_order[self.trial_num + 1] == 0:
                    self.delta_d = float(psi_obj1.xCurrent)
                    self.ids.cw.false_ref = 55
                    self.ids.cw.degree_dir = -1
                else:
                    self.delta_d = float(psi_obj2.xCurrent)

            self.ids.cw.degree = float(self.delta_d)
            self.trial_num += 1

            # The buttons will be reactivated after 1.2s
            Clock.schedule_once(self.reactivate_leftbutton, 2)

        print("nTrials: ", self.psi_nTrials, "first B trials:", self.first_few['B'], 'psi_order', self.psi_order)

        ## change the colors of the screen
        Clock.schedule_once(self.change_col_setting, 2)


    def reset(self):
        # Dump everything to the store
        store.put(subid, subj_info = subj_info, subj_anth = subj_anth, subj_trial_info = self.subj_trial_info)

        self.parent.ids.outsc.avg_performance = str(np.mean(np.array(self.psi_stims[-11:])) - 5.0)

        # Trial number renewed
        self.trial_num = 0

        self.leftbutton.disabled = False
        self.rightbutton.disabled = False

        self.first_few = {}
        # Just not calling defaultdict...
        self.first_few['A'] = list()
        self.first_few['B'] = list()

        self.fucked_up_cnt = 0

        # False reference moving to 55
        self.ids.cw.false_ref = 55
        # ... and the psi output will again be "subtracted"
        self.ids.cw.degree_dir = -1

        self.subj_trial_info = {}

        # Go to the outcome screen
        self.parent.current = "outcome_screen"

class TestScreenRN(Screen):

    handedness = ObjectProperty(None)
    leftbutton = ObjectProperty()
    rightbutton = ObjectProperty()
    trialcount = ObjectProperty()
    psi_order = ListProperty()
    psi_nTrials = NumericProperty()
    delta_d = NumericProperty()


    def __init__(self, **kwargs):
        super(TestScreenRN, self).__init__(**kwargs)
        self.rgblist1 = [(1, 0, 0, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
        self.rgblist2 = [(0, 0, 1, 1), (0.5, 0, 1, 1), (1, 0.56, 0.75, 1)]
        self.rgbindex = 0
        # check the trial number(within a session)
        self.trial_num = 0
        # delta_d will alternate between psi_obj and psi_obj2
        self.delta_d = 15.0
        self.psi_stims = list()
        self.subj_trial_info = {}

        self.psi_type = ['A', 'B']
        stimvals = [15.0, 13.0, 11.0, 9.5, 8.0, 6.5, 5.5, 4.5, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, .5]
        posneg = stimvals + [-x for x in stimvals]
        posnegdict = {x:(0 if x is not 15.0 else 1) for x in posneg}            # First trial's angle = 15.0
        self.stim_vals = np.array(stimvals)
        self.stim_count = posnegdict

    # changes the color of the buttons as well as the screen
    def change_col_setting(self, *largs):
        rgb_index = randrange(0, 3, 1)
        while rgb_index == self.rgbindex:
            rgb_index = randrange(0, 3, 1)
        self.ids.cw.bg_color_after = self.rgblist1[rgb_index]
        self.ids.cw.bg_color_before = self.rgblist2[rgb_index]
        self.ids._more_left.background_normal = ''
        self.ids._more_left.background_color = self.ids.cw.bg_color_after
        self.ids._more_right.background_normal = ''
        self.ids._more_right.background_color = self.ids.cw.bg_color_before
        self.rgbindex = rgb_index

    def save_trial_data(self, rel_pos): 

        self.psi_stims.append(self.ids.cw.degree)
        print(self.stim_count)

        # Count how many times each stimulus was presented.
        key = self.ids.cw.degree * self.ids.cw.degree_dir
        if key in self.stim_count.keys():
            self.stim_count[key] += 1
        else:
            self.stim_count[key] = 1

        x_coord_current = self.ids.cw.quad_points[4]
        if x_coord_current > self.ids.cw.x_correct:
            correct_ans = "left"
        else:
            correct_ans = "right"

        # Compare if the response is correct
        self.right_or_wrong = int(rel_pos == correct_ans) 

        psi_order = [0,0,0,0,0,1,1,1,1,1]
        self.psi_order = np.array(psi_order*9)

        self.subj_trial_info["_".join(["TRIAL", str(self.trial_num)])] = {'trial_num': self.trial_num, 'Psi_obj': self.psi_type[int(self.psi_order[self.trial_num])], 'Psi_stimulus(deg)': self.ids.cw.degree, 'Visual_stimulus(deg)': self.ids.cw.false_ref + self.ids.cw.degree_dir*self.ids.cw.degree, 'correct_ans': correct_ans, 'response': rel_pos, 'response_correct': self.right_or_wrong}

    def reactivate_leftbutton(self, *largs):
        self.leftbutton.disabled = False
        self.rightbutton.disabled = False
        self.trialcount.text = str(self.trial_num + 1)

    def screen_blackout(self, *largs): 
        self.ids.cw.bg_color_before = (0, 0, 0, 1)
        self.ids.cw.bg_color_after = (0, 0, 0, 1)
        self.trialcount.text = ""

    def is_temp_in(self, temp, shift = False, dirshift = None):
        if shift:
            if dirshift == 'thumb_to_middle':
                self.ids.cw.degree_dir = 1
            elif dirshift == 'middle_to_thumb':
                self.ids.cw.degree_dir = -1

        temp_counted = self.ids.cw.degree_dir * temp

        if temp_counted in self.stim_count.keys():
            while (self.stim_count[temp_counted] == 3):
                temp = self.stim_vals[np.random.randint(len(self.stim_vals))]
                temp_counted = self.ids.cw.degree_dir * temp

        self.delta_d = float(temp)

    def where_is_your_finger(self, rel_pos):

        # buttons are disabled for a short amount of time
        self.leftbutton.disabled = True
        self.rightbutton.disabled = True

        # Screen blacks out and the count is momentarily wiped out too.
        self.screen_blackout()

        self.save_trial_data(rel_pos)

        print('status?', self.trial_num)

        # If You are on the final trial, reset everything
        # The trial number should be 89 (in total 90 trials)
        if self.trial_num == len(self.psi_order)-1:
            self.reset()
        else:
            temp = self.stim_vals[np.random.randint(len(self.stim_vals))]
            if self.psi_order[self.trial_num] == 0:
                if self.psi_order[self.trial_num + 1] == 1:
                    self.is_temp_in(temp, shift = True, dirshift = 'thumb_to_middle')
                else:
                    self.is_temp_in(temp, shift = False)
            else:
                if self.psi_order[self.trial_num + 1] == 0:
                    self.is_temp_in(temp, shift = True, dirshift = 'middle_to_thumb')
                else:
                    self.is_temp_in(temp, shift = False)

            self.ids.cw.degree = float(self.delta_d)
            self.trial_num += 1

            # The buttons will be reactivated after 1.2s
            Clock.schedule_once(self.reactivate_leftbutton, 2)

        ## change the colors of the screen
        Clock.schedule_once(self.change_col_setting, 2)


    def reset(self):
        # Dump everything to the store
        store.put(subid, subj_info = subj_info, subj_anth = subj_anth, subj_trial_info = self.subj_trial_info)

        self.parent.ids.outsc.avg_performance = str(np.mean(np.array(self.psi_stims[-11:])) - 5.0)

        # Trial number renewed
        self.trial_num = 0

        self.leftbutton.disabled = False
        self.rightbutton.disabled = False

        # False reference moving to 55 - is it needed?
        #self.ids.cw.false_ref = 55
        # ... and the psi output will again be "subtracted"
        self.ids.cw.degree_dir = -1

        self.subj_trial_info = {}

        # Go to the outcome screen
        self.parent.current = "outcome_screen"

class OutcomeScreen(Screen):

    avg_performance = StringProperty('')

    def start_a_new_subject(self):

        # Prepare the trial count for the test screen(Psi-marginal)
        self.parent.ids.testsc_pm.trialcount.text ="1"
        # Reset the inputs of the first parameter input screen
        self.parent.ids.paramscone.pid_text_input.text = ''
        self.parent.ids.paramscone.age_text_input.text = ''
        self.parent.ids.paramscone.male_chkbox.active = False
        self.parent.ids.paramscone.female_chkbox.active = False
        self.parent.ids.paramscone.left_chkbox.active = False
        self.parent.ids.paramscone.right_chkbox.active = False
        #self.parent.ids.paramscone.psi_chkbox.active = False
        self.parent.ids.paramscone.dd_btn.text = 'Press here'
        #self.parent.ids.paramscone.psi_longer_chkbox.active = False 
        #self.parent.ids.paramscone.rn_chkbox.active = False
        #self.parent.ids.paramscone.adapst_chkbox.active = False
        # Reset the inputs of the second parameter input screen
        self.parent.ids.paramsc.flen_text_input.text = ''
        self.parent.ids.paramsc.fwid_text_input.text = ''
        self.parent.ids.paramsc.initd_text_input.text = ''
        self.parent.ids.paramsc.mprad_text_input.text = ''
        self.parent.current = "param_screen_one"


class screen_manager(ScreenManager):
    pass

class ProprioceptiveApp(App):

    def build(self):
        return screen_manager(transition=FadeTransition())

if __name__ == '__main__':
    Window.fullscreen = True
    Window.maximize()
    ProprioceptiveApp().run()

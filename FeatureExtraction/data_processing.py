# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:19:52 2022

@author: AD
"""

import numpy as np
import mne
import re
import scipy.io
from scipy.signal import butter, filtfilt, band_stop_obj
import os
import csv

from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, logspace, r_, sqrt, tan, log10,
                   arctan, arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, full, array,
                   mintypecode)
from numpy.polynomial.polynomial import polyval as npp_polyval

from scipy import special, optimize, fft as sp_fft
from scipy.special import comb, factorial

class data_processing(object):
    def __init__(self):
        self.path = ''
        
    def get_recording_idx(self, edf):
        inter = edf.split("_")[-1]
        inter = inter.split(".")[0]
        idx = int(inter[1:])
        
        return idx
        
        
    def find_recording(self, root, subject, variation):
        return
        
    def get_content(self, root):
        contents = os.listdir(root)
        files = sorted(contents.copy())
        # for index, file in enumerate(contents):
        #     if os.path.isdir(file):
        #         print(file)
        #         files.remove(file)
        return files
    
    def get_edf(self, root):
        contents = os.listdir(root)
        files = sorted(contents.copy())
        edf = [file for index, file in enumerate(files) if ('.edf' in file or '.EDF' in file)]
        edf.sort(key = self.get_recording_idx)
        
        return edf
    
    def get_tsv(self, edf):
        tsv_0 = edf[0:-4] + '_a1.tsv'
        tsv_1 = edf[0:-4] + '_a2.tsv'
        tsv_2 = edf[0:-4] + '_a3.tsv'
        return tsv_0, tsv_1, tsv_2
    
    def matbuttord(self, wp, ws, gpass, gstop, analog=False, fs=None):
        """Butterworth filter order selection.
    
        Return the order of the lowest order digital or analog Butterworth filter
        that loses no more than `gpass` dB in the passband and has at least
        `gstop` dB attenuation in the stopband.
    
        Parameters
        ----------
        wp, ws : float
            Passband and stopband edge frequencies.
    
            For digital filters, these are in the same units as `fs`.  By default,
            `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
            where 1 is the Nyquist frequency.  (`wp` and `ws` are thus in
            half-cycles / sample.)  For example:
    
                - Lowpass:   wp = 0.2,          ws = 0.3
                - Highpass:  wp = 0.3,          ws = 0.2
                - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
                - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]
    
            For analog filters, `wp` and `ws` are angular frequencies (e.g. rad/s).
        gpass : float
            The maximum loss in the passband (dB).
        gstop : float
            The minimum attenuation in the stopband (dB).
        analog : bool, optional
            When True, return an analog filter, otherwise a digital filter is
            returned.
        fs : float, optional
            The sampling frequency of the digital system.
    
            .. versionadded:: 1.2.0
    
        Returns
        -------
        ord : int
            The lowest order for a Butterworth filter which meets specs.
        wn : ndarray or float
            The Butterworth natural frequency (i.e. the "3dB frequency").  Should
            be used with `butter` to give filter results. If `fs` is specified,
            this is in the same units, and `fs` must also be passed to `butter`.
    
        See Also
        --------
        butter : Filter design using order and critical points
        cheb1ord : Find order and critical points from passband and stopband spec
        cheb2ord, ellipord
        iirfilter : General filter design using order and critical frequencies
        iirdesign : General filter design using passband and stopband spec
    
        Examples
        --------
        Design an analog bandpass filter with passband within 3 dB from 20 to
        50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
        Plot its frequency response, showing the passband and stopband
        constraints in gray.
    
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt
    
        >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
        >>> b, a = signal.butter(N, Wn, 'band', True)
        >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
        >>> plt.semilogx(w, 20 * np.log10(abs(h)))
        >>> plt.title('Butterworth bandpass filter fit to constraints')
        >>> plt.xlabel('Frequency [radians / second]')
        >>> plt.ylabel('Amplitude [dB]')
        >>> plt.grid(which='both', axis='both')
        >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
        >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
        >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
        >>> plt.axis([10, 100, -60, 3])
        >>> plt.show()
    
        """
        wp = atleast_1d(wp)
        ws = atleast_1d(ws)
        if fs is not None:
            if analog:
                raise ValueError("fs cannot be specified for an analog filter")
            wp = 2*wp/fs
            ws = 2*ws/fs
    
        filter_type = 2 * (len(wp) - 1)
        filter_type += 1
        if wp[0] >= ws[0]:
            filter_type += 1
    
        # Pre-warp frequencies for digital filter design
        if not analog:
            passb = tan(pi * wp / 2.0)
            stopb = tan(pi * ws / 2.0)
        else:
            passb = wp * 1.0
            stopb = ws * 1.0
    
        if filter_type == 1:            # low
            nat = stopb / passb
        elif filter_type == 2:          # high
            nat = passb / stopb
        elif filter_type == 3:          # stop
            wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                     args=(0, passb, stopb, gpass, gstop,
                                           'butter'),
                                     disp=0)
            passb[0] = wp0
            wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                     args=(1, passb, stopb, gpass, gstop,
                                           'butter'),
                                     disp=0)
            passb[1] = wp1
            nat = ((stopb * (passb[0] - passb[1])) /
                   (stopb ** 2 - passb[0] * passb[1]))
        elif filter_type == 4:          # pass
            nat = ((stopb ** 2 - passb[0] * passb[1]) /
                   (stopb * (passb[0] - passb[1])))
    
        nat = min(abs(nat))
    
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))
    
        # Find the Butterworth natural frequency WN (or the "3dB" frequency")
        # to give exactly gpass at passb.
        try:
            W0 = nat / ((GSTOP - 1.0) ** (1.0 / (2.0 * abs(ord))))
        except ZeroDivisionError:
            W0 = 1.0
            print("Warning, order is zero...check input parameters.")
    
        # now convert this frequency back from lowpass prototype
        # to the original analog filter
    
        if filter_type == 1:  # low
            WN = W0 * passb
        elif filter_type == 2:  # high
            WN = passb / W0
        elif filter_type == 3:  # stop
            WN = np.zeros(2, float)
            discr = sqrt((passb[1] - passb[0]) ** 2 +
                         4 * W0 ** 2 * passb[0] * passb[1])
            WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
            WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
            WN = np.sort(abs(WN))
        elif filter_type == 4:  # pass
            W0 = np.array([-W0, W0], float)
            WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
                  sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                       passb[0] * passb[1]))
            WN = np.sort(abs(WN))
        else:
            raise ValueError("Bad type: %s" % filter_type)
    
        if not analog:
            wn = (2.0 / pi) * arctan(WN)
        else:
            wn = WN
    
        if len(wn) == 1:
            wn = wn[0]
    
        if fs is not None:
            wn = wn*fs/2
    
        return ord, wn    
    
    def preprocessing_emg(self, data, hp, nf, r, fs):
        
        f0 = fs/2
        Wp = hp[0]/f0
        Ws = hp[1]/f0
        Rp = hp[2]
        Rs = hp[3]
        fildata = np.zeros(data.shape)
        highfildata = np.zeros(data.shape)
        n, Wn = self.matbuttord(Wp,Ws,Rp,Rs)
        b, a = butter(n, Wn, 'high')
        
        tetha = 2*np.pi*nf/fs
        b1 = [1,-2*np.cos(tetha),1]
        b1 = b1/np.sum(b1)
        a1 = [1,-2*r*np.cos(tetha),np.square(r)]
        a1 = a1/np.sum(a1)
        
        padlen = 3*(max(len(b),len(a))-1)
        for i in range(data.shape[0]):
            highfildata[i,:] = filtfilt(b, a, data[i,:], padtype='odd', padlen = padlen)
        
        
        padlen1 = 3*(max(len(b1),len(a1))-1)
        for j in range(data.shape[0]):
            fildata[j,:] = filtfilt(b1, a1, highfildata[j,:], padtype='odd', padlen = padlen1)
        
        return fildata

    def preprocessing_acc(self, data, hp, fs):
        
        data[abs(data) < 50] = 0

        f0 = fs/2
        Wp = np.array(hp[0])/f0
        Ws = np.array(hp[1])/f0
        order = 6 
        b, a = butter(order, [Wp, Ws], 'bandpass')
        fildata = np.zeros(data.shape)
        padlen = 3*(max(len(b),len(a))-1)
        for i in range(data.shape[0]):
            fildata[i,:] = filtfilt(b, a, data[i,:], padtype='odd', padlen = padlen)
        
        print(max(fildata[0,:]))
        
        return fildata

if __name__ == '__main__':
    data_processing = data_processing()
    a = data_processing.get_content('D:\Learning\SeizeIT2EMG\Dataformat')
    contents = os.listdir('D:\Learning\SeizeIT2EMG\Dataformat')
    data = np.array([range(1,200)])
    fildata = data_processing.preprocessing_emg(data, [20,10,0.1,40], 50, 0.95, 250)
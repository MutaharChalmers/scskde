#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from kdetools import gaussian_kde
from tqdm.auto import tqdm

class SCSKDE():
    """Sequential Conditional Sampling from Kernel Density Estimates (SCS-KDE).
    
    Fit time series models using non-parametric KDE methods and simulate
    synthetic realisations with optional exogenous forcing.
    """

    def __init__(self, ordern=1, orderx=0, bw_method='silverman',
                 bw_type='covariance', verbose=True):
        """Class constructor.

        Parameters
        ----------
            ordern : int, optional
                Order of model, i.e. longest time lag, for endogenous features.
                Defaults to 1.
            orderx : int, optional
                Order of model, i.e. longest time lag, for exogenous features.
                Should be less than or equal to ordern for current version.
                Defaults to 0.
            bw_method : str, optional
                KDE bandwidth selection method. Options are the same as for 
                `kdetools.gaussian_kde.set_bandwidth`. Defaults to 'silverman'.
            bw_type : str, optional
                Type of bandwidth matrix used. Options are the same as for 
                `kdetools.gaussian_kde.set_bandwidth`. Defaults to 'covariance'.
            verbose : bool, optional
                Show tqdm toolbar during fitting and simulation, or not.
                Defaults to True.
        """

        if orderx > ordern:
            print(f'orderx ({orderx}) should be <= ordern ({ordern})')
            return None
        if ordern < 1:
            print(f'ordern ({ordern}) should be >= 1')
            return None

        self.ordern = ordern
        self.orderx = orderx
        self.bw_method = bw_method
        self.bw_type = bw_type
        self.verbose = verbose

        self.order = max(ordern, orderx)
        self.Xs = {}
        self.models = {}

    def fit(self, X_endog, dep_endog, X_exog=None, dep_exog=None,
            periods=None):
        """Fit model.

        Parameters
        ----------
            X_endog : ndarray
                Training data for endogenous features only.
            dep_endog : dict
                Dependency graph of endogenous features on other endogenous
                features. Structure is as follows: 
                    {(m1, n1): [v1, v2, ..., vj],
                     (m1, n2): [v1, v2, ..., vj],
                     ...,
                     (mi, nj): [v1, v2, ..., vj]}
                for `i` periods and `j` endogenous features.
                Keys of the dictionary must cover all combinations of periods
                and endogenous features - all features being modelled must
                depend on something.
            X_exog : ndarray, optional
                Exogenous forcing. Currently only support a single realisation
                of `X_exog`. Defaults to None.
            dep_exog : dict, optional
                Dependency graph of endogenous features on exogenous
                features. Structure is as follows: 
                    {(m1, n1): [w1, w2, ..., wk],
                     (m1, n2): [w1, w2, ..., wk],
                     ...,
                     (mi, nj): [w1, w2, ..., wk]}
                for `i` periods, `j` endogenous and `k` exogenous features.
                Unlike `dep_endog`, the keys of `dep_exog` do not need to
                cover all combinations of periods and endogenous features.
                Defaults to None.
            periods : ndarray, optional
                PeriodID for each time step, allowing different models to be
                fit to subsets of the data. Must be the same length as
                as X_endog.shape[0]. If None (default), all data is used to
                fit a single model.
        """

        # Input validation
        if periods is None:
            periods = np.zeros(X_endog.shape[0])
            self.uperiods = {0}
        elif periods.shape[0] == X_endog.shape[0]:
            periods = np.array(periods)
            self.uperiods = set(periods)
        else:
            print('`periods` must be the same length as `X_endog.shape[0]`')
            return None

        if dep_endog is None:
            print('Dependency dictionary `dep_endog` must be specified')
            return None

        if X_exog is not None:
            if X_endog.shape[0] != X_exog.shape[0]:
                print('`X_exog` must have the same number of rows as `X_endog`')
                return None
            if dep_exog is None:
                print('Dependency dictionary `dep_exog` must be specified')
                return None
            mx, nx = zip(*[(m, n) for m, n in dep_exog.keys()])
            if set(mx) != self.uperiods:
                print(f'Periods `m` in `dep_exog` must match {self.uperiods}')
                return None
            if not set(nx).issubset(set(range(X_exog.shape[1]))):
                print('Variables `n` in `dep_exog` must be a subset of '
                      f'{range(X_exog.shape[1])}')
                return None

            self.dx = dep_exog
            self.Nx = X_exog.shape[1]
            Xx = np.array(X_exog)

        mn, nn = zip(*[(m, n) for m, n in dep_endog.keys()])
        if set(mn) != set(periods):
            print(f'Periods `m` in `dep_endog` must match {set(periods)}')
            return None
        if set(nn) != set(range(X_endog.shape[1])):
            print(f'Variables `n` in `dep_endog` must match {range(X_endog.shape[1])}')
            return None

        self.dn = dep_endog
        self.Nn = X_endog.shape[1]
        Xn = np.array(X_endog)
        #M = Xn.shape[0]

        # Loop over periods
        pbar = tqdm(total=len(self.uperiods)*self.Nn, disable=not self.verbose)
        for m in self.uperiods:
            # Loop over endogenous variables to be modelled
            for n in range(self.Nn):
                # Endogenous variables
                lags = range(self.ordern)
                XN = ([np.roll(Xn, -lag, axis=0)[:,self.dn[m,n]] for lag in lags] +
                      [np.roll(Xn[:,[n]], -self.ordern, axis=0)])

                # Exogenous variables
                if X_exog is None:
                    XX = []
                else:
                    lags = range(self.ordern-self.orderx, self.ordern+1)
                    XX = [np.roll(Xx, -lag, axis=0)[:,self.dx[m,n]]
                          for lag in lags if self.dx.get((m,n), None) is not None]

                # Select relevant records only
                mask = np.roll(periods, -self.order)[:-self.order] == m
                self.Xs[m,n] = np.hstack(XX + XN)[:-self.order][mask]

                # Fit KDEs
                if self.bw_method == 'silverman':
                    self.models[m,n] = gaussian_kde(self.Xs[m,n].T)
                    bw = self.models[m,n].silverman_factor_ref().mean()
                    self.models[m,n].set_bandwidth(bw_method=bw)
                else:
                    self.models[m,n] = gaussian_kde(self.Xs[m,n].T)
                    self.models[m,n].set_bandwidth(bw_method=self.bw_method,
                                                   bw_type=self.bw_type)
                pbar.update(1)
        pbar.close()

    def simulate(self, Nt, X0, X_exog=None, batches=1, periods=None, seed=42):
        """Simulate from fitted model.

        Parameters
        ----------
            Nt : int
                Number of time steps to simulate.
            X0 : ndarray
                Inital values to be used in the simulation. If using different
                initial values for each batch, X0 must be 3D with shape
                (# batches, model order, # endogenous features). If using the
                same initial values for each batch, X0 must be 2D with shape
                (model order, # endogenous features).
            X_exog : ndarray, optional
                Exogenous forcing to be used in the simulation. If using
                different forcings for each batch, X_exog must be 3D with shape
                (# batches, time steps, # exogenous features). If using the
                same forcing for each batch, X_exog must be 2D with shape
                (model order, # exogenous features).
            batches : int, optional
                Number of batches, or ensemble members, to simulate.
            periods : ndarray, optional
                PeriodID for each time step, allowing different models to be
                used for subsets of the data. Must be length Nt. If None (default)
                all time steps modelled identically.
            seed : {int, `np.random.Generator`, `np.random.RandomState`}, optional
                Seed or random number generator state variable.

        Returns
        -------
            Y : ndarray
                Simulated data.
        """

        # Input validation
        if X0.shape != (batches, self.order, self.Nn):
            print(f'`X0.shape` {X0.shape} must be consistent with'
                  ' (# batches, model order, # of endogenous features)'
                  f' ({batches}, {self.order}, {self.Nn})')
            return None
        if X_exog is not None:
            if X_exog.shape != (batches, Nt, self.Nx):
                print(f'`X_exog.shape` {X_exog.shape} must be consistent with'
                      ' (# batches, time steps, # of exogenous features)'
                      f' ({batches}, {Nt}, {self.Nx})')
                return None
        if periods is None:
            periods = np.zeros(Nt, dtype=int)
        elif periods.shape[0] == Nt:
            if set(periods).issubset(self.uperiods):
                periods = np.array(periods)
            else:
                print(f'`periods` has periodIDs not in {self.uperiods}')
                return None
        else:
            print(f'`periods` must be length Nt={Nt} or `None`')
            return None

        # Initialise random number generator
        prng = np.random.RandomState(seed)

        # Initialise output array
        Y = np.zeros(shape=(batches, Nt, self.Nn))
        Y[:,:self.order,:] = X0

        # Loop over time steps
        for i in tqdm(range(self.order, Nt), disable=not self.verbose):
            m = periods[i]
            # Loop over variables
            for n in range(self.Nn):
                # Define conditioning vector
                if X_exog is None:
                    x_cond = np.hstack([Y[:,i-lag,self.dn[m,n]]
                                        for lag in range(self.ordern, 0, -1)])
                else:
                    x_cond_x = [X_exog[:,i-lag,self.dx[m,n]]
                                for lag in range(self.orderx, -1, -1)
                                if self.dx.get((m,n), None) is not None]
                    x_cond_n = [Y[:,i-lag,self.dn[m,n]]
                                for lag in range(self.ordern, 0, -1)]
                    x_cond = np.hstack(x_cond_x + x_cond_n)

                # Across all batches, sample 1 realisation for each dimension
                Y[:,i,n] = self.models[m,n].conditional_resample(1,
                                                                 x_cond=x_cond,
                                                                 dims_cond=range(x_cond.shape[1]),
                                                                 seed=prng)[:,0,0]
        return Y

    def whiten(self, X):
        """ZCA/Mahalanobis whitening.

        Simulated stochastic principal components with a complex dependency
        structure can end up being non-orthogonal. When recombining stochastic
        PCs with their EOFs, the PCs must be orthogonalised. According to
        Kessey et al (2016) "Optimal Whitening and Decorrelation", the optimal
        whitening transformation to minimise the changes from the original data
        is the ZCA/Mahalanobis transformation with the whitening matrix being
        the inverse-square root of the covariance matrix.

        Parameters
        ----------
            X : ndarray
                Array to be whitened of shape (m, n) where m denotes records
                and n features.

        Returns
        -------
            Xw : ndarray
                Whitened version of input array.
        """

        S = np.cov(X.T)
        u, v = np.linalg.eigh(S)
        S_root = v * np.sqrt(np.clip(u, np.spacing(1), np.inf)) @ v.T
        W = np.linalg.inv(S_root)
        return (X @ W.T) * X.std(axis=0)

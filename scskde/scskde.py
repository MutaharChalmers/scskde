#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import kdetools as kt
import json
from tqdm.auto import tqdm

class SCSKDE():
    """Sequential Conditional Sampling from Kernel Density Estimates (SCS-KDE).
    
    Fit time series models using non-parametric KDE methods and simulate
    synthetic realisations with optional exogenous forcing.
    """

    def __init__(self, ordern=1, orderx=None, bw_method='silverman',
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
                Defaults to None.
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

        if orderx is not None and orderx > ordern:
            print(f'orderx ({orderx}) should be <= ordern ({ordern})')
            return None
        if ordern < 1:
            print(f'ordern ({ordern}) should be >= 1')
            return None
        bw_methods = ['silverman', 'scott', 'silverman_ref', 'cv']
        if isinstance(bw_method, str) and bw_method not in bw_methods:
            print(f'bw_method ({bw_method}) should be one of {bw_methods}')
            return None
        bw_types = ['covariance', 'diagonal', 'equal']
        if isinstance(bw_type, str) and bw_type not in bw_types:
            print(f'bw_type ({bw_type}) should be one of {bw_types}')
            return None

        self.ordern = ordern
        self.orderx = orderx
        self.bw_method = bw_method
        self.bw_type = bw_type
        self.verbose = verbose
        self.models = {}

    def fit(self, Xn, depn, Xx=None, depx=None, periods=None):
        """Fit model.

        Parameters
        ----------
            Xn : ndarray
                Training data for endogenous features only. (m, n) 2D array
                of `m` samples and `n` features.
            depn : dict
                Dependency graph of endogenous features on other endogenous
                features. Structure is as follows:
                    {(m1, n1): [n1, n2, ..., nj],
                     (m1, n2): [n1, n2, ..., nj],
                     ...,
                     (mi, nj): [n1, n2, ..., nj]}
                for periods `mi`  and endogenous features `nj`.
                Keys must cover all combinations of periods and endogenous
                features - modelled features must depend on something.
            Xx : ndarray, optional
                Exogenous forcing. Defaults to None.
            depx : dict, optional
                Dependency graph of endogenous features on exogenous
                features. Structure is as follows: 
                    {(m1, n1): [x1, x2, ..., xk],
                     (m1, n2): [x1, x2, ..., xk],
                     ...,
                     (mi, nj): [x1, x2, ..., xk]}
                for periods `mi`, endogenous `nj` and exogenous `xk`.
                The keys of `depx` need not cover all combinations of periods
                and endogenous features. Defaults to None.
            periods : ndarray, optional
                PeriodID for each time step, so different models can be fit to
                subsets of the data. If None (default), all data used to fit a
                single model, otherwise must be the same length as Xn.shape[0].
        """

        # Input validation
        if periods is None:
            periods = np.zeros(Xn.shape[0])
            self.uperiods = {0}
        elif periods.shape[0] == Xn.shape[0]:
            periods = np.array(periods)
            self.uperiods = set([int(p) for p in periods])
        else:
            print('`periods` must be the same length as `Xn.shape[0]`')
            return None

        if depn is None:
            print('Dependency dictionary `depn` must be specified')
            return None

        if Xx is not None:
            if self.orderx is None:
                print('`self.orderx` is None - cannot pass an exogenous forcing `Xx`')
                return None
            if Xx.shape[0] != Xn.shape[0]:
                print('`Xx` must have the same number of rows as `Xn`')
                return None
            if depx is None:
                print('Dependency dictionary `depx` must be specified')
                return None
            mx, nx = zip(*[(m, n) for m, n in depx.keys()])
            if set(mx) != self.uperiods:
                print(f'Periods `m` in `depx` must match {self.uperiods}')
                return None
            if not set(nx).issubset(set(range(Xx.shape[1]))):
                print('Variables `n` in `depx` must be a subset of '
                      f'{range(Xx.shape[1])}')
                return None

            self.dx = depx
            self.Nx = Xx.shape[1]
            Xx = np.array(Xx)
        else:
            self.dx = None
            self.Nx = None

        mn, nn = zip(*[(m, n) for m, n in depn.keys()])
        if set(mn) != self.uperiods:
            print(f'Periods `m` in `depn` must match {self.uperiods}')
            return None
        if set(nn) != set(range(Xn.shape[1])):
            print(f'Variables `n` in `depn` must match {range(Xn.shape[1])}')
            return None

        self.dn = depn
        self.Nn = Xn.shape[1]
        Xn = np.array(Xn)

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
                if Xx is None:
                    XX = []
                else:
                    lags = range(self.ordern-self.orderx, self.ordern+1)
                    XX = [np.roll(Xx, -lag, axis=0)[:,self.dx[m,n]]
                          for lag in lags if self.dx.get((m,n), None) is not None]

                # Select relevant records only
                mask = np.roll(periods, -self.ordern)[:-self.ordern] == m
                X = np.hstack(XX + XN)[:-self.ordern][mask]

                # Fit KDEs
                if self.bw_method == 'silverman_ref':
                    self.models[m,n] = kt.gaussian_kde(X.T)
                    bw = self.models[m,n].silverman_factor_ref().mean()
                    self.models[m,n].set_bandwidth(bw_method=bw,
                                                   bw_type='covariance')
                    self.models[m,n].bw_method = 'constant'
                else:
                    self.models[m,n] = kt.gaussian_kde(X.T)
                    self.models[m,n].set_bandwidth(bw_method=self.bw_method,
                                                   bw_type=self.bw_type)
                pbar.update(1)
        pbar.close()

    def simulate(self, Nt, X0, Xx=None, batches=1, periods=None, seed=42):
        """Simulate from fitted model.

        Parameters
        ----------
            Nt : int
                Number of time steps to simulate.
            X0 : ndarray
                Inital values to be used in the simulation. If using different
                initial values for each batch, `X0` must be 3D with shape
                (# batches, model order, # endogenous features). If using the
                same initial values for each batch, `X0` must be 2D with shape
                (model order, # endogenous features).
            Xx : ndarray, optional
                Exogenous forcing to be used in the simulation. If using
                different forcings for each batch, `Xx` must be 3D with shape
                (# batches, time steps, # exogenous features). If using the
                same forcing for each batch, `Xx` must be 2D with shape
                (model order, # exogenous features).
            batches : int, optional
                Number of batches, or ensemble members, to simulate.
            periods : ndarray, optional
                PeriodID for each time step, allowing different models to be
                used for subsets of the data. Must be length `Nt`.
                If None (default) all time steps modelled identically.
            seed : {int, `np.random.Generator`, `np.random.RandomState`}, optional
                Seed or random number generator state variable.

        Returns
        -------
            Y : ndarray
                Simulated data.
        """

        # Input validation
        if X0.shape != (batches, self.ordern, self.Nn):
            print(f'`X0.shape` {X0.shape} must be consistent with'
                  ' (# batches, model order, # of endogenous features)'
                  f' ({batches}, {self.ordern}, {self.Nn})')
            return None
        else:
            X0 = np.array(X0)
        if Xx is not None:
            if Xx.shape != (batches, Nt, self.Nx):
                print(f'`Xx.shape` {Xx.shape} must be consistent with'
                      ' (# batches, time steps, # of exogenous features)'
                      f' ({batches}, {Nt}, {self.Nx})')
                return None
            else:
                Xx = np.array(Xx)
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
        Y[:,:self.ordern,:] = X0

        # Loop over time steps
        for i in tqdm(range(self.ordern, Nt), disable=not self.verbose):
            m = periods[i]
            # Loop over variables
            for n in range(self.Nn):
                # Define conditioning vector
                if Xx is None:
                    x_cond = np.hstack([Y[:,i-lag,self.dn[m,n]]
                                        for lag in range(self.ordern, 0, -1)])
                else:
                    x_cond_x = [Xx[:,i-lag,self.dx[m,n]]
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

    def save(self, outpath, model_name, overwrite=False):
        """Save model to disk.

        Parameters
        ----------
        outpath : str
            Outpath.
        model_name : str
            Model name.
        overwrite : bool
            Overwrite data if it exists. Default False.
        """

        # Create directory
        outpath = os.path.join(outpath, model_name)
        try:
            os.makedirs(outpath)
        except:
            if overwrite:
                print('Model file exists; overwriting...')
            else:
                print('Model file exists; aborting...')
                return None

        # Define metadata to recreate the SCS-KDE model and write to file
        meta = {'name': model_name,
                'ordern': self.ordern,
                'orderx': self.orderx,
                'bw_method': self.bw_method,
                'bw_type': self.bw_type,
                'verbose': self.verbose,
                'uperiods': list(self.uperiods),
                'Nn': self.Nn,
                'Nx': self.Nx}

        with open(os.path.join(outpath, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        # Loop over models and write each one to disk
        for k, v in self.models.items():
            v.save(outpath, '_'.join(map(str, k)), overwrite=True, verbose=False)

        # Save dependency graphs after converting to JSONable formats
        dn_str = {'_'.join(map(str, k)): v.tolist() for k, v in self.dn.items()}
        with open(os.path.join(outpath, 'depn.json'), 'w') as f:
            json.dump(dn_str, f)
        if self.orderx is not None:
            dx_str = {'_'.join(map(str, k)): v.tolist() for k, v in self.dx.items()}
            with open(os.path.join(outpath, 'depx.json'), 'w') as f:
                json.dump(dx_str, f)

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

def load(inpath, model_name):
    """Load model from disk.

    Parameters
    ----------
    inpath : str
        Inpath.
    model_name : str
        Model name.

    Returns
    -------
    scskde : scskde.SCSKDE
        Instance of SCS-KDE model.
    """

    # Load metadata
    inpath = os.path.join(inpath, model_name)
    with open(os.path.join(inpath, 'meta.json'), 'r') as f:
        meta = json.load(f)

    # Create object and define fitted characteristics
    scskde = SCSKDE(ordern=meta['ordern'], orderx=meta['orderx'],
                    bw_method=meta['bw_method'], bw_type=meta['bw_type'],
                    verbose=meta['verbose'])
    scskde.uperiods = set(meta['uperiods'])
    scskde.Nn = meta['Nn']
    scskde.Nx = meta['Nx']

    # Load dependency matrices
    with open(os.path.join(inpath, 'depn.json'), 'r') as f:
        depn = json.load(f)
    scskde.dn = {tuple(map(int, k.split('_'))): np.array(v)
                 for k, v in depn.items()}
    if meta['Nx'] is not None:
        with open(os.path.join(inpath, 'depx.json'), 'r') as f:
            depx = json.load(f)
        scskde.dx = {tuple(map(int, k.split('_'))): np.array(v)
                     for k, v in depx.items()}

    # Load models
    for model in os.listdir(inpath):
        if os.path.isdir(os.path.join(inpath, model)):
            m, n = tuple(map(int, model.split('_')))
            scskde.models[m, n] = kt.load(inpath, model)

    return scskde

import sys
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
from scipy import stats
from multiprocessing import Pool, cpu_count
import time

# ###############################################################################
#
# Copyright (C) 2018-2020, Harry Desmond & Federico Lelli
# E-mail: harry.desmond@physics.ox.ac.uk, LelliF@cardiff.ac.uk
#
# Up-to-date versions of the software are available from:
# https://www2.physics.ox.ac.uk/contacts/people/desmond
# http://astroweb.cwru.edu/SPARC/
# https://www.lellifederico.com/
#
# If you have used this software in your research, please acknowledge
# Lelli et al. (2019, MNRAS, 484, 2367) where we describe in detail
# the maximum-likelihood (ML) method to fit a linear model y=a*x+b
# considering either vertical or the orthogonal intrinsic scatter.
#
# This software is provided as it is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#################################################################################
#+
# NAME:
#       BayesLineFit
#
# PURPOSE:
#       MCMC-based linear fit to data with errors in both coordinates
#       and non-negligible intrinsic scatter. See Appendix A here:
#       https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.3267L/abstract
#
# EXPLANATION:
#       Fit a linear model (y= a*x + b) to a set of points (xi, yi) with
#       errors (err_xi, err_yi) and non-negligible intrinsic scatter (s).
#       The errors are assumed to be Gaussian-distributed and independent.
#       The intrinsic scatter is assumed to be Gaussian either in the
#       vertical direction y or the direction orthogonal to the best-fit line.
#       The program uses an affine-invariant MCMC sampler to map out the full
#       posteriors of slope, intercept and intrinsic scatter, plots the results
#       and prints summary statistics.
#
# DEPENDENCIES:
#       Python packages numpy, emcee, corner, matplotlib, scipy, multiprocessing.
#       Tested on Python 3.7.5.
#
# OUTPUTS:
#       Several diagnostics of the data and linear fit are printed to screen.
#       Asymmetric errors are estimated at the 68% confidence level of the
#       marginalized 1-D posterior distributions of the fitting parameters.
#       The following files are produced:
#       outfile_chain: data from MCMC chains
#       outfile_bestfit: best-fit values and summary statistics
#       outplot_convergence: walker convergence plot
#       outplot_corner: corner plot of parameter constraints
#       outplot_bestfit: plot of best-fit line and uncertainty overlaid on the data
#
# MODIFICATION HISTORY:
#       Harry Desmond, Oxford, June 2018: main functions and MCMC implementation
#       Federico Lelli, Cardiff, February 2020: ODR and priors implementation
#       Harry Desmond, East Knoyle, April 2020: I/O implementations & py-module
#       Federico Lelli, Cardiff, June 2020: change severe errors with warnings
#       Harry Desmond, East Knoyle, July 2020: fixed likelihood plot bug
#-
#------------------------------------------------------------------------------


def BayesLineFit(x, y, err_x=None, err_y=None, orthfit=True, nwalkers=50, max_iters=10000, outfile_chain="outchain.dat", outfile_bestfit="bestfitvalues.dat", outplot_convergence="convergence", outplot_corner="cornerplot", outplot_bestfit="bestfitplot", slope_bounds=None, int_bounds=None, plotpdf=True, quiet=False, plot_title=None):
    '''
    Performs a Bayesian fit of a straight line to data including orthogonal or vertical intrinsic scatter

    Args:
        x (1D float array): Data on x-axis
        y (1D float array): Data on y-axis
        err_x (float or 1D float array, optional): Uncertainty on x-coordinate; default=0
        err_y (float or 1D float array, optional): Uncertainty on y-coordinate; default=0
        orthfit (bool, optional): Whether to model scatter in the orthogonal (True) or vertical (False) direction
        nwalkers (int, optional): Number of emcee walkers
        max_iters (int, optional): Maximum number of iterations in MCMC
        outfile_chain (str, optional): File in which to store output chain; set to None to suppress output
        outfile_bestfit (str, optional): File in which to store best-fitting parameters; set to None to suppress output
        outplot_convergence (str, optional): Name of walker convergence plot; set to None to suppress output
        outplot_corner (str, optional): Name of corner plot; set to None to suppress output
        outplot_bestfit (str, optional): Name of plot of best-fit line superimposed on data; set to None to suppress output
        slope_bounds (1D float array of length 2, optional): Min and max allowed value of slope; default=wide range
        int_bounds (1D float array of length 2, optional): Min and max allowed value of intercept; default=wide range
        plotpdf (bool): Create plots in pdf (True) or png (False) format
        quiet (bool): Suppress output to screen (True) or show it (False)

    Returns a, b, s, sobs:
        a (1D float array, length 4): Maximum-likelihood, median, upper error and lower error values of slope
        b (1D float array, length 4): Maximum-likelihood, median, upper error and lower error values of intercept
        s (1D float array, length 4): Maximum-likelihood, median, upper error and lower error values of intrinsic scatter
        sobs (float): Maximum-likelihood rms observed scatter (orthogonal if orthfit, vertical otherwise)
    '''
    if err_x is None:
        err_x = np.abs(x)/1.e10         # Assume small uncertainties by default, just to soften likelihood
    if err_y is None:
        err_y = np.abs(y)/1.e10

    if isinstance(err_x, (list, tuple, np.ndarray)):
        if any(t < 0. for t in err_x):
            print("At least one of your x errors is negative.")
            quit()
    elif err_x<0:
        print("Your x error is negative")
        quit()
    if isinstance(err_y, (list, tuple, np.ndarray)):
        if any(t < 0. for t in err_y):
            print("At least one of your y errors is negative.")
            quit()
    elif err_y<0:
        print("Your y error is negative")
        quit()

    if outfile_chain is not None and type(outfile_chain) != str:
        raise TypeError("outfile_chain should be a string")
    if outfile_bestfit is not None and type(outfile_bestfit) != str:
        raise TypeError("outfile_bestfit should be a string")
    if outplot_convergence is not None and type(outplot_convergence) != str:
        raise TypeError("outplot_convergence should be a string")
    if outplot_corner is not None and type(outplot_corner) != str:
        raise TypeError("outplot_corner should be a string")
    if outplot_bestfit is not None and type(outplot_bestfit) != str:
        raise TypeError("outplot_bestfit should be a string")

    if slope_bounds is not None:
        if len(slope_bounds) != 2:
            raise TypeError("slope_bounds must be a float list of length 2")
        slope_min = float(slope_bounds[0])
        slope_max = float(slope_bounds[1])

    if int_bounds is not None:
        if len(int_bounds) != 2:
            raise TypeError("int_bounds must be a float list of length 2")
        int_min = float(int_bounds[0])
        int_max = float(int_bounds[1])

    if not quiet:
        print("Number of data points:", len(x))

    ##### CORRELATION TESTS ######

    Pearson = stats.pearsonr(x, y)
    Spearman = stats.spearmanr(x, y)
    Kendall = stats.kendalltau(x, y)
    if not quiet:
        print("-------------------------")
        print("PEARSON'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Pearson[0]), float('%.4g'%Pearson[1])))
        print("-------------------------")
        print("SPEARMAN'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Spearman[0]), float('%.4g'%Spearman[1])))
        print("-------------------------")
        print("KENDALL'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Kendall[0]), float('%.4g'%Kendall[1])))
        print("-------------------------")

    ###### INITIAL ODR FIT #######

    model = Model(func)
    data = RealData(x, y, err_x, err_y)
    odr = ODR(data, model, [1,1])
    odr.set_job(fit_type=0)
    output = odr.run()
    a_ODR = output.beta[0] #slope
    b_ODR = output.beta[1] #normalization
    err_a_ODR = output.sd_beta[0] #error on slope
    err_b_ODR = output.sd_beta[1] #error on norm
    res_ODR = (y - output.beta[0]*x - output.beta[1]) #residuals
    rms_ODR = np.sqrt(np.mean(res_ODR**2., dtype=np.float64)) #observed scatter
    s_ODR = np.sqrt(rms_ODR**2. - np.mean(err_y)**2.) #estimate of intrinsic scatter
    if (np.isnan(a_ODR) or np.isnan(b_ODR)):
        sys.exit("-----------------------------------------------------------\n\
SEVERE ERROR: The INITIAL ODR FIT FAILED! CHECK YOUR INPUTS!\n\
--------------------------------------------------------------")
    if (np.isnan(s_ODR)):
        print("-----------------------------------------------------------\n\
WARNING: OBSERVED VERTICAL SCATTER IS SMALLER THAN EXPECTED FROM Y-ERRORS. YOUR ERRORS MAY BE OVER-ESTIMATED!\n\
--------------------------------------------------------------")

    if slope_bounds is None:
        slope_min = a_ODR - 20.*err_a_ODR
        slope_max = a_ODR + 20.*err_a_ODR
    if int_bounds is None:
        int_min = b_ODR - 20.*err_b_ODR
        int_max = b_ODR + 20.*err_b_ODR

    ### BAYESIAN FIT ####

    # Initialize walkers
    ndim = 3
    p0 = []
    for i in range(nwalkers):
        pi = [np.random.normal(a_ODR, err_a_ODR), np.random.normal(b_ODR, err_b_ODR), np.random.uniform(rms_ODR/10., rms_ODR)]
        p0.append(pi)

    index = 0
    autocorr = np.empty(max_iters)
    old_tau = np.inf

    if not quiet:
        print("Running MCMC with", cpu_count(), "cores. Please wait...")

    start = time.time()
    with Pool() as pool:
        if orthfit:        # Default sampler move
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_orthogonal, args=[x, err_x, y, err_y, slope_min, slope_max, int_min, int_max], pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_vertical, args=[x, err_x, y, err_y, slope_min, slope_max, int_min, int_max], pool=pool)

        for sample in sampler.sample(p0, iterations=max_iters, progress=False):
            if sampler.iteration==max_iters-1:
                print("The sampler did not converge. Probably either your data is pathological or a linear fit is bad or unconstrained. If you think this is not the case, try increasing the maximum number of iterations, imposing bounds on the parameters, or using a different move method in the sampling.")
                quit()

            if sampler.iteration % 100:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

    end = time.time()

    if not quiet:
        print("Converged after", sampler.iteration, "iterations in", round(end-start), "seconds")
        print("Mean acceptance fraction:", round(np.mean(sampler.acceptance_fraction),3))
        print("-------------------------")

    # Calculate autocorrelation and make convergence plots
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=False, thin=thin)
    log_prob_samples_flat = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

    mask = log_prob_samples_flat > -1.e300
    samples = samples[mask]

    if outplot_convergence is not None:
        fig = plt.figure(figsize=(10, 4), dpi=300)
        for i in range(nwalkers):
            y_arr = log_prob_samples[:,i]
            x_arr = np.arange(0, len(y_arr), 1)
            plt.plot(x_arr, y_arr, '.')
        plt.xlabel("Walker step")
        plt.ylabel("ln(Likelihood)")
        if plotpdf:
            plt.savefig(outplot_convergence+".pdf", format='pdf', dpi=300)
        else:
            plt.savefig(outplot_convergence+".png")

    # Write chain file
    if outfile_chain is not None:
        all_samples = np.concatenate((samples, log_prob_samples_flat[mask, None]), axis=1)
        np.savetxt(outfile_chain, all_samples, header="slope intercept sigma lnLike")

    # Maximum-likelihood values
    ML = np.max(log_prob_samples_flat)
    index = np.where(log_prob_samples_flat==ML)[0][0]
    a_ML = samples[index,0]
    b_ML = samples[index,1]
    s_ML = samples[index,2]
    # Posterior medians
    a_med = corner.quantile(samples[:,0], 0.5)
    b_med = corner.quantile(samples[:,1], 0.5)
    s_med = corner.quantile(samples[:,2], 0.5)
    # Errors (down/up)
    a_dw = corner.quantile(samples[:,0], 0.16)-a_med
    a_up = corner.quantile(samples[:,0], 0.84)-a_med
    b_dw = corner.quantile(samples[:,1], 0.16)-b_med
    b_up = corner.quantile(samples[:,1], 0.84)-b_med
    s_dw = corner.quantile(samples[:,2], 0.16)-s_med
    s_up = corner.quantile(samples[:,2], 0.84)-s_med
    # Residuals and observed scatter
    res_ML = (y - a_ML*x - b_ML)
    rms_ML = np.sqrt(np.mean(res_ML**2., dtype=np.float64))
    res_ML_orth = (y - a_ML*x - b_ML)/np.sqrt(a_ML*a_ML + 1.)
    rms_ML_orth = np.sqrt(np.mean(res_ML_orth**2., dtype=np.float64))

    # Write outputs
    if not quiet:
        print("Maximum likelihood (ML) value: %s" % (float('%.4g'%ML)))
        print("Slope (ML, median, upper error, lower error): %s; %s; +%s, %s" % (float('%.4g'%a_ML), float('%.4g'%a_med), float('%.4g'%a_up), float('%.4g'%a_dw)))
        print("Intercept (ML, median, upper error, lower error): %s; %s; +%s; %s" % (float('%.4g'%b_ML), float('%.4g'%b_med), float('%.4g'%b_up), float('%.4g'%b_dw)))
        print("Intrinsic scatter (ML, median, upper error, lower error): %s; %s; +%s; %s" % (float('%.4g'%s_ML), float('%.4g'%s_med), float('%.4g'%s_up), float('%.4g'%s_dw)))
        print("ML observed scatter (vertical): %s" % (float('%.4g'%rms_ML)))
        print("ML observed scatter (orthogonal): %s" % (float('%.4g'%rms_ML_orth)))
        print("*** NB medians and errors only meaningful for unimodal posteriors. Check the corner plot! ***")
        print("-------------------------")

    if outfile_bestfit is not None:
        f = open(outfile_bestfit, "w")
        f.write("Pearson r: %f; p-value: %f \n" % (Pearson[0], Pearson[1]))
        f.write("Spearman rho: %f; p-value: %f \n" % (Spearman[0], Spearman[1]))
        f.write("Kendall tau: %f; p-value: %f \n" % (Kendall[0],Kendall[1]))
        f.write("Maximum likelihood (ML) value: %s\n" % (float('%.4g'%ML)))
        f.write("Slope (ML, median, upper error, lower error): %s; %s; +%s, %s\n" % (float('%.4g'%a_ML), float('%.4g'%a_med), float('%.4g'%a_up), float('%.4g'%a_dw)))
        f.write("Intercept (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (float('%.4g'%b_ML), float('%.4g'%b_med), float('%.4g'%b_up), float('%.4g'%b_dw)))
        f.write("Intrinsic scatter (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (float('%.4g'%s_ML), float('%.4g'%s_med), float('%.4g'%s_up), float('%.4g'%s_dw)))
        f.write("ML rms observed scatter (vertical): %s\n" % (float('%.4g'%rms_ML)))
        f.write("ML rms observed scatter (orthogonal): %s\n" % (float('%.4g'%rms_ML_orth)))
        f.close()

    if outplot_corner is not None:
        # Make corner plot
        # Sigma levels in 2D = 1-exp( (x/r)**2/2)
        figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.5,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10})
        # Maximum-likelihood values
        Med_value = [a_med, b_med, s_med]
        axes = np.array(figure.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(Med_value[i], color="r")
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(Med_value[xi], color="r")
                ax.axhline(Med_value[yi], color="r")
                ax.plot(Med_value[xi], Med_value[yi], "sr")
        if plotpdf:
            figure.savefig(outplot_corner+".pdf", format='pdf', dpi=300)
        else:
            figure.savefig(outplot_corner+".png")

    if outplot_bestfit is not None:
        # Make best-fit plot
        fig = plt.figure(figsize=(4, 4), dpi=300)
        x_vec = np.array([np.min(x), np.max(x)])
        y_vec = a_med*x_vec + b_med
        y_old_vec = 1.54*x_vec - 1.175  # K&dlR+21
        plt.errorbar(x, y, xerr=err_x, yerr=err_y, fmt = ' ', ecolor = 'grey', capsize = 3, alpha=0.3, zorder=0)
        plt.scatter(x, y, color="magenta", edgecolor='k', linewidth=1, s=15, zorder=1)
        if plot_title == "boissier":
            plt.plot(x_vec, y_vec, '-r', zorder=3, label=f"Fit: α={10**b_med[0]:.2e}, n={a_med[0]:.2f}")
            plt.title(r"Boissier SFL")
            plt.xlabel(r"log$(\Sigma_{gas})$ $(M_\odot pc^{-2})$")
            plt.ylabel(r"log$(\Sigma_{SFR}/\Omega)$ $(M_\odot pc^{-2})$")
        if plot_title == "kennicutt":
            plt.plot(x_vec, y_vec, '-r', zorder=3, label=f"Fit: A={10**b_med[0]:.2e}, N={a_med[0]:.2f}")
            plt.plot(x_vec, y_old_vec, '--o', zorder=3, label=f"K&dlR+21 (A={10**(-1.175):.2e}, N={1.54})")
            plt.title(r"Kennicutt-Schmidt SFL")
            plt.xlabel(r"log$(\Sigma_{gas})$ $(M_\odot pc^{-2})$")
            plt.ylabel(r"log$(\Sigma_{SFR})$ $(M_\odot pc^{-2} Gyr^{-1})$")
        plt.grid(zorder=0)
        plt.legend()
        if orthfit:
            dist = np.sqrt( s_med**2 + (s_med*a_med)**2. )
            ymin = y_vec - dist
            ymax = y_vec + dist
            plt.fill_between(x_vec, ymin, ymax, color='k', alpha=0.2, zorder=2)
        else:
            ymin = y_vec - s_med
            ymax = y_vec + s_med
            plt.fill_between(x_vec, ymin, ymax, color='k', alpha=0.2, zorder=2)
        if plotpdf:
            fig.savefig(outplot_bestfit+".pdf", format='pdf', dpi=300)
        else:
            fig.savefig(outplot_bestfit+".png")

    # Return best-fit arrays for slope, intercept and scatter, and observed scatter
    a = [a_ML, a_med[0], a_up[0], a_dw[0]]
    b = [b_ML, b_med[0], b_up[0], b_dw[0]]
    s = [s_ML, s_med[0], s_up[0], s_dw[0]]
    if orthfit:
        sobs = rms_ML_orth
    else:
        sobs = rms_ML

    return a, b, s, sobs


def func(beta, x):
    '''Linear model to fit'''
    y = beta[0]*x + beta[1]
    return y

def lnprob_vertical(x, x_arr, err_x_arr, y_arr, err_y_arr, slope_min, slope_max, int_min, int_max):
    '''Likelihood function for vertical scatter'''
    slope, intercept, sigma = x[0], x[1], x[2]
    if sigma < 0. or slope < slope_min or slope > slope_max or intercept < int_min or intercept > int_max:
        return -1.e300
    # Expected M for given V
    mean = intercept + slope * np.array(x_arr)
    # Difference between point and line in vertical direction
    dist = np.array(y_arr) - mean
    # Sum up in quadrature contributions to scatter in vertical direction
    scatter = np.sqrt(np.array(err_y_arr)*np.array(err_y_arr) + slope*slope*np.array(err_x_arr)*np.array(err_y_arr) + sigma*sigma)
    chi_sq = dist*dist / (scatter*scatter)
    # Define Gaussian likelihood
    L = np.exp(-chi_sq/2.) / (np.sqrt(2.*np.pi) * scatter)
    if np.min(L) < 1.e-300:
        return -1.e300
    return np.sum(np.log(L))

def lnprob_orthogonal(x, x_arr, err_x_arr, y_arr, err_y_arr, slope_min, slope_max, int_min, int_max):
    '''Likelihood function for orthogonal scatter'''
    slope, intercept, sigma = x[0], x[1], x[2]
    if sigma < 0. or slope < slope_min or slope > slope_max or intercept < int_min or intercept > int_max:
        return -1.e300
    # Expected y for given x
    mean = intercept + slope * np.array(x_arr)
    # Difference between point and line in vertical direction
    delta_y = y_arr - mean
    # Shortest squared distance between point and line (in orthogonal direction)
    dist2 = (y_arr - slope*x_arr - intercept)**2./(slope**2 + 1.)
    # Sum up in quadrature contributions to scatter in orthogonal direction
    scatter2 = sigma*sigma + dist2/(delta_y*delta_y)*np.array(err_y_arr)*np.array(err_y_arr) + (1. - dist2/(delta_y*delta_y))*np.array(err_x_arr)*np.array(err_x_arr)
    chi_sq = dist2 / scatter2
    # Define Gaussian likelihood
    L = np.exp(-chi_sq/2.) / np.sqrt(2.*np.pi*scatter2)
    if np.min(L) < 1.e-300:
        return -1.e300
    return np.sum(np.log(L))

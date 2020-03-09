import pandas as pd
from datetime import datetime
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.stats as stats
import numpy as np
from dataclasses import dataclass
from tarpan.cmdstanpy.analyse import save_analysis
from tarpan.cmdstanpy.tree_plot import save_tree_plot
from tarpan.cmdstanpy.cache import run
from tarpan.cmdstanpy.compare_parameters import save_compare_parameters
from tarpan.shared.compare_parameters import CompareParametersType
from tarpan.shared.info_path import InfoPath, get_info_path
from tarpan.shared.tree_plot import TreePlotParams
from tarpan.shared.summary import save_summary
from tarpan.shared.histogram import save_histogram
from tarpan.shared.tree_plot import save_tree_plot as shared_save_tree_plot
from tarpan.plot.kde import save_scatter_and_kde

from tarpan.plot.posterior import (
    save_posterior_scatter_and_kde, PosteriorKdeParams)


# Parameters for data analysys
@dataclass
class AnalysisSettings:
    # Name of the parent directory where the plots will be created.
    dir_name: str = "model_info"

    # Data for Stan model
    data = None

    # Path to the .stan model file
    stan_model_path: str = None

    # Stan's sampling parameter
    max_treedepth: float = 10

    # Location of plots and summaries
    info_path: InfoPath = None


def load_data(data_path):
    """
    Load data.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.


    Returns
    -------
    list of datetime:
        Days corresponding to the confirmed cases
    list of float:
        Number of people infected (confirmed).
    """
    df = pd.read_csv(data_path)
    column_names = list(df)
    i_first_day = column_names.index('1/22/20')  # First date column

    dates = []
    cases = []

    for i_day in range(i_first_day, len(column_names)):
        column_name = column_names[i_day]
        date = datetime.strptime(column_name, '%m/%d/%y')
        dates.append(date)
        confirmed = df[column_name].sum()
        cases.append(confirmed)

    return dates, cases


def standardise(measurements):
    """
    Converts an array to standardised values
    (i.e. z-scores, values minus their means divided by standard deviation)

    Parameters
    ----------

    measurements: list of float
        Values to standardise

    Returns
    --------
    np.array:
        Standardised values.
    """
    measurements = np.array(measurements)
    mean = measurements.mean()
    std = measurements.std()

    return (measurements - mean) / std


def standardised_data_for_stan(measurements, uncertainties,
                               sigma_stdev_prior):
    """
    Returns data for the model.

    Parameters
    ----------

    measurements: list of float
        Measurements.
    uncertainties: list of float
        Measurement uncertainties.
    sigma_stdev_prior: float
        Prior for standard deviation of the population spread,
        in measurement units (i.e. non-stsndardised).

    Returns
    -------

    dict:
        Data that is supplied to Stan model.
    """

    measurements = np.array(measurements)
    std = measurements.std()
    measurements_std = standardise(measurements=measurements)
    uncertainties_std = uncertainties / std
    sigma_stdev_prior_std = sigma_stdev_prior / std

    return {
        "y": measurements_std,
        "uncertainties": uncertainties_std,
        "n": len(measurements_std),
        "sigma_stdev_prior": sigma_stdev_prior_std
    }


def destandardise_stan_output(fit, measurements):
    """
    Transform Stan output values from standardised form to the scale
    of observations.

    Parameters
    ----------

    fit: cmdstanpy.stanfit.CmdStanMCMC
        Stan'd output to standardise. The function transforms `fit.sample`
        array from standard form.

    measurements: list of float
        Measurements, non-standardised.
    """
    measurements = np.array(measurements)
    mean = measurements.mean()
    std = measurements.std()

    # Destandardise 'mu' parameters: 'mu.1', 'mu.2' etc.
    # -----------

    mu_columns = [name for name in fit.column_names if name.startswith('mu')]

    for column_name in mu_columns:
        column_id = fit.column_names.index(column_name)

        for chain_id in range(fit.chains):
            values = fit.sample[:, chain_id, column_id]
            fit.sample[:, chain_id, column_id] = values * std + mean


    # De-standardise `sigma`
    # ----------

    column_id = fit.column_names.index('sigma')

    for chain_id in range(fit.chains):
        values = fit.sample[:, chain_id, column_id]
        fit.sample[:, chain_id, column_id] = values * std


def run_stan(output_dir, settings: AnalysisSettings):
    """
    Run Stan model and return the samples from posterior distributions.

    Parameters
    ----------
    output_dir: str
        Directory where Stan's output will be created
    settings: AnalysisSettings
        Analysis settings.

    Returns
    -------
    cmdstanpy.CmdStanMCMC
        Stan's output containing samples of posterior distribution
        of parameters.
    """

    model = CmdStanModel(stan_file=settings.stan_model_path)

    fit = model.sample(
        data=settings.data, seed=333,
        adapt_delta=0.99, max_treedepth=settings.max_treedepth,
        sampling_iters=4000, warmup_iters=1000,
        chains=4, cores=4,
        show_progress=True,
        output_dir=output_dir)

    # Make summaries and plots of parameter distributions
    save_analysis(fit, param_names=["r", "mu", "sigma"],
                  info_path=settings.info_path)

    return fit


def check_all_days_present(dates):
    """
    Throws exception if there are days missing in the `dates` array
    (for example, if it's Sep 1, Sep 2, Sep 3, Sep 5, where Sep 4 is missing).
    """

    prev_day = None

    for date in dates:
        if prev_day is None:
            prev_day = date
            continue

        delta = date - prev_day

        if delta.days != 1:
            raise ValueError(f'ERROR: missing days between {prev_day} and {date}')

        prev_day = date


def run_model(settings):
    dates, cases = load_data("data/time_series_19-covid-Confirmed.csv")
    check_all_days_present(dates)

    # plt.scatter(dates, cases)
    # plt.savefig("confirmed.png", dpi=300)
    # fit = run(func=run_stan, settings=settings)

    # return fit


if __name__ == '__main__':
    print("Running the model...")
    settings = AnalysisSettings(stan_model_path="code/stan_model/model.stan")
    run_model(settings=settings)
    print('We are done')

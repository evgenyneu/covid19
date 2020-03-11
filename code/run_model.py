import os
import shutil
from shutil import copyfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
from dateutil import rrule
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmdstanpy import CmdStanModel
from dataclasses import dataclass
from tarpan.cmdstanpy.analyse import save_analysis
from tarpan.shared.info_path import InfoPath
from tarpan.cmdstanpy.cache import run
import tarpan


# Parameters for data analysys
@dataclass
class AnalysisSettings:
    # Data for Stan model (dictionary)
    data = None

    csv_path: str = "data/time_series_19-covid-Confirmed.csv"

    # URL to the data
    data_url: str = "https://raw.githubusercontent.com/CSSEGISandData/\
COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/\
time_series_19-covid-Confirmed.csv"

    # Path to the .stan model file
    stan_model_path: str = "code/stan_model/logistic.stan"

    # Stan's sampling parameter
    max_treedepth: float = 10

    # Location of plots and summaries
    info_path: InfoPath = InfoPath()

    # Fraction of people who have cofonavarus and who have been reported
    # as confirmed case. For example,
    # 1 means all, 0.5 means 50% of sick people get reported)
    fraction_cofirmed = 1

    population_size: float = 7800000

    marker_color: str = "#0060ff44"

    marker_edgecolor: str = "#0060ff"

    mu_line_color: str = "#444444"
    mu_hpdi_color: str = "#44444477"
    cases_hpdi_color: str = "#ff770066"

    marker: str = "o"

    grid_color: str = "#aaaaaa"

    grid_alpha: float = 0.2


def download_data(settings):
    """
    Downloads the CSV file containing data about convermed COVID-19 cases
    """

    data_path = settings.csv_path
    data_url = settings.data_url

    time_now = datetime.now()
    mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
    delta = time_now - mod_time
    delta_hours = delta.total_seconds() / 60 / 60
    max_hours_diff = 12

    if delta_hours < max_hours_diff:
        # Data is up to date
        return

    shutil.rmtree(settings.info_path.dir())  # Remove data directory
    print(f"Data last downloaded {round(delta_hours)} hours ago.")
    print(f"Re-downloading data from:\n{data_url}")

    # Download
    response = requests.get(data_url)

    try:
        # Check if download was successful
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"Error downloading data from {data_url}.")
        print(f"Using previous data")
        return

    data = response.text

    # Save to file
    with open(data_path, "w") as text_file:
        text_file.write(data)

    # Save with time stamp in archive folder
    # ------

    path = Path(data_path)
    data_dir = path.parent
    archive_dir = os.path.join(data_dir, "archive", "confirmed")

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)

    archive_file_name = time_now.strftime('%Y-%m-%d.csv')
    archive_path = os.path.join(archive_dir, archive_file_name)

    copyfile(data_path, archive_path)


def load_data(settings):
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

    data_path = settings.csv_path
    download_data(settings=settings)

    df = pd.read_csv(data_path)
    df = df[df['Country/Region'] != 'Mainland China']
    column_names = list(df)
    i_first_day = column_names.index('1/22/20')  # First date column

    dates = []
    cases = []

    for i_day in range(i_first_day, len(column_names)):
        column_name = column_names[i_day]
        date = datetime.strptime(column_name, '%m/%d/%y')
        dates.append(date)
        confirmed = df[column_name].sum()
        cases.append(int(confirmed))

    return dates, cases


def data_for_stan(cases, settings):
    """
    Returns data for the model.

    Parameters
    ----------

    list of float:
        Number of people infected (confirmed).

    Returns
    -------

    dict:
        Data that is supplied to Stan model.
    """

    total_confirmed = settings.population_size * settings.fraction_cofirmed

    q = -1 + total_confirmed / cases[0]

    return {
        "n": len(cases),
        "cases": cases,
        "k": total_confirmed,
        "q": q
    }


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
    save_analysis(fit, param_names=["b", "sigma"])

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
            raise ValueError(
                f'ERROR: missing days between {prev_day} and {date}')

        prev_day = date


def model_function(x, k, q, b):
    """
    Calculates number of infected people using logistic function.

    Parameters
    ---------
    x: numpy.ndarray
        Day numbers

    k, q, b: float
        Parameters of logitic function.

    Returns
    -------
    numpy.ndarray:
        Number of infected people
    """

    return float(k) / (1 + q * np.exp(-(b * x)))


def simulated(mu, sigma):
    return stats.norm.rvs(size=len(sigma), loc=mu, scale=sigma)


def calculate_all_infected_day(k, q, b):
    """
    Calculates the day when almost all almost people that can be reported
    are reported.

    Parameters
    ----------

    k, q, b: float
        Parameters of logitic function.

    Returns
    -------

    The day number when almost all people that can be reported as infected
    are reported.
    """

    day_all_infected = 0
    b_mean = b.mean()

    while True:
        sim_confirmed = model_function(x=day_all_infected, k=k, q=q, b=b_mean)

        if abs(sim_confirmed - k) < 1000:
            break

        day_all_infected += 1

    return day_all_infected


def plot_data_and_model(fit, dates, cases, settings):
    sns.set(style="ticks")
    posterior = fit.get_drawset(params=['b', 'sigma'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.ticklabel_format(style='sci')

    # Plot data
    # ----------

    ax.scatter(dates, cases,
               marker=settings.marker,
               color=settings.marker_color,
               edgecolor=settings.marker_edgecolor)

    # Plot posterior
    # ---------

    # Model parameters
    b = posterior["b"].to_numpy()  # Growth rate
    sigma = posterior["sigma"].to_numpy()  # Spear of observations
    k = settings.data["k"]  # Maximum number cases that can be confirmed
    q = settings.data["q"]  # Parameter related to initial number of infected
    n = settings.data['n']  # Number of data points

    day_all_infected = calculate_all_infected_day(k=k, q=q, b=b)
    x_values = np.array(range(0, day_all_infected))

    mu = [
        model_function(x=x, k=k, q=q, b=b)
        for x in x_values
    ]

    mu = np.array(mu)

    # Plot mean
    mu_mean = mu.mean(axis=1)

    x_dates = list(rrule.rrule(freq=rrule.DAILY,
                               count=day_all_infected, dtstart=dates[0]))

    x_dates = np.array(x_dates)
    ax.plot(x_dates, mu_mean, color=settings.mu_line_color)

    # Plot HPDI interval
    # --------

    hpdi = np.apply_along_axis(tarpan.shared.stats.hpdi, 1, mu,
                               probability=0.95)

    ax.fill_between(x_dates, hpdi[:, 0], hpdi[:, 1],
                    facecolor=settings.mu_hpdi_color)

    # Plot simulated observations

    simulated_cases = [
        simulated(mu=mu[i, :], sigma=sigma)
        for i in range(len(x_values))
    ]

    simulated_cases = np.array(simulated_cases)

    cases_hpdi = np.apply_along_axis(
        tarpan.shared.stats.hpdi, 1, simulated_cases, probability=0.95)

    ax.fill_between(x_dates,
                    cases_hpdi[:, 0], cases_hpdi[:, 1],
                    facecolor=settings.cases_hpdi_color,
                    linewidth=0)

    # Format plot
    # ----------

    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_format)

    title = (
        "Number of people infected with COVID-19 worldwide\n"
        "Data from Johns Hopkins University, Mar 10, 2010"
    )

    ax.set_title(title)
    ax.set_ylabel("Infected people (confirmed cases)")

    ax.grid(color=settings.grid_color, linewidth=1,
            alpha=settings.grid_alpha)

    fig.tight_layout()

    # Plot entire population
    # ---------
    info_path = InfoPath(**settings.info_path.__dict__)
    info_path.base_name = "covid19_infected_population"
    info_path.extension = "png"
    fig.savefig(str(info_path), dpi=info_path.dpi)

    # Plot at scale of observations
    # ---------

    last_data = simulated_cases.mean(axis=1)[n - 1]
    margins = last_data * 0.1
    day_margin = timedelta(days=1)
    ax.set_xlim([dates[0] - day_margin, dates[-1] + day_margin])
    ax.set_ylim([0 - margins, last_data + margins])
    info_path.base_name = "covid19_infected"
    info_path.extension = "png"
    fig.savefig(str(info_path), dpi=info_path.dpi)


def do_work():
    register_matplotlib_converters()
    settings = AnalysisSettings()
    dates, cases = load_data(settings=settings)
    check_all_days_present(dates)
    settings.data = data_for_stan(cases, settings=settings)
    # output_dir = os.path.join(settings.info_path.dir(), "stan_cache")
    # fit = run_stan(output_dir=output_dir, settings=settings)
    fit = run(func=run_stan, settings=settings)
    plot_data_and_model(fit=fit, dates=dates, cases=cases, settings=settings)


if __name__ == '__main__':
    print("Running the model...")
    do_work()
    print('We are done')

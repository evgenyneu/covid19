# Modelling spread of infectious desease using logisic growth model

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
import matplotlib
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
time_series_covid19_confirmed_global.csv"

    # Path to the .stan model file
    stan_model_path: str = "code/stan_model/logistic.stan"

    # Location of plots and summaries
    info_path: InfoPath = InfoPath()

    plots_dir: str = "plots"

    # Stan's sampling parameter
    max_treedepth: float = 10

    # Number of hours to wait before downloading the data from the Web
    max_hours_diff = 12

    # Width of HPDI (highest posterior density interval) that is used
    # to plot the shaded region around the predicted mean line.
    hpdi_width: float = 0.95

    # Maximum number of people that can be infected
    population_size: float = 1_900_000

    # Difference between the maximum number of confirmed cases
    # and the actual number of confirmed cases at which we consider
    # all people to be reported
    tolerance_cases = 1000

    marker_color: str = "#F4A92800"
    marker_edgecolor: str = "#F4A928"

    mu_line_color: str = "#28ADF4"
    mu_hpdi_color: str = "#6ef48688"
    cases_hpdi_color: str = "#e8e8f455"

    # Plot's background color
    background_color = '#023D45'

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

    if delta_hours < settings.max_hours_diff:
        # Data is up to date
        return

    # Remove data directory
    shutil.rmtree(settings.info_path.dir(), ignore_errors=True)

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
        Cumulative number of people infected (confirmed).
    """

    data_path = settings.csv_path
    download_data(settings=settings)

    df = pd.read_csv(data_path)

    # Exclude China because its data do not show exponential growth
    df = df[df['Country/Region'] != 'Mainland China']
    df = df[df['Country/Region'] != 'China']

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
        Cumulative number of people infected (confirmed).

    Returns
    -------

    dict:
        Data that is supplied to Stan model.
    """

    q = -1 + settings.population_size / cases[0]

    return {
        "n": len(cases),
        "cases": cases,
        "k": settings.population_size,
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
        iter_sampling=4000, iter_warmup=1000,
        chains=4, cores=4,
        show_progress=True,
        output_dir=output_dir)

    # Make summaries and plots of parameter distributions
    save_analysis(fit, param_names=["r", "sigma"])

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
        Cumulative number of infected people
    """

    return float(k) / (1 + q * np.exp(-(b * x)))


def simulated(mu, sigma):
    return stats.norm.rvs(size=len(sigma), loc=mu, scale=sigma)


def calculate_all_infected_day(k, q, b, settings):
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

        # Stop if number of confirmed cases is almost at maximum level
        if abs(sim_confirmed - k) < settings.tolerance_cases:
            break

        day_all_infected += 1

    return day_all_infected


def plot_data_and_model(fit, dates, cases, settings):
    sns.set(style="ticks")
    plt.style.use('dark_background')
    posterior = fit.get_drawset(params=['r', 'sigma'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor(settings.background_color)
    fig.set_facecolor(settings.background_color)

    # Plot posterior
    # ---------

    # Model parameters
    b = posterior["r"].to_numpy()  # Growth rate
    sigma = posterior["sigma"].to_numpy()  # Spear of observations
    k = settings.data["k"]  # Maximum number cases that can be confirmed
    q = settings.data["q"]  # Parameter related to initial number of infected
    n = settings.data['n']  # Number of data points

    day_all_infected = calculate_all_infected_day(k=k, q=q, b=b,
                                                  settings=settings)

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
    ax.plot(x_dates, mu_mean, color=settings.mu_line_color, label="Model",
            zorder=10)

    # Plot HPDI interval
    # --------

    hpdi = np.apply_along_axis(tarpan.shared.stats.hpdi, 1, mu,
                               probability=settings.hpdi_width)

    ax.fill_between(x_dates, hpdi[:, 0], hpdi[:, 1],
                    facecolor=settings.mu_hpdi_color, zorder=7,
                    linewidth=0)

    # Plot simulated observations

    simulated_cases = [
        simulated(mu=mu[i, :], sigma=sigma)
        for i in range(len(x_values))
    ]

    simulated_cases = np.array(simulated_cases)

    cases_hpdi = np.apply_along_axis(
        tarpan.shared.stats.hpdi, 1, simulated_cases,
        probability=settings.hpdi_width)

    ax.fill_between(x_dates,
                    cases_hpdi[:, 0], cases_hpdi[:, 1],
                    facecolor=settings.cases_hpdi_color,
                    linewidth=0,
                    label=f"{round(settings.hpdi_width*100)}% HPDI", zorder=5)

    # Plot data
    # ----------

    ax.scatter(dates, cases,
               marker=settings.marker,
               color=settings.marker_color,
               edgecolor=settings.marker_edgecolor,
               label="Reported",
               zorder=9)

    # Format plot
    # ----------

    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_format)

    date_str = datetime.now().strftime('%b %d, %Y')

    title = (
        "Total confirmed cases of COVID-19 worldwide, excluding China.\n"
        f"Data retrieved from Johns Hopkins University on {date_str}."
    )

    ax.set_title(title)
    ax.set_ylabel("Total confirmed cases")

    ax.grid(color=settings.grid_color, linewidth=1,
            alpha=settings.grid_alpha)

    # Set thousand separator
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax.legend(facecolor=settings.background_color)
    fig.tight_layout()

    # Save images
    # ------

    os.makedirs(settings.plots_dir, exist_ok=True)

    # Plot predictions into the future
    # ---------

    day_margin = timedelta(days=5)
    ax.set_xlim([dates[0] - day_margin, x_dates[-1] + day_margin])
    info_path = InfoPath(**settings.info_path.__dict__)
    filename_date = datetime.now().strftime('%Y_%m_%d')
    filename = f"{filename_date}_extrapolated.png"
    image_path = os.path.join(settings.plots_dir, filename)
    fig.savefig(image_path, dpi=info_path.dpi, facecolor=fig.get_facecolor())
    print("Created plots:")
    print(image_path)

    filename = f"recent_extrapolated.png"
    image_path = os.path.join(settings.plots_dir, filename)
    fig.savefig(image_path, dpi=info_path.dpi, facecolor=fig.get_facecolor())

    # Plot at scale of observations
    # ---------

    last_data = cases[n - 1]
    margins = last_data * 0.1
    day_margin = timedelta(days=2)
    ax.set_xlim([dates[0] - day_margin, dates[-1] + day_margin])
    ax.set_ylim([0 - margins, last_data + margins])
    filename = f"{filename_date}_observed.png"
    image_path = os.path.join(settings.plots_dir, filename)
    fig.savefig(image_path, dpi=info_path.dpi, facecolor=fig.get_facecolor())
    print(image_path)


    filename = f"recent_observed.png"
    image_path = os.path.join(settings.plots_dir, filename)
    fig.savefig(image_path, dpi=info_path.dpi, facecolor=fig.get_facecolor())
    # plt.show()


def do_work():
    register_matplotlib_converters()
    settings = AnalysisSettings()
    dates, cases = load_data(settings=settings)
    check_all_days_present(dates)
    settings.data = data_for_stan(cases, settings=settings)
    output_dir = os.path.join(settings.info_path.dir(), "stan_cache")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    fit = run_stan(output_dir=output_dir, settings=settings)
    # fit = run(func=run_stan, settings=settings)
    plot_data_and_model(fit=fit, dates=dates, cases=cases, settings=settings)


if __name__ == '__main__':
    print("Running the model...")
    do_work()
    print('We are done')

import pandas as pd
from datetime import datetime
from cmdstanpy import CmdStanModel
from dataclasses import dataclass
from tarpan.cmdstanpy.analyse import save_analysis
from tarpan.cmdstanpy.cache import run
from tarpan.shared.info_path import InfoPath
from tarpan.shared.summary import save_summary


# Parameters for data analysys
@dataclass
class AnalysisSettings:
    # Name of the parent directory where the plots will be created.
    dir_name: str = "model_info"

    # Data for Stan model (dictionary)
    data = None

    # Path to the .stan model file
    stan_model_path: str = None

    # Stan's sampling parameter
    max_treedepth: float = 10

    # Location of plots and summaries
    info_path: InfoPath = None

    population_size: float = 7800000

    q: float = None


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

    q = -1 + settings.population_size / cases[0]
    print(q)

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
            raise ValueError(f'ERROR: missing days between {prev_day} and {date}')

        prev_day = date


def run_model(settings):
    dates, cases = load_data("data/time_series_19-covid-Confirmed.csv")
    check_all_days_present(dates)
    settings.data = data_for_stan(cases, settings=settings)

    # plt.scatter(dates, cases)
    # plt.savefig("confirmed.png", dpi=300)
    fit = run(func=run_stan, settings=settings)


if __name__ == '__main__':
    print("Running the model...")
    model_path = "code/stan_model/logistic.stan"
    settings = AnalysisSettings(stan_model_path=model_path)
    run_model(settings=settings)
    print('We are done')

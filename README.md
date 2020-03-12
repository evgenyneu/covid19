# Using logistic function to model number of COVID-19 confirmed cases in Stan and Python

This is a Stan model for analysing time series of confirmed cases of Coronavirus (COVID-19). The program does the following:

* Downloads new data on confirmed COVID-19 cases from Johns Hopkins University [Github's repository](
https://github.com/CSSEGISandData/COVID-19).

* Runs Stan mode that uses logistic function.

* Creates a plot showing observed data with round blue markers and the model with black line. The orange shaded region indicates model's uncertainty and corresponds to 95% DHPI (highest posterior density interval).

![Modelling COVID-19 confirmed cases with logistic function, observed cases](https://github.com/evgenyneu/covid19/raw/master/plots/recent_observed.png)

![Modelling COVID-19 confirmed cases with logistic function, extrapolated](https://github.com/evgenyneu/covid19/raw/master/plots/recent_extrapolated.png)

As the new data becomes available daily, I will be running this script and adding the new plots to [plots](plots) directory. This way we can see the changes in the data and the model with time.

## Assumptions

In the model we make the following assumptions:

* The number of cases can be modelled with logistic function.

* The virus is able to infect the entire population of 7.8 billion people.

* All people who are infected with COVID-19 are reported.


## Exclusion of China

The model does not include the data for China because the the number of confirmed cases have levelled off, as shown below. This could indicate that the virus has invected most people. Alternatively, it could mean that the data are not accurate, and/or that the health care system is overwhelmed and is not able to test all people who are infected.

![Modelling COVID-19 confirmed cases in China with logistic function, observed cases](https://github.com/evgenyneu/covid19/raw/master/plots/2020_03_12_observed_china.png)

![Modelling COVID-19 confirmed cases in China with logistic function, extrapolated](https://github.com/evgenyneu/covid19/raw/master/plots/2020_03_12_extrapolated_china.png)

## Setup

In order to run the model, download and install all required software with the following steps.


### 1. Install python libraries

First, install Python 3.7 or newer, and then run:

```
pip install tarpan
```

If you are having issues with running the code, use `pip install -Iv tarpan==0.3.8` instead.


### 2. Install Stan

```
install_cmdstan
```

## Usage

To run the code, first, download this repository:

```bash
git clone https://github.com/evgenyneu/covid19.git
cd covid19
```

Finally, run the Python script:

```bash
python code/run_model.py
```

The script will run the model and save plots into the [plots](plots) directory.

## Data source

The data is taken from Johns Hopkins University [Github's repository](
https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv).


## The unlicense

This work is in [public domain](LICENSE).

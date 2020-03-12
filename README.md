# Analysing time series of world wide confirmed cases of Coronavirus (COVID-19) with Stan and Python

This is a Stan model for analysing time series of confirmed cases of Coronavirus (COVID-19). The code creates a plot showing observed data (round blue markers) and the model (black line):

![Modelling COVID-19 confirmed cases with logistic function, observed cases](https://github.com/evgenyneu/covid19/raw/master/plots/2020_03_12_observed.png)

![Modelling COVID-19 confirmed cases with logistic function, extrapolated](https://github.com/evgenyneu/covid19/raw/master/plots/2020_03_12_extrapolated.png)


## Setup

Download and install all required software with the following steps.


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
cd covid19/code
```

Finally, run the Python script:

```bash
python run_mode.py
```

## Data source

The data is taken from [https://github.com/CSSEGISandData/COVID-19](
https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv).


## The unlicense

This work is in [public domain](LICENSE).

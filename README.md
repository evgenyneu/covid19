# Using logistic function to model number of COVID-19 confirmed cases in Stan and Python

This is a Stan model for analysing time series of confirmed cases of Coronavirus (COVID-19). It uses logistic function, which I described in [this blog post](https://evgenii.com/blog/logistic-function/).

### ⚠️⚠️⚠️WARNING ⚠️⚠️⚠️
This is a simple model that was done for educational purposes only. It is almost certainly **very inaccurate** and can not be used to predict the spread of the disease.

## How the model works

The program does the following:

* Downloads new data on confirmed COVID-19 cases from Johns Hopkins University [Github's repository](
https://github.com/CSSEGISandData/COVID-19).

* Runs Stan model that uses logistic function.

* Creates a plot showing observed data with round orange markers and the model with blue line. The bright shaded region indicates model's uncertainty and corresponds to 95% DHPI (highest posterior density interval).

As new data become available daily, I will be running this script and adding the new plots to [plots](plots) directory. This way we can see the changes in the data and the model with time.

![Modelling COVID-19 confirmed cases with logistic function, observed cases](https://github.com/evgenyneu/covid19/raw/master/plots/recent_observed.png)

### Extrapolating the data

**Attention**, the following plot was made for fun and is complete nonsense because it was bade with unrealistic model assumptions, which are described below. The plot also extrapolates the data way far into the future, which is silly.

![Modelling COVID-19 confirmed cases with logistic function, extrapolated](https://github.com/evgenyneu/covid19/raw/master/plots/recent_extrapolated.png)


## Assumptions

In the model we make the following assumptions:

* The number of cases can be modelled with logistic function.

* The virus is able to infect the entire population. We used a fixed value of 7.8 billion people for this parameter. However, it will be very interesting to make it a free parameter and let Stan estimate its distribution instead.

* All people who are infected with COVID-19 are reported.


## Exclusion of China

The model does not include the data for China because the number of confirmed cases have levelled off, as shown below. Assuming these data are correct, this could indicate that the spread of the disease is under control. Well done, China!

![Modelling COVID-19 confirmed cases in China with logistic function, observed cases](https://github.com/evgenyneu/covid19/raw/master/plots/2020_03_13_observed_china.png)


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

The data are taken from Johns Hopkins University [Github's repository](
https://github.com/CSSEGISandData/COVID-19/).


## Learning stats

If you want to learn more about Stan and statistical programming, I would highly recommend
[Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) textbook by Richard McElreath.


## The unlicense

This work is in [public domain](LICENSE).

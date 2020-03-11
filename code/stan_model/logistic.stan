data {
  int n;           // Number of data point
  real cases[n];   // Number of infected people at successive days
  real k;          // Population size

  // Logistic function parameter related to initial number of infected people
  real q;
}

parameters {
  real<lower=0> b;      // Growth rate
  real<lower=0> sigma;  // Spread of the cases
}

model {
  vector[n] mu;
  sigma ~ exponential(0.01);
  b ~ exponential(1);

  for (day in 1:n) {
      mu[day] = k / (1 + q * exp(-(b * (day - 1))) );
  }

  cases ~ normal(mu, sigma);
}

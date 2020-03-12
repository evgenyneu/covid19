/*
This is a statistical model that describes spread of infectious desease
through poplation using logistic growth model.
*/

// Data block describes known variables
data {
  int n;           // Number of data points
  real cases[n];   // Number of infected people at successive days
  real k;          // Maximum number people that can be infected

  // Logistic function parameter related to initial number of cases a=cases[1]
  // as follows: q = k/a - 1
  real q;
}

// Parameters block describes unknown variables we want to calculate
parameters {
  real<lower=0> r;      // Growth rate
  real<lower=0> sigma;  // Spread of cases
}

model {
  vector[n] mu; // Average number of infected people at each day
  sigma ~ exponential(0.01); // Prior for the spread
  r ~ exponential(1); // Prior for the growth rate

  // For each day, calculate the mu value using logistic function
  for (day in 1:n) {
      mu[day] = k / (1 + q * exp(-(r * (day - 1))) );
  }

  // We assume the number of infected people `cases` is normally distributed
  // with average `mu` and standard deviation `sigma`
  cases ~ normal(mu, sigma);
}

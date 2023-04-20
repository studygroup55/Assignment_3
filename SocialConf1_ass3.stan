
// Input data
data {
  int<lower=0> N;  #sample size
  array[N] int change; #difference between first and second rating
  array[N] real <lower= 0, upper = 1> FirstRating;
  array[N] real <lower= 0, upper = 1> OtherRating;
}


// Transformed data - unconstraining from 0-1 to log-odds scale
transformed data{
  array[N] real l_FirstRating;
  array[N] real l_SecondRating;
  l_FirstRating = logit(FirstRating);
  l_SecondRating = logit(OtherRating);
}

// Parameters
parameters {
  real bias;
}

//The model 
model {
  target +=  normal_lpdf(bias | 0, 1); # Prior for bias.  we use the normal distribution since our data is continuous at this point
  target +=  normal_lpdf(change | (bias + to_vector(l_FirstRating) + to_vector(l_OtherRating)) - to_vector(l_FirstRating);
}

// this is not yet finished
// generated quantities{
//   real bias_prior;
//   array[N] real log_lik;
//   
//   bias_prior = normal_rng(0, 1);
//   
//   for (n in 1:N){  
//     log_lik[n] = bernoulli_logit_lpmf(y[n] | bias + w*l_FirstRating[n] +  w*l_SecondRating[n]); # we must have the weight being on the right scale
//   }
//   
// }


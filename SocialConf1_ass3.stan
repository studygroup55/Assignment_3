
// Input data
data {
  int <lower = 0> N; //sample size
  array[N] int <lower = -7, upper = 7> Change; //difference between first and second rating
  array[N] int <lower= 1, upper = 8> FirstRating;
  array[N] int <lower= 1, upper = 8> GroupRating;
}


// Transformed data - unconstraining from 0-1 to log-odds scale
transformed data{ // Transforming data into prob scale
  array[N] real <lower = 0.1, upper = 0.9> FirstRating_transformed;
  array[N] real <lower = 0.1, upper = 0.9> GroupRating_transformed;
  array[N] real l_FirstRating;
  array[N] real l_GroupRating;
  
  // Here there is an issue. Consider changing something to vectors from the beginning
  FirstRating_transformed = (((to_vector(FirstRating) - 1)/7)*0.8) + 0.1;  // keeping it between 0.1 and 0.9
  GroupRating_transformed = (((to_vector(GroupRating) - 1)/7)*0.8) + 0.1; // keeping it between 0.1 and 0.9
  
  l_FirstRating = logit(FirstRating_transformed);
  l_GroupRating = logit(GroupRating_transformed);
}

// Parameters
parameters {
  real bias;
}


//The model 
model {
  target +=  normal_lpdf(bias | 0, 1); // Prior for bias.  we use the normal distribution since our data is continuous at this point
  target +=  normal_lpdf(Change | inv_logit(bias + to_vector(l_FirstRating) + to_vector(l_GroupRating)));
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


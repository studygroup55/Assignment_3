// Input data
data {
  int <lower = 0> N; //sample size
  vector [N] SecondRating; //difference between first and second rating <lower = -7, upper = 7> 
  vector [N] FirstRating;
  vector [N] GroupRating;
}


// Transformed data - unconstraining from 0-1 to log-odds scale
transformed data{ // Transforming data into prob scale
  vector[N] FirstRating_transformed;
  vector[N] GroupRating_transformed;
  vector[N] l_FirstRating;
  vector[N] l_GroupRating;
  
  // Here there is an issue. Consider changing something to vectors from the beginning
  for (trial in 1:N) { 
    FirstRating_transformed[trial] = (((FirstRating[trial] - 1)/7.0)*0.8) + 0.1;  // keeping it between 0.1 and 0.9
    GroupRating_transformed[trial] = (((GroupRating[trial] - 1)/7.0)*0.8) + 0.1; // keeping it between 0.1 and 0.9
  
    l_FirstRating[trial] = logit(FirstRating_transformed[trial]);
    l_GroupRating[trial] = logit(GroupRating_transformed[trial]);
  }
}

// Parameters
parameters {
  real bias;
}


//The model 
model {
  target +=  normal_lpdf(bias | 0, 0.1); // Prior for bias.  we use the normal distribution since our data is continuous at this point

  for (trial in 1:N) { 
    target +=  normal_lpdf(SecondRating[trial] | inv_logit(bias + 0.5 * l_FirstRating[trial] + 0.5 * l_GroupRating[trial]), 1);
  }
}


generated quantities{
   real bias_probability;
   real bias_prior;
   array[N] real log_lik;
   
   
   bias_probability = inv_logit(bias);
   
   bias_prior = normal_rng(0, 1);

   for (trial in 1:N){  
     log_lik[trial] = normal_lpdf(SecondRating[trial] | inv_logit(bias + 0.5*l_FirstRating[trial] +  0.5*l_GroupRating[trial]), 1); // we must have the weight being on the right scale
   }
}


# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

## Click Below To Browse Topics

[Home](index.md)  |  [Neural Networks](NeuralNet.md)  |  [Hierarchical Bayesian Analysis](HierarchicalBayes.md)  |      

## Bayesian Statistics
#### Note: Information on Bayesian statistics is an interpretation from my notes from Columbia University, STATGU4224 (2024) taught by Prof. Ronald Neath

Bayesian statistics are established on the principle of Bayes theorem, published by Thomas Bayes. Unlike frequentist approaches, Bayesian methods rely on incorporating prior knowledge with the observed data to determine the posterior probability of the event. Bayes rule can be represented as shown below:

![image](https://github.com/user-attachments/assets/47716545-63a4-4428-a632-7c901fcfcfdb)

Here, *p(θ&#124;y)* represents the posterior belief of *θ* conditional on the data *y*, *p(θ)* represents the prior belief about *θ* before the data is observed, *p(y&#124;θ)* is the likelihood which represents the probability of observing data *y* given *θ*, and *p(y)* is the marginal likelihood. The marginal likelihood is commonly represented as:

![image](https://github.com/user-attachments/assets/30ce3c68-e6f3-4935-a0b7-e64ab4206e35)

This acts as a normalizing constant for the posterior distribution. In the absence of this marginal likelihood, which is not always straightforward to determine, the posterior is often just calculated as being proportional to the product of the prior and the likelihood:

![image](https://github.com/user-attachments/assets/c3db2d67-918b-4b29-a049-05cdad5f8320)

Bayesian statistics allow us to use our beliefs about an event and update this belief as more data is observed. Moreover, it allows us to discuss the posterior in terms of probability of events occurring and can provide more interesting insights compared to frequentist approaches. 

### Modeling The Data
Herein, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team. We will attempt to do this based on the offense's ability to score a FG or touchdown when beginning (with a first down) between 60 and 70 yards from the endzone. This was chosen as this is the most common first down position we see from the offense. Here we can model the data in terms of attempts and successes where a single attempt is a Bernoulli distribution between 0 and 1 with a probability of success (p) where 0 < p < 1. The data of a single game has the distribution:

![image](https://github.com/user-attachments/assets/db9fcabd-1e90-4dbd-91d6-aa8d08ea02bb)

Without going too much into the math, the conjugate prior for a binomial sampling model is a beta distribution. Thus, when the prior distribution is *Beta(α, β)*, the posterior distribution is shown below where *y* is the number of successes and *n* is the number of attempts:

![image](https://github.com/user-attachments/assets/333f6527-1830-4808-a607-afacc2800dd5)


To model one game data, we need to choose a prior distribution, setting *α* and *β*. If we were to do this and we had no strong prior belief about the odds of success we would want to set an uninformative prior. For this scenario, a uniform distribution between 0 and 1 represents the uninformed prior. Thus our prior is *p(θ) = Beta(1,1)* and our posterior would be easily determined by:

![image](https://github.com/user-attachments/assets/c31d014b-2216-4ef5-9a9f-2153a36b6fd8)

Unfortunately, one game provides very little data to determine how successful the offense is. Moreover, we care about offensive success during the entire season. Thus, we want to consider the data from every game played in the 2024 season. 

## Hierarchical Bayesian Model
We have a few options to model the entire season. First, we could model each game individually, not pooling the data at all. This is problematic since we have so little data which makes determining the overall success of the offense during the season difficult. The second option we have is to pool all the data, but this is inappropriate since games have a great amount of variability, so all *θ<sub>j* cannot reasonably be expected to be equal. Instead, we can think of our data hierarchically and model it in a way where we acknowledge the connectedness of the data without pooling the data completely. A diagram of what the hierarchical model looks like is shown below:

![image](https://github.com/user-attachments/assets/4f77bf04-ddfb-40c5-9f87-3e2c2f5c2252)

We have already discussed that the data of a single game, *j*, has a Binomial distribution. In a hierarchical model, we will have the prior distribution:

![image](https://github.com/user-attachments/assets/0acecbee-c3f1-4188-91ef-d1e0fb1cfa58)

Thus, we have a hyperprior *p(α, β)* with a joint posterior distribution:

![image](https://github.com/user-attachments/assets/ddd59619-1b9e-4f44-8635-2a3b606adef9)

A reasonable diffuse prior can be set by using μ and ψ: 

![image](https://github.com/user-attachments/assets/c508436d-ca7d-4450-bb85-5f144fe6d16f)

In the original scale:

![image](https://github.com/user-attachments/assets/3daa6930-7e38-4394-9174-a4034655551c)

We can use Monte Carlo to sample from the discrete grid-based approximation of *p(α, β&#124;y)*  by drawing from:

![image](https://github.com/user-attachments/assets/5f40571d-9b3b-48de-be95-7e0570cf9e13)

The code adapted from STATGU4224, Columbia University (2024) for this full process in R is shown:
#### (Data was gathered from collegefootballdata.com)
```
y <- c(1, 1, 2, 0, 0, 1, 1, 1, 2, 0, 3, 2)
n <- c(7,  5, 11,  7,  4,  8,  8,  9,  6,  5, 12,  9)
m <- length(y)
S <- 5000

logit_mu <- seq(-2.5, 0.1, .01)
log_psi <- seq(-1, 12, .01) 
I <- length(logit_mu)
J <- length(log_psi)

log_post <- matrix(NA, I, J)
for(i in 1:I){for(j in 1:J){
  mu <- inv.logit(logit_mu[i]) 
  psi <- exp(log_psi[j])
  alpha <- mu*psi
  beta <- (1-mu)*psi
  log_post[i,j] <- (((-3 * log(alpha + beta))-m * lbeta(alpha, beta)) + sum(lbeta(alpha + y, beta + n - y)))+ log(alpha) + log(beta)
}}

log_post <- log_post - max(log_post)
post <- exp(log_post)
delta <- (logit_mu[2] - logit_mu[1]) / 2
epsilon <- (log_psi[2] - log_psi[1]) / 2
post_l_m <- apply(post, 1, sum) 
logit_mu_sim <- rep(NA, S) 
log_psi_sim <- rep(NA, S)
for(s in 1:S)
{
  i <- sample(I, 1, prob=post_l_m)
  j <- sample(J, 1, prob=post[i,])
  logit_mu_sim[s] <- logit_mu[i] + runif(1, -delta, delta)
  log_psi_sim[s] <- log_psi[j] + runif(1, -epsilon, epsilon)
}
```
Plotting this distribution with contours gives:
![image](https://github.com/user-attachments/assets/210bc8e0-d72c-4e03-b93e-3d2a7bb83735)


We then can perform 5000 draws from the above distribution to approximate the posterior for each game: 
```
mu_sim <- inv.logit(logit_mu_sim) 
psi_sim <- exp(log_psi_sim)
alpha_sim <- psi_sim * mu_sim 
beta_sim <- (1-mu_sim) * psi_sim
theta_sim <- rbeta(m * S, shape1 = outer(y, alpha_sim, "+"), shape2 = outer(n - y, beta_sim, "+"))
theta_sim <- matrix(theta_sim, m, S)
```
![image](https://github.com/user-attachments/assets/cf547e3c-2704-4baa-a79a-656565281c56)


We can see that the values are all somewhere between the observed probability of success for an individual game (no pooling) and the average probability of success from all games (complete pooling). This shrinking effect pulls probabilities towards the pooled average and away from the observed rates. We now want to consider the posterior probability of success if the 2024 team were to play one more game, based on the data from the entire season. We can simulate this with the following code:
```
theta_new <- rbeta(S, alpha_sim, beta_sim)
```
Then, plotting the posterior probability of success:
![image](https://github.com/user-attachments/assets/95f02b62-bf82-4a58-89d3-546d70b8d136)


## Analysis of offense improvement
Often, Bayesian statistics are used to make some statement about relative probabilities of an event occurring. For Nebraska football, an interesting comparison to make is between seasons to see if the team has improved from the previous year. Here, we will compare seasons from the Matt Rhule era (2023 and 2024). We can perform the exact analysis above on the 2023 data and plot the two distributions together as shown below:
![image](https://github.com/user-attachments/assets/eb6f34e1-29b2-4d84-8aaa-288f63e6e9e6)

From the distributions we can check our posterior belief that another game from the 2024 team will have a higher probability of success than another for the 2023 team:
```
sum(theta_new_2024>theta_new_2023)/S
```
From this we see a 71.9% posterior belief that the 2024 team would have a higher probability of success. 

## Analysis of defense improvement
We will perform the analysis as above but for the defense where the success is determined by the opposing team not scoring when starting between 60 and 70 yards from the endzone. Comparing the two posterior predictive distributions, the belief that the 2024 team would have a higher chance of success is 58.6%:

![image](https://github.com/user-attachments/assets/2661d4d3-c694-49de-925b-6e630bb2b255)


### Additional Thoughts
 - Many metrics for offensive and defensive success can exist so exploring other data may provide more valuable insights. 

 - This model avoids addressing confounding features such as weather, strength of opponent, previous games played, etc.... Including these variables in a hierarchical linear model may provide more detail on the differences between games and seasons.




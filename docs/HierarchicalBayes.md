# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

[Neural Networks](NeuralNet.md)  |  [Hierarchical Bayesian Analysis](HierarchicalBayes.md)  |  

## Bayesian Statistics
#### Note: Information on Bayesian statistics is an interpretation from my notes from Columbia University, STATGU4224 (2024) taught by Prof. Ronald Neath

Bayesian statistics is established on the principle of Bayes theorem, published by Thomas Bayes. Unlike frequentist approaches, Bayesian methods rely on incorporating prior knowledge with the observed data to determine the posterior probability of the event. Bayes rule can be represented as shown below:

$$ p(\theta|y) = {p(\theta)p(y|\theta)\over p(y)} $$

Here, $p(\theta|y)$ represents the posterior belief of $\theta$ conditional on the data $y$, $p(\theta)$ represents the prior belief about $$\theta$$ before the data is observed, $p(y|\theta)$ is the likelihood which represents the probability of observing data $y$ given $\theta$, and $p(y)$ is the marginal likelihood. The marginal likelihood is commonly represented as:

$$ p(y) = \int p(\theta)p(\theta|y)d\theta $$

This acts as a normalizing constant for the posterior distribution. In the absence of this marginal likelihood, which is not always straightforward to determine, the posterior is often just calculated as being proportional to the product of the prior and the likelihood:

$$ p(\theta|y) \propto {p(\theta)p(y|\theta)} $$

Bayesian statistics allow us to use our beliefs about an event and update this belief as more data is observed. Moreover, it allows us to discuss the posterior in terms of probability of events occuring and can provide more interesting insights compared to frequentist approaches. 

### Modeling The Data
Herein, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team. We will attempt to do this based on the offenses ability to score a FG or touchdown when beginning (with a first down) between 60 and 70 yards from the endzone. This was chosen as a touchback is placed on the 25 yard line and is the most common drive we should expect an offense to see. Here we can model the data in terms of attempts and successes where a single attmpt is a Bernoulli distribution between 0 and 1 with a probability of success (p) where 0 < p < 1. The data of a single game has the distribution:

$$ p(y|\theta) = Binomial(n, \theta) = {{n}\choose{y}} \theta^y(1-\theta)^{n-y} $$

Without going too much into the math, the conjugate prior for a binomial sampling model is a beta distribution. Thus, when the the prior distribution is $Beta(\alpha,\beta)$, the posterior distribution is:

$$ p(\theta|y) = Beta(\alpha +y, \beta +n-y) $$

Where $y$ is the number of successes and $n$ is the number of attempts. 

To model one game data, we need to choose a prior distribution, setting $\alpha and $\beta. We will approach this as if we have no strong prior belief about the odds of success and set an uninformative prior. For this scenario, a uniform distribution between 0 and 1 represents the uninformed prior. Thus our prior is $p(\theta) = Beta(1,1)$ and our posterior becomes:

$$ p(\theta|y) = Beta(1+y, 1+n-y) $$

Unfortunately, one game provides very little data to determine how successful the offense is. Moreover, we care about the offensive success during the entire season. Thus, we want to consider the data from every game played in the 2024 season. 

We have a few options to do this. First, we could model each game individually, not pooling the data at all. This is problematic since we have so little data and makes determining the overall success of the offense during the season difficult. The second option we have is to pool all of the data, but this is innapropriate since in each game has a great amount of variability, so all $\theta_j can not reasonably be expected to be equal. Instead, we can think of our data hierarchically and model it in a way where we aknowledge the connectedness of the data without pooling the data completly. 


## Hierarchical Bayesian Model
Here, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team with a heirarchical model. The data of a single game, $j$, has already been discussed to have the distribution:

$$ p(y_j|\theta_j) = Binomial(n_j, \theta_j) $$

In a hierarchical model, we will have the prior distribution:

$$ \theta_j|\alpha, \beta \approx Beta(\alpha, \beta) $$

Thus, we have a hyperprior $p(\alpha, \beta)$ with a joint posterior distribution:

$$ p(\theta,\alpha,\beta|y) \propto p(\alpha,\beta)p(\theta|\alpha,\beta)p(y|\alpha,\beta) $$

A reasonable diffuse prior can be set by letting $\mu = {\alpha \over \alpha + \beta}$ and $\psi = \alpha + \beta$ and then setting the hyperprior $p(\mu, \psi) = \propto psi^{-2}$ (cite). In the original scale:


$$ p(\alpha, \beta) \propto (\alpha + \beta)^{-3} $$  $$ \alpha + \beta >1 $$

with the log posterior:

$$ p(\alpha, \beta | y) = -3 * log(\alpha + \beta) - m*logBeta(\alpha, \beta) + \Sigma log Beta(\alpha + y_j, \beta + n_j - y_j) $$

where $m$ is the number of games played in the season. 

We can use Monte Carlo to sample from this distribution by generating samples from $(log(\alpha/\beta),log(\alpha + \beta))$. In R below:
```
code here
```
Which we can plot:








## Analysis of offense improvement
Talk about the data. (and lack of)

discuss complete pooling

discuss no pooling





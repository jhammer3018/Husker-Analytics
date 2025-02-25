# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

[Neural Networks](NeuralNet.md)  |  [Hierarchical Bayesian Analysis](HierarchicalBayes.md)  |  

## Bayesian Statistics
#### Note: Information on Bayesian statistics is an interpretation from my notes from Columbia University, STATGU4224 (2024) taught by Prof. Ronald Neath

Bayesian statistics is established on the principle of Bayes theorem, published by Thomas Bayes. Unlike frequentist approaches, Bayesian methods rely on incorporating prior knowledge with the observed data to determine the posterior probability of the event. Bayes rule can be represented as shown below:

$$ {p(\theta|y) = {p(\theta)p(y|\theta)\over p(y)}} $$

Here, $p(\theta|y)$ represents the posterior belief of $\theta$ conditional on the data $y$, $p(\theta)$ represents the prior belief about $${\theta}$$ before the data is observed, ${p(y|\theta)}$ is the likelihood which represents the probability of observing data ${y}$ given ${\theta}$, and ${p(y)}$ is the marginal likelihood. The marginal likelihood is commonly represented as:

$$ {p(y) = \int p(\theta)p(\theta|y)d\theta} $$

This acts as a normalizing constant for the posterior distribution. In the absence of this marginal likelihood, which is not always straightforward to determine, the posterior is often just calculated as being proportional to the product of the prior and the likelihood:

$$ {p(\theta|y) \propto {p(\theta)p(y|\theta)}} $$

Bayesian statistics allow us to use our beliefs about an event and update this belief as more data is observed. Moreover, it allows us to discuss the posterior in terms of probability of events occuring and can provide more interesting insights compared to frequentist approaches. 

### Modeling The Data
Herein, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team. We will attempt to do this based on the offenses ability to score a FG or touchdown when beginning (with a first down) between 60 and 70 yards from the endzone. This was chosen as a touchback is placed on the 25 yard line and is the most common drive we should expect an offense to see. Here we can model the data in terms of attempts and successes where a single attmpt is a Bernoulli distribution between 0 and 1 with a probability of success (p) where 0 < p < 1. The data of a single game has the distribution:

$$ {p(y|\theta) = Binomial(n, \theta) = {{n}\choose{y}} \theta^y(1-\theta)^{n-y}} $$

Without going too much into the math, the conjugate prior for a binomial sampling model is a beta distribution. Thus, when the the prior distribution is $Beta(\alpha,\beta)$, the posterior distribution is:

$$ {p(\theta|y) = Beta(\alpha +y, \beta +n-y)} $$

Where ${y}$ is the number of successes and ${n}$ is the number of attempts. 

To model one game data, we need to choose a prior distribution, setting ${\alpha}$ and ${\beta}$. If we were to do this and we had no strong prior belief about the odds of success we would want to set an uninformative prior. For this scenario, a uniform distribution between 0 and 1 represents the uninformed prior. Thus our prior is ${p(\theta) = Beta(1,1)}$ and our posterior would be easily determined by:

$$ {p(\theta|y) = Beta(1+y, 1+n-y)} $$

Unfortunately, one game provides very little data to determine how successful the offense is. Moreover, we care about the offensive success during the entire season. Thus, we want to consider the data from every game played in the 2024 season. 

We have a few options to do this. First, we could model each game individually, not pooling the data at all. This is problematic since we have so little data and makes determining the overall success of the offense during the season difficult. The second option we have is to pool all of the data, but this is innapropriate since in each game has a great amount of variability, so all ${\theta_j}$ can not reasonably be expected to be equal. Instead, we can think of our data hierarchically and model it in a way where we aknowledge the connectedness of the data without pooling the data completly. 


## Hierarchical Bayesian Model
Here, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team with a heirarchical model. A diagram of what the hierarchical model looks like is shown below:


We have already discussed that the data of a single game, ${j}$, has the distribution:

$$ {p(y_j|\theta_j) = Binomial(n_j, \theta_j)} $$

In a hierarchical model, we will have the prior distribution:

$$ {\theta_j|\alpha, \beta \approx Beta(\alpha, \beta)} $$

Thus, we have a hyperprior ${p(\alpha, \beta)}$ with a joint posterior distribution:

$$ {p(\theta,\alpha,\beta|y) \propto p(\alpha,\beta)p(\theta|\alpha,\beta)p(y|\alpha,\beta)} $$

We can't just use a uniform distribution to set this hyperprior. Instead a reasonable diffuse prior can be set by letting ${\mu = {\alpha \over \alpha + \beta}}$ and ${\psi = \alpha + \beta}$ and then setting the hyperprior ${p(\mu, \psi) \propto psi^{-2}}$ (cite). In the original scale:


$$ {p(\alpha, \beta) \propto (\alpha + \beta)^{-3}} $$  
$$ {\alpha + \beta >1} $$

We can use Monte Carlo to sample from the discrete grid-based approximation of $ {p(\alpha, \beta|y)} $ by instead drawing from ${p(log(\alpha/\beta), log(\alpha + \beta)|y) \propto \alpha \beta p(\alpha, \beta|y)}$, enabling to better sample from the distribution. Plotting this sampling below:
![image](https://github.com/user-attachments/assets/210bc8e0-d72c-4e03-b93e-3d2a7bb83735)


We then can draw S draws from the above distribution to approximate the posterior. When we do this our posterior probability for each game is:

![image](https://github.com/user-attachments/assets/cf547e3c-2704-4baa-a79a-656565281c56)


We can see that the values are all somewhere between the observed probability of success for an individual game (no pooling) and the average probability of succes (complete pooling). This effect of bringing the median probabiliity of success towards the middle is called .... where games with a smaller sample size are more affected. 

We now want to consider the posterior probability of success if the 2024 team were to play one more game, based on the data from the entire season. We can simulate this with the following code:
```
code here
```
We then can plot the posterior probability of success:
![image](https://github.com/user-attachments/assets/95f02b62-bf82-4a58-89d3-546d70b8d136)


## Analysis of offense improvement
Often times we want to use Bayesian statistics to compare distributions to make some statement about relative probabilities of an event occuring. For Nebraska football, an interesting comparison to make is between seasons to see if the team has improved from the previous year. Here, we will compare seasons from the Matt Rhule era (2023 and 2024). We can perform the exact analysis above on the 2023 data and plot the two distributions together as shown below:
![image](https://github.com/user-attachments/assets/eb6f34e1-29b2-4d84-8aaa-288f63e6e9e6)

From the distributions we can check our posterior belief that another game from the 2024 team will have a higher probability of success than another for the 2023 team:
```
code here
```
From this we see a 71.9% posterior belief that the 2024 team would have a higher probability of success. 
## Analysis of defense improvement


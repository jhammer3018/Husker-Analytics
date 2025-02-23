# Husker-Analytics
Machine learning for Nebraska football

This page is meant to merge passions: Nebraska football, statistics, and machine learning. This page is not affiliated with the University of Nebraska.

### Go Big Red!

[Neural Networks](NeuralNet.md)  |  [Bayesian Statistics](HierarchicalBayes.md)  |  

## Bayesian Statistics

Bayesian statistics is established on the princible of Bayes theorem, published by Thomas Bayes in 1763. Unlike frequentist approaches, Bayesian methods rely on incorporating prior knowledge with the observed data to determine the posterior probability of the event. Bayes rule can be represented as shown below:

$$ p(\theta|y) = {p(\theta)*p(y|\theta)\over p(y)} $$

Here, $p(\theta|y)$ represents the posterior belief of $\theta$ conditional on the data $y$, $p(\theta)$ represents the prior belief about $$\theta$$ before the data is observed, $p(y|\theta)$ is the likelihood which represents the probability of observing data $y$ given $\theta$, and $p(y)$ is the marginal likelihood. The marginal likelihood is commonly represented as:

$$ p(y) = \int p(\theta)p(\theta|y)d\theta $$

This acts as a normalizing constant for the posterior distribution. In the absence of this marginal likelihood, which is not always straightforward to determine, the posterior is often just calculated as:

$$ p(\theta|y) \propto {p(\theta)*p(y|\theta)} $$

where the posterior is proportional to the product of the prior and the likelihood. 

Bayesian statistics allow us to use our beliefs about an event and update this belief as more data is observed. Moreover, it allows us to discuss the posterior in terms of probability of events occuring and can provide more interesting insights compared to frequentist approaches. 

## Hierarchical Bayesian Model
Hierarchical Bayesian models use Bayesian statistics to determine our posterior belief about hierarchical data. Commonly, non-hierarchical models are innaproppriate when data is connected but does not share all of the same conditions. For example, when modeling all offensive game data for a single team, it is easy to understand why we might not want to pool all of the data since multiple different opponents are faced over the course of the season. Instead we can think of this data hierarchically where there is the commonality of a single team on offense but differences in the defense that is being faced. 

This is the exact scenario that we will attempt to model. Here, we will determine our posterior belief of offensive "success" for the 2024 Nebraska football team. We will attempt to do this based on the offenses ability to score a FG or touchdown when beginning (with a first down) between 60 and 70 yards from the endzone. This was chosen as a touchback is placed on the 25 yard line and is the most common drive we should expect an offense to see. Here we can model the data in terms of attempts and successes where a single attmpt is a Bernoulli distribution between 0 and 1 with a probability of success (p) where 0 < p < 1. The data of a single game, $j$, can be modeled as:

$$ y \sim Binomial(nj, \theta j) $$



## Analysis of offense improvement
Talk about the data. (and lack of)

discuss complete pooling

discuss no pooling





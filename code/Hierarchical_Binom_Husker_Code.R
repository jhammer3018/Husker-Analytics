# Adapted by Joseph Hammer from Columbia University, STATGU4224 (2024) example code written by Prof. Ronald Neath
# Data collected from collegefootballdata.com

##2024 offense
#y <- c(3, 2, 4, 2, 3, 1, 1, 2, 0, 2, 2, 0, 2)
#n <- c(11, 11, 12,  8,  7,  4,  6,  7,  6, 11, 11,  4,  6)

## 2023 offense
#y <- c(1, 1, 2, 0, 0, 1, 1, 1, 2, 0, 3, 2)
#n <- c(7,  5, 11,  7,  4,  8,  8,  9,  6,  5, 12,  9)


##2024 defense
#y <- c(9, 7, 1, 5, 7, 5, 6, 8, 1, 3, 1, 5, 8)
#n <- c(11,  7,  2,  6,  8,  5,  6, 10,  2,  3,  2,  5,  9)

##2023 defense
y <- c(6, 6, 6, 3, 1, 9, 8, 9, 8, 8, 7, 1)
n <- c(8,  8,  6,  4,  5, 10,  9, 12,  8,  9,  7,  2)

#number of samples
S <- 5000
m <- length(y)

## These seq parameters need to be set via trial and error to get all data on the resulting contour plot
## The correct range is dependent on the observed data
##2024 offense
#logit_mu <- seq(-2.5, 0.1, .01)
#log_psi <- seq(-1, 12, .01) 

##2023 offense
#logit_mu <- seq(-3, 0.1, .01)
#log_psi <- seq(-1, 12, .01) 

##2024 defense
#logit_mu <- seq(-0.5, 3.5, .01)
#log_psi <- seq(-2, 11, .01) 

#2023 defense
logit_mu <- seq(-0.7, 3, .01)
log_psi <- seq(-2, 9, .01) 

I <- length(logit_mu)
J <- length(log_psi)

# Generating the posterior distributions on the data grid
log_post <- matrix(NA, I, J)
for(i in 1:I){ for(j in 1:J){
  mu <- inv.logit(logit_mu[i]) 
  psi <- exp(log_psi[j])
  alpha <- mu*psi
  beta <- (1-mu)*psi
  l_post <- ((( -3 * log(alpha + beta))-m * lbeta(alpha, beta)) + sum(lbeta(alpha+y, beta+n-y))) + log(alpha) + log(beta)
  log_post[i,j] <- l_post 
}}

# Sampling S values from the posterior
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

# Plotting the contours from alpha and beta sampling
contours <- c(.001, .01, seq(.05, .95, .10))
contour(logit_mu, log_psi, post, levels=contours, drawlabels=F,
        xlab=expression("log( "*alpha~"/ "*beta~")"), ylab=expression("log( "*alpha~"+ "*beta~")"), col = 'black')
points(logit_mu_sim, log_psi_sim, cex = 0.7,  col = 'red') 
grid(FALSE)


# Determining posterior probability of success
# Converting to alpha and beta
mu_sim <- inv.logit(logit_mu_sim) 
psi_sim <- exp(log_psi_sim)
alpha_sim <- psi_sim * mu_sim 
beta_sim <- (1-mu_sim) * psi_sim

#Simulating theta for each game
theta_sim <- rbeta(m*S, shape1=outer(y, alpha_sim, "+"),
                   shape2=outer(n-y, beta_sim, "+") )
theta_sim <- matrix(theta_sim, m, S)


#Plotting the values as a function of observation rate with 95% CI
medians <- apply(theta_sim, 1, median) #medians of theta
percentile <- apply(theta_sim, 1, quantile, prob=c(.025, .975)) #grabbing 95%
plot(y/n, medians, xlim=range(0,1), ylim=range(0,1),
     xlab="Observed Rate of Success", ylab="Posterior Probabilty of Success", col = 'black')
for(j in 1:m){ lines(rep((y/n)[j],2), percentile[,j]) }

#Plotting comparative solutions for complete and no pooling
#No pooling
abline(0, 1, lty=2, col = 'red')
text(0.38, 0.4, "No Pooling", srt = 30)

#Complete Pooling
abline(h=sum(y)/sum(n), lty=2, col = 'red')
text(0.4, 0.17, "Complete Pooling")




#Generating the probability of success if another game were played
theta_new <- rbeta(S, alpha_sim, beta_sim)
hist(theta_new, breaks=100, col = 'red', freq = FALSE,  
     xlab="Posterior Probabilty of Success" , xlim=range(0,1))
lines(density(theta_new, adj=2), col="black", lwd=2)
quantile(theta_new, c(.025, .975))


##saving distributions
#theta_new_2024_offense <- theta_new
#theta_new_2023_offense <- theta_new
#theta_new_2024_defense <- theta_new
#theta_new_2023_defense <- theta_new

#Plotting both distributions



#Comparing distributions
hist24 <- hist(theta_new_2024_defense, breaks=100)
hist23 <- hist(theta_new_2023_defense, breaks=100)


plot(hist24, col=rgb(1,0,0,1/4), xlim=c(0,1), freq = FALSE, xlab = 'Posterior Probability of Success', ylim = range(0,10))  # first histogram
plot(hist23, col=rgb(0,0,0,1/4), xlim=c(0,1),freq = FALSE, add=T)
legend("topright", c("2024", "2023"), col=c(rgb(1,0,0,1/4), rgb(0,0,0,1/4)), lwd=10)

lines(density(theta_new_2024_defense, adj=2), col="red", lwd=2)
lines(density(theta_new_2023_defense, adj=2), col="black", lwd=2)

#Quantifying p(theta1>theta2|y)
sum(theta_new_2024_defense>theta_new_2023_defense)/S
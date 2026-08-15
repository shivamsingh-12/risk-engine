Probabilty notes (all that are referenced/used in this project):
Sample Space: The sample space for a given set of events is the set of all possible values the events may assume. May also be known as a event space or possibility space.
    For example: the sample space of a toss of two coins, each of which may land heads or tails is the set of all possible outcomes: HH, TH, TH, TT.

Conditional Probability: Measures the probability that a certain event will occur based on the occurence or non-occurence of other related events. Researchers examine two or more events with related probabilities and ask "If we know A has happened, what is the chance of B also happening?"
    Formula: P(A|B) = P(A and B) / P(B)
        P(A and B) is the chance of both events
        P(B) is the chance of the given event
        P(B) must be greater than 0.

Bayes Theorem: Mathematical formula used to update the probability of a hypothesis as more evidence or information becomes available. It calculates how likely an event is given a specific condition, which then allows to flip a known conditional probability around.
    Formula: P(A|B) = (P(B|A) * P(A)) / P(B)
    P(A|B): Likelihood of event A being true after considering B
    P(B|A): Probability of seeing B assuming A is true

Random variable: A variable whose possible values are numerical outcomes of a random phenomenon. There are two types of random variables, discrete and continuous.
    Discrete: May take only a countable number of distinct values, usually counts, finite = discrete. Example: number of children in a family.
    Continuous: Takes an infinite number of possible values. Usually measurements. Example: height.

PMF (Probability Mass Function): Used for discrete data, gives the exact probability that a random variable equals a certain value, the sum is always 1.

PDF (Probability Density Function): Used for continuous data, gives the density of probability at a point, the probability of finding the variable inside a range is found by calculating the area under the curve, the total area under the curve is equal to 1.

CDF (Cumulative Distribution Function): Works for both discrete and continuous data, gives the probability that a random variable is less than or equal to a specific value P(X <= x), starts at 0 and climbs to a maximum value of 1, found by summing PMF or integrating PDF.

Expectation: Long term average value of a random variable if you repeat an experiment many times. Calculated by multiplying each possible outcome by its probability and adding those values together, acting as the center point of a probability distribution.
    Discrete: If a variable can take specific seperate values (like numbers on a die), you multiply each value by its chance of happening and sum them.
    Continuous: If a variable can take any value within a range (like exact time or weight), you use integration over a probability density function
        Discrete Formula: E(X) = sigma(x) * P(X = x)
        Continuous Formula: E(X) = integral(x) * f(x)dx (f(x) is the density function)
*The expectation of a fixed number is c is just c itself*

Variance: A statistical measure that tells you how spread out a set of numbers is from the mean. Lower = closer to mean, higher = further
    Formula: S^2 = (sigma(xi - x))^2 / n - 1

Covariance: A statistical measure that shows how two variables change together. Positive = both variables increase/decrease together, negative = move in opposite directions
    Sample covariance and formula: Used when working with a subset or sample of data from larger group.
     \(\text{Cov}(X,Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n - 1}\)
    Population covariance and formula: Used when dataset includes every member
     \(\text{Cov}(X,Y) = \frac{\sum (x_i - \mu_x)(y_i - \mu_y)}{N}\)

Independence: Two events are independent if the outcome of one does not change the chance of the other happening.

Law of Large Numbers: States that as you repeat a random test more and more times, the average of all results gets closer to the true expected value. As short term results can swing wildly, long term results become stable and predictable.

Central Limit Theorem: States that when you take sufficiently large samples from a population, the distribution of the sample means will closely approximate a normal or bell-shaped distribution, regardless of the shape of the original population's distribution.

Bernoulli trial: Single experiment consisting of either success(p) or failure(1-p). Mean: p, Variance: p(1-p)

Binomial Distribution: Models multiple independent trials, counts total successes(k). Parameters are 
number of trials n and success probability p, Mean: np, Variance: np(1-p).

Bernoulli Distribution: Simply a binomial distribution where n=1.

Uniform Distribution: A probability function where every outcome or interval within a specific range has an equal and constant chance of occuring.

Normal Distribution: A symmetric, bell-shaped curve most data points cluster near the central average.

Exponential Distribution: Models the time or space between idependent events in a Poisson process.
    Poisson process: A model for tracking random events that happen independently over time or space at a steady, average rate.
    Formula: P(X <= x) = 1 - e^-(rate parameter * x)
        Rate parameter: The average number of events per unit of time or space

Poisson Distribution: Calculates the probability of a specific number of independent events occuring in a fixed interval of time or space.
    Formula: P(X = x) = (lamba^x * e^(-lambda)) / x!
    lambda: The known average number of occurences in the given interval.
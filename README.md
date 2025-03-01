# Project Overview
This project addresses the optimization problem of a discrete-time stochastic process by minimizing its variance and maximizing its expected value across different sliding windows and budget constraints. The goal is to assign optimal coefficients to an n-dimensional stationary stochastic process to achieve the desired statistical properties. 

# Problem Description
Given an n-dimensional stationary stochastic process $X(t)$, we aim to find the optimal coefficients $\alpha_i$ such that:<br>
$Y(t) = \sum_{i=1}^n \alpha_i X_i(t)$<br>
The optimization is performed in two main steps:<br>
    1- Minimizing the variance of $𝑌(𝑡)$ while considering a set of constraints.<br>
    2- Maximizing the expected value of $𝑌(𝑡)$, taking into account the minimized variance 
  from the first step along with the initial constraints.

# Solutionn Approach
The optimization is solved using the Gurobi library in Python through two constrained optimization problems:<br>
    1- The first optimization reduces variance.<br>
    2- The second optimization maximizes the expectation based on the results of the first step.

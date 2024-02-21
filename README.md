# Copula

In this repository I deepened my knowledge about probability theory and the foundationas of various Copulas. Copulas are a flexible way to construct dependence strucutres and honestly look quite cool. Since C++ does not come with a lot of probabilistic functions, I had to implement some by my own. I wanted to make the overall project to a .dll in order to later use it in different projects with the namespace 'copula'. 

In detail I implemented the Gaussian copula, which is defined as:

$C_R (u_1,u_2) = \Phi_R(\Phi^{-1}(u_1),\Phi^{-1}(u_2))$

as well as the Frank copula

$\frac{-1.0}{\alpha} log \biggl[1.0 + exp(-sum) * (exp(-\alpha) - 1))$

## To-Do's

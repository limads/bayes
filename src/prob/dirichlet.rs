
/*
Dir(alpha) generates probability vectors in the simplex via the density:
(gamma( sum_i(alpha_i) ) / prod_i(gamma(alpha_i))) * prod_k \theta_k^{alpha_k - 1}
*/

// From Robert (Bayesian Essentials with R)
// To generate random dirichlet variates:
/*rdirichlet=function(n=1,par=rep(1,2)){
k=length(par)
mat=matrix(0,n,k)
for (i in 1:n){
sim=rgamma(k,shape=par,scale=1)
mat[i,]=sim/sum(sim)
}
mat
}*/


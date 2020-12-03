#include <iostream>
#include "mcmc.hpp"

typedef struct {
	void* model;
	double (*lp_func)(void*,double*,size_t);
} DistrPtr;

double vec_log_prob(const arma::vec& param, void* distr) {
	DistrPtr* dp = (DistrPtr*) distr;
	double lp = dp->lp_func(dp->model, (double*) param.memptr(), param.n_rows);
	return lp;
}

extern "C" {

	// Starts the Random Walk metropolis hastings algorithm. The initial values is a vector
	// of size p of parameter values from which the chain departs. If succesful, the samples
	// of dimension n (rows) x p (columns) are copied into the out double precision array, 
	// assumed to be pre-allocated by the caller and having at least n*p double precision entries.
	bool distr_mcmc(double* init_vals, double* out, size_t n, size_t p, size_t burn, DistrPtr* distr) {
		arma::vec vec_init_vals = arma::vec(init_vals, p, 1);
		
		mcmc::algo_settings_t settings;
		settings.rwmh_n_draws = n;
		settings.rwmh_n_burnin = burn;
		
		arma::mat mat_draws_out;
		bool result = mcmc::rwmh(vec_init_vals, mat_draws_out, vec_log_prob, (void*) distr, settings);
		
		// std::cout << "Samples: " << mat_draws_out << std::endl;
		// std::cout << "Samples nrows: " << mat_draws_out.n_rows << " Samples ncols: " << mat_draws_out.n_cols << std::endl;
		
		if(result) {
			memcpy( (void*) out, (void*) mat_draws_out.memptr(), sizeof(double)*n*p );
		}
		
		return result;
	}

}





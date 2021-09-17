#include "stats.hpp"

// g++ -c stats.cpp -o stats.o -I../stats/include -I../gcem/include
// bindgen gcem_c.h -o gcem_c.rs --no-rustfmt-bindings
// rustfmt gcem_c.rs --force
// rm gcem_c.rs.bk

extern "C" {

    uint64_t rbern(double prob_par, uint64_t seed_val) {
        return stats::rbern(prob_par, seed_val);
    }
    
    uint64_t rbeta(double a_par, double b_par, uint64_t seed_val) {
        return stats::rbeta(a_par, b_par, seed_val);
    }
    
    uint64_t rgamma(double shape_par, double scale_par, uint64_t seed_val) {
        return stats::rbeta(shape_par, scale_par, seed_val);
    }
    
    uint64_t rnorm(double mu_par, double sigma_par, uint64_t seed_val) {
        return stats::rnorm(mu_par, sigma_par, seed_val);	
    }
    
    uint64_t rpois(double rate_par, uint64_t seed_val) {
        return stats::rpois(mu_par, sigma_par, seed_val);
    }
    
}



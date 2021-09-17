#include "gcem.hpp"
// #include "gcem_c.h"

// g++ -c gcem_c.cpp -o gcem_c.o -I../gcem/include
// bindgen gcem_c.h -o gcem_c.rs --no-rustfmt-bindings
// rustfmt gcem_c.rs --force
// rm gcem_c.rs.bk

extern "C" {

    double binomial_coef(double n, double k) {
        return gcem::binomial_coef(n, k);
    }

    double log_binomial_coef(double n, double k) {
        return gcem::log_binomial_coef(n, k);
    }

    double beta(double a, double b) {
        return gcem::beta(a, b);
    }

    double lbeta(double a, double b) {
        return gcem::lbeta(a, b);
    }

    double tgamma(double x) {
        return gcem::tgamma(x);
    }

    double lgamma(double x) {
        return gcem::lgamma(x);
    }

    double lmgamma(double a, double p) {
        return gcem::lmgamma(a, p);
    }

    double erf(double x) {
        return gcem::erf(x);
    }

    double incomplete_beta(double a, double b, double z) {
        return gcem::incomplete_beta(a, b, z);
    }

    double incomplete_gamma(double a, double x) {
        return gcem::incomplete_gamma(a, x);
    }

    double erf_inv(double p) {
        return gcem::erf_inv(p);
    }

    double incomplete_beta_inv(double a, double b, double p) {
        return gcem::incomplete_beta_inv(a, b, p);
    }

    double incomplete_gamma_inv(double a, double p) {
        return gcem::incomplete_gamma_inv(a, p);
    }

}

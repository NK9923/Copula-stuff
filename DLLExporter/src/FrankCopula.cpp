#include "Header.h"
#include "pch.h"

// Frank Copula implementation

namespace copula {

    inline void FrankCopula::PrintInfo() {
        std::cout << "== Frank Copula ===" << std::endl;
        std::cout << "Dimension: " << dim << std::endl;
        std::cout << "Alpha: " << alpha << std::endl;
    }

    // CDF of the Frank copula
    inline double FrankCopula::cdfExpr(const std::vector<double>& u, int dim) {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            sum += -log((exp(-alpha * u[i]) - 1) / (exp(-alpha) - 1));
        }
        return -1.0 / alpha * log(1.0 + exp(-sum) * (exp(-alpha) - 1));
    };

    // PDF of the Frank copula
    inline double FrankCopula::pdfExpr(const std::vector<double>& u) {
        double term1 = -log((exp(-alpha * u[0]) - 1) / (exp(-alpha) - 1)) +
            -log((exp(-alpha * u[1]) - 1) / (exp(-alpha) - 1));

        double term2 = exp(-term1) * (exp(-alpha * u[1]) * alpha / (exp(-alpha) - 1) /
            ((exp(-alpha * u[1]) - 1) / (exp(-alpha) - 1))) *
            (exp(-alpha * u[0]) * alpha / (exp(-alpha) - 1) /
                ((exp(-alpha * u[0]) - 1) / (exp(-alpha) - 1))) *
            (exp(-alpha) - 1) / (1 + exp(-term1) * (exp(-alpha) - 1));

        double term3 = exp(-term1) * (exp(-alpha * u[0]) * alpha / (exp(-alpha) - 1) /
            ((exp(-alpha * u[0]) - 1) / (exp(-alpha) - 1))) *
            (exp(-alpha) - 1) *
            exp(-term1) * (exp(-alpha * u[1]) * alpha / (exp(-alpha) - 1) /
                ((exp(-alpha * u[1]) - 1) / (exp(-alpha) - 1))) *
            (exp(-alpha) - 1) / (1 + exp(-term1) * (exp(-alpha) - 1));

        return -1.0 / alpha * (term2 - term3);
    };

    // Function to generate random samples from a multivariate Frank copula
    std::pair<std::vector<double>, std::vector<double>> FrankCopula::rfrankCopula(int n) {
        if (dim == 2) {
            return rfrankBivCopula(n);
        }

        std::vector<double> samples_U;
        std::vector<double> samples_V;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        if (StatsFunctions().rlogseries_ln1p(2).first == 0) {
            return std::make_pair(std::vector<double>{1}, std::vector<double>{1});
        }

        // Check for conditions and handle them accordingly
        if (std::abs(alpha) < std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3)) {
            std::cerr << "Alpha was chosen to be too small" << std::endl;
            assert(false);
        }
        else {
            // Generate samples using the log-series distribution and inverse psi function
            // ...
        }

        return std::make_pair(samples_U, samples_V);
    }

    inline std::pair<std::vector<double>, std::vector<double>> FrankCopula::rfrankBivCopula(int n) {
        std::vector<double> U_samples;
        std::vector<double> V_samples;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double a = -std::abs(alpha);

        // Generate samples
        for (int i = 0; i < n; ++i) {
            double U = dist(gen);
            double V = dist(gen);

            // Inversion of the Frank copula's CDF
            V = -1 / a * log1p(-V * expm1(-a) / (exp(-a * U) * (V - 1) - V));

            U_samples.push_back(U);
            V_samples.push_back((alpha > 0) ? 1 - V : V);
        }

        return std::make_pair(U_samples, V_samples);
    }

    std::pair<std::vector<double>, std::vector<double>> FrankCopula::Frank_paretoMarginals(int n, double Theta, double Alpha1, double Alpha2, double Gamma1, double Gamma2) {
        assert(n >= 1 && n == round(n) && "Sample size n must be greater than or equal to 1 (integer)");
        assert(Theta != 0 && "Theta cannot be zero");
        assert(Alpha1 > 0 && "Alpha1 must be positive");
        assert(Alpha2 > 0 && "Alpha2 must be positive");
        assert(Gamma1 > 0 && "Gamma1 must be positive");
        assert(Gamma2 > 0 && "Gamma2 must be positive");

        std::vector<double> X_values;
        std::vector<double> Y_values;

        for (int i = 0; i < n; ++i) {
            double U = ((double)rand() / RAND_MAX);
            double a = ((double)rand() / RAND_MAX);
            double V = (-1 / Theta) * log(1 + a * (exp(-Theta) - 1) / (exp(-Theta * U) - a * (exp(-Theta * U) - 1)));
            double X = (1 / Alpha1) * (pow(U, -1 / Gamma1) - 1);
            double Y = (1 / Alpha2) * (pow(V, -1 / Gamma2) - 1);

            X_values.push_back(X);
            Y_values.push_back(Y);
        }

        return std::make_pair(X_values, Y_values);
    }
}
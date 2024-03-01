#include "Header.h"
#include "pch.h"

namespace copula {

    template <typename T>
    struct is_vector : std::false_type {};

    template <typename T>
    struct is_vector<std::vector<T>> : std::true_type {};

    template <typename T>
    T generate_uniform(int N_sim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        if constexpr (is_vector<T>::value) {
            std::vector<typename T::value_type> randomValues(N_sim);
            for (int i = 0; i < N_sim; ++i) {
                randomValues[i] = dis(gen);
            }
            return randomValues;
        }
        else {
            return dis(gen);
        }
    }

    template <typename T>
    T runif(int N_sim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        if constexpr (is_vector<T>::value) {
            std::vector<typename T::value_type> randomValues(N_sim);
            for (int i = 0; i < N_sim; ++i) {
                randomValues[i] = dis(gen);
            }
            return randomValues;
        }
        else {
            return dis(gen);
        }
    }

    std::function<double(double)> CopulaSampling::getQuantileFunction(MarginalType type, const std::vector<double>& parameters) {
        switch (type) {
        case NORMAL:
            return std::bind(StatsFunctions::norm_q, std::placeholders::_1, parameters[0], parameters[1]);
        case UNIFORM:
            return std::bind(StatsFunctions::unif_q, std::placeholders::_1, parameters[0], parameters[1]);
        case GAMMA:
            return std::bind(StatsFunctions::gamma_q, std::placeholders::_1, parameters[0], parameters[1], 1e-6, 1000);
        case EXPONENTIAL:
            return std::bind(StatsFunctions::exp_q, std::placeholders::_1, parameters[0]);
        case BETA:
            return std::bind(StatsFunctions::beta_q, std::placeholders::_1, parameters[0], parameters[1], 1e-6, 1000);
        case UNKNOWN:
            throw std::invalid_argument("Unsupported marginal distribution type");
        }
    }

    Eigen::MatrixXd CopulaSampling::rmvt_samples(int n, double df, double sigma, double mean) {
        assert(df >= 0);

        Eigen::MatrixXd standard_normal = GaussCopula().rmvnorm_samples(n, mean, sigma);
        auto chi_squared_vector = StatsFunctions().generate_chi_squared(n, df);
        Eigen::VectorXd Chi_squared = Eigen::Map<const Eigen::VectorXd>(chi_squared_vector.data(), chi_squared_vector.size());

        assert(Chi_squared.minCoeff() > 0);

        if (this->debug) {
            GaussCopula().printMatrix(standard_normal, "Standard Normal");
            GaussCopula().printMatrix(Chi_squared, "Chi Squared");
            GaussCopula().printMatrix(Chi_squared / df, "Chi Squared / df");
            GaussCopula().printMatrix((1.0 / (Chi_squared / df).array().sqrt()).matrix(), "(1.0 / (Chi_squared / df).array().sqrt()).matrix()");
            GaussCopula().printMatrix(standard_normal.array().topRows(5), "standard_normal.array()");
            GaussCopula().printMatrix((1.0 / (Chi_squared / df).array().sqrt()).matrix().array().replicate(1, standard_normal.cols()).topRows(5), "Multiply");

        }

        Eigen::MatrixXd random_t = standard_normal.array() * ((1.0 / (Chi_squared / df).array().sqrt()).matrix().array().replicate(1, standard_normal.cols()));

        if (this->debug) {
            GaussCopula().printMatrix(random_t.topRows(5), "random_t");
        }

        random_t.colwise() += Eigen::VectorXd::Constant(n, mean);
        return random_t;
    }

    std::pair<std::vector<double>, std::vector<double>> CopulaSampling::rCopula(int n, const CopulaInfo& copula, const MarginalInfo& marginals) {
        std::vector<double> u(n);
        std::vector<double> v(n);

        if (copula.type == "independent") {
            for (int i = 0; i < n; ++i) {
                u[i] = runif<double>(1);
                v[i] = runif<double>(1);
            }
        }
        else if (copula.type == "normal") {
            Eigen::MatrixXd samples = GaussCopula().rmvnorm_samples(n, copula.parameters[0], copula.parameters[1]);
            for (int i = 0; i < n; ++i) {
                u[i] = StatsFunctions::norm_cdf(samples(i, 0));
                v[i] = StatsFunctions::norm_cdf(samples(i, 1));
            }
        }
        else if (copula.type == "t") {
            Eigen::MatrixXd samples = CopulaSampling().rmvt_samples(n, copula.parameters[0], copula.parameters[1], copula.parameters[2]);
            for (int i = 0; i < n; ++i) {
                u[i] = StatsFunctions::t_pdf(samples(i, 0), copula.parameters[0]);
                v[i] = StatsFunctions::t_pdf(samples(i, 1), copula.parameters[0]);
            }
        }
        else if (copula.type == "clayton") {
            for (int i = 0; i < n; ++i) {
                double random1 = runif<double>(1); // u
                double random2 = runif<double>(1); // v

                u[i] = random1;
                v[i] = std::pow(std::pow(random1, -copula.parameters[0]) + std::pow(random2, (-copula.parameters[0] / (copula.parameters[0] + 1))) - 1.0, -1.0 / copula.parameters[0]);
            }
        }
        else if (copula.type == "gumbel") {
            // Need to be checked
            for (int i = 0; i < n; ++i) {
                double random1 = runif<double>(1); // u
                double random2 = runif<double>(1); // v

                u[i] = std::pow(-std::log(random1), 1.0 / random2);
                v[i] = u[i] * (-std::log(random1));
            }
        }
        else if (copula.type == "frank") {
            for (int i = 0; i < n; ++i) {
                double random1 = runif<double>(1); // u
                double random2 = runif<double>(1); // v

                double a = -abs(copula.parameters[0]);

                double tmp = -1 / a * std::log1p(-random2 * std::expm1(-a) / (std::exp(-a * random1) * (random2 - 1) - random2));
                u[i] = random1;
                v[i] = (copula.parameters[0] > 0) ? (1 - tmp) : tmp;
            }
        }
        else {
            throw std::invalid_argument("Invalid copula type.");
        }

        std::function<double(double)> qdfExpr1 = CopulaSampling().getQuantileFunction(marginals.type1, marginals.params1.parameters);
        std::function<double(double)> qdfExpr2 = CopulaSampling().getQuantileFunction(marginals.type2, marginals.params2.parameters);

        for (int i = 0; i < n; ++i) {
            u[i] = qdfExpr1(u[i]);
            v[i] = qdfExpr2(v[i]);
        }

        std::pair<std::vector<double>, std::vector<double>> result_copula;
        result_copula.first = u;
        result_copula.second = v;
        return result_copula;
    }
}
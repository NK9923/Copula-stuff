#pragma once

#ifdef EXPORT
#define dllimportexport dllexport
#else
#define dllimportexport dllimport
#endif

#include <iostream>
#include <vector>
#include <optional>
#include <string>
#include <map>
#include <cassert>
#include <algorithm>
#include <Eigen/Dense>

namespace copula {

    __declspec(dllimportexport) void getwd();

    class __declspec(dllimportexport) NumericalDifferentiation {
        public:
            static std::vector<double> gradient(double (*f)(const std::vector<double>&), const std::vector<double>& x0, double heps = 1e-9);
            
            template<typename VectorType>
            VectorType gradient(const std::function<double(const VectorType&)>& f, const VectorType& x0, double heps);
    };
    
	class __declspec(dllimportexport) ECDF {
        public:
            ECDF(const std::vector<double>& data);
            const std::vector<double>& getSortedData() const;
            void plotECDF();

            double operator()(double x) const;
        private:
            std::vector<double> sorted_data;
            std::vector<double> ecdf_values;
    };

    class __declspec(dllimportexport) StatsFunctions {
        public:
            // Core Stats Functions
            std::map<std::string, double> fitMoments(const std::vector<double>& data);
            inline double calculateShape(double locHat, double sig2Hat) const;
            inline double calculateScale(double locHat, double shapeHat) const;
            inline double mean(const std::vector<double>& data) const;
            inline double variance(const std::vector<double>& data) const;
            inline double skewness(const std::vector<double>& data) const;
            inline double kurtosis(const std::vector<double>& data) const;

            static std::pair<double, bool> rlogseries_ln1p(double a, double cutoff = log(2));
            inline static float erfinv(float x);

            template <typename T1, typename T2>
            static inline typename T1::value_type Quantile(const T1& x, T2 q);

            // Random Number Generation
            std::vector<double> generate_uniform(int N_sim);
            std::vector<double> generate_gaussian(int N_sim, double mean, double stddev);
            std::vector<double> generate_pareto(int N, double g, double k);
            std::vector<double> generate_cauchy(int N, double location, double scale);
            std::vector<double> generate_beta(int N, double alpha, double beta);

            // Normal distribution
            inline static double pnorm(double value);
            inline static double qnorm(double p, double mean = 0, double sigma = 1);
            inline static double qunif(double p, double a = 0.0, double b = 1.0);

            //Student's t distribution
            inline static double trapezoidal(double a, double b, int n, std::function<double(double)> f);
            inline static double pdf_t(double x, int df);
            inline static double cdf_t(double x, int df);
            inline static double q_t(double p, int df, double tol = 1e-6, int max_iter = 1000);
            std::vector<double> rt(int n, int df);

            // Plot Distribution
            static void plotDistribution(std::vector<double>& data);
    };

    class __declspec(dllimportexport) GaussCopula {
        public:
            GaussCopula(bool Debug = false) : debug(Debug) {};
            using QuantileFunction = std::function<double(double)>;
        
            Eigen::MatrixXd rmvnorm_samples(int n, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma);
            std::pair<std::vector<double>, std::vector<double>> rGaussCopula(int N_sim, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma, QuantileFunction f1, QuantileFunction f2);
            
            void PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, double cor);
            void printMatrix(const Eigen::MatrixXd& matrix, const std::string& name);
        private:
            bool debug;
    };

    class __declspec(dllimportexport) FrankCopula {
        public:
            FrankCopula(double alpha = 2.0, int dim = 2) : alpha(alpha), dim(dim) {
                if (dim > 2) {
                    std::cerr << "Only implemented for dimension 2. Higher dimensions require dynamic differentiation" << std::endl;
                    assert(false);
                }
                this->PrintInfo();
            }
            inline void PrintInfo();
            inline double cdfExpr(const std::vector<double>& u, int dim);
            inline double pdfExpr(const std::vector<double>& u);

            std::pair<std::vector<double>, std::vector<double>> rfrankCopula(int n);
            std::pair<std::vector<double>, std::vector<double>> Frank_paretoMarginals(int n, double Theta, double Alpha1, double Alpha2, double Gamma1, double Gamma2);
            void PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data);

        private:
            double alpha;
            int dim;

            inline std::pair<std::vector<double>, std::vector<double>> rfrankBivCopula(int n);
    };

    class __declspec(dllimportexport) EVTCopula {
        private:
            struct __declspec(dllimportexport) GPDResult {
                std::vector<double> excesses;
                double shape;
                double scale;
                double threshold;
            };
        public:
            std::vector<EVTCopula::GPDResult> f_FitGPD(const std::vector<std::vector<double>>& data, std::optional<double> lower = std::nullopt,
                std::optional<double> upper = std::nullopt, int min_obs = 150, std::string method = "MLE", bool lower_tail = false,
                bool double_tail = false);

            std::vector<double> f_FastpSPGPD(const std::vector<double>& data, EVTCopula::GPDResult& fit);

            std::vector<std::vector<double>> f_CopulasEmpirical(const std::vector<std::vector<double>>& data, std::vector<EVTCopula::GPDResult>& fit);
            double f_TailDep(const std::vector<std::vector<double >>& data, double threshold);
    };
}

template <typename T1, typename T2>
static inline typename T1::value_type copula::StatsFunctions::Quantile(const T1& x, T2 q) {
    assert(q >= 0.0 && q <= 1.0);

    using ValueType = typename T1::value_type;
    std::vector<ValueType> data(std::begin(x), std::end(x));
    data.erase(std::remove_if(data.begin(), data.end(), [](ValueType val) { return std::isnan(val); }), data.end());
    std::sort(data.begin(), data.end());

    const auto n = data.size();
    const auto id = static_cast<typename T1::size_type>((n - 1) * q);
    const auto lo = static_cast<typename T1::size_type>(std::floor(id));
    const auto hi = static_cast<typename T1::size_type>(std::ceil(id));
    const auto qs = data[lo];
    const auto h = id - lo;

    return (1.0 - h) * qs + h * data[hi];
}
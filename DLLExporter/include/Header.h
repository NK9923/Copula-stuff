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

            inline static double trapezoidal(double a, double b, int n, std::function<double(double)> f);

            // Random Number Generation
            std::vector<double> generate_uniform(int N_sim);
            std::vector<double> generate_gaussian(int N_sim, double mean, double stddev);
            std::vector<double> generate_pareto(int N, double g, double k);
            std::vector<double> generate_chi_squared(int N, double df);
            std::vector<double> generate_cauchy(int N, double location, double scale);
            std::vector<double> generate_beta(int N, double alpha, double beta);
            std::vector<double> generate_t(int N, int df);

            // Normal distribution
            inline static double norm_cdf(double value);
            inline static double norm_pdf(double value);
            inline static double norm_q(double p, double mean = 0, double sigma = 1);

            // Uniform distribution
            inline static double unif_pdf(double x, double a = 0.0, double b = 1.0);
            inline static double unif_cdf(double x, double a = 0.0, double b = 1.0);
            inline static double unif_q(double p, double a = 0.0, double b = 1.0);

            // Student's t distribution            
            inline static double t_pdf(double x, int df);
            inline static double t_cdf(double x, int df);
            inline static double t_q(double p, int df, double tol = 1e-6, int max_iter = 1000);

            // Beta distribution
            inline static double beta_pdf(double x, double shape1, double shape2);
            inline static double beta_cdf(double x, double shape1, double shape2);
            inline static double beta_q(double p, double shape1, double shape2, double tol = 1e-6, int max_iter = 1000);

            // Gamma distribution
            inline static double gamma_pdf(double x, double shape, double scale);
            inline static double gamma_cdf(double x, double shape, double scale);
            inline static double gamma_q(double p, double shape, double scale, double tol = 1e-6, int max_iter = 1000);

            // Exponential distribution
            inline static double exp_pdf(double x, double rate);
            inline static double exp_cdf(double x, double rate);
            inline static double exp_q(double p, double rate);

            // Plot Distribution
            static void plotDistribution(std::vector<double>& data);
    };

    class __declspec(dllimportexport) GaussCopula {
        public:
            GaussCopula(bool Debug = false) : debug(Debug) {};
            using QuantileFunction = std::function<double(double)>;
        
            Eigen::MatrixXd rmvnorm_samples(int n, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma);
            Eigen::MatrixXd rmvnorm_samples(int n, const double Mean, const double& Sigma);
            std::pair<std::vector<double>, std::vector<double>> rGaussCopula(int N_sim, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma, QuantileFunction f1, QuantileFunction f2);
            
            void PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, double cor);
            void printMatrix(const Eigen::MatrixXd& matrix, const std::string& name, int rowsToPrint = 5);
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

    class __declspec(dllimportexport) CopulaSampling {
        private:
            enum MarginalType {
                NORMAL,
                UNIFORM,
                GAMMA,
                EXPONENTIAL,
                BETA,
                UNKNOWN
            };

            struct DistributionParams {
                std::vector<double> parameters;
            };

            MarginalType getMarginalType(const std::string& type) {
                if (type == "normal") {
                    return NORMAL;
                }
                else if (type == "uniform") {
                    return UNIFORM;
                }
                else if (type == "gamma") {
                    return GAMMA;
                }
                else if (type == "exponential") {
                    return EXPONENTIAL;
                }
                else if (type == "beta") {
                    return BETA;
                }
                else {
                    return UNKNOWN;
                }
            }
        
            std::function<double(double)> getQuantileFunction(MarginalType type, const std::vector<double>& parameters);
            bool debug = false; // change if EigenOutput is needed to be visible for debugging purpose

        public:
            struct __declspec(dllimportexport) MarginalInfo {
                MarginalType type1;
                MarginalType type2;
                DistributionParams params1;
                DistributionParams params2;

                MarginalInfo(MarginalType t1, MarginalType t2, const std::vector<double>& p1, const std::vector<double>& p2)
                    : type1(t1), type2(t2), params1({ p1 }), params2({ p2 }) {}
            };

            struct __declspec(dllimportexport) CopulaInfo {
                std::string type;
                std::vector<double> parameters;
            };

            MarginalInfo getMarginalInfo(const std::string& type1, const std::vector<double>& parameters1,
                const std::string& type2, const std::vector<double>& parameters2) {
                if ((type1 == "normal" || type1 == "uniform" || type1 == "gamma" || type1 == "exponential" || type1 == "beta") &&
                    (type2 == "normal" || type2 == "uniform" || type2 == "gamma" || type2 == "exponential" || type2 == "beta")) {
                    return { getMarginalType(type1), getMarginalType(type2), { parameters1 }, { parameters2 } };
                }
                else {
                    return { UNKNOWN, UNKNOWN, {}, {} };
                }
            }

            Eigen::MatrixXd rmvt_samples(int n, double df, double sigma, double mean = 0);
            static std::pair<std::vector<double>, std::vector<double>> rCopula(int n, const CopulaInfo& copula, const MarginalInfo& marginals);
            void PlotRCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, std::string CopulaTyp, std::string mar1, std::string mar2, std::string filename);
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
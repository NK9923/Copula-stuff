#pragma once

#define EXPORT

#ifdef EXPORT
#define dllimportexport dllexport
#else
#define dllimportexport dllimport
#endif

#include<vector>
#include<optional>
#include<string>
#include<map>
#include<cassert>
#include<algorithm>

namespace copula {
    struct __declspec(dllimportexport) GPDResult {
        std::vector<double> excesses;
        double shape;
        double scale;
        double threshold;
    };

    
    class __declspec(dllimportexport) differentiation {
        public:
            static std::vector<double> gradient(double (*f)(const std::vector<double>&), const std::vector<double>& x0, double heps = 1e-9);
    };
    

	class __declspec(dllimportexport) ECDF {
        public:
            ECDF(const std::vector<double>& data);
            double operator()(double x) const;
            double head_ecdf(int min_obs) const;

        private:
            std::vector<double> sorted_data;
            std::vector<double> ecdf_values;
    };

    class __declspec(dllimportexport) StatsFunctions {
        public:
            std::map<std::string, double> fitMoments(const std::vector<double>& data);
            inline double calculateShape(double locHat, double sig2Hat) const;
            inline double calculateScale(double locHat, double shapeHat) const;
            inline double mean(const std::vector<double>& data) const;
            inline double variance(const std::vector<double>& data) const;
            inline double skewness(const std::vector<double>& data) const;
            inline double kurtosis(const std::vector<double>& data) const;
            static std::pair<double, bool> rlogseries_ln1p(double a, double cutoff = log(2));

            template <typename T1, typename T2>
            static inline typename T1::value_type Quantile(const T1& x, T2 q);

            std::vector<double> generate_uniform(int N_sim);
            std::vector<double> generate_gaussian(int N_sim, double mean, double stddev);
            std::vector<double> generate_pareto(int N, double g, double k);
            std::vector<double> generate_cauchy(int N, double location, double scale);
            std::vector<double> generate_beta(int N, double alpha, double beta);

            static void plotDistribution(std::vector<double>& data);
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
        void PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data);

    private:
        double alpha;
        int dim;

        inline std::pair<std::vector<double>, std::vector<double>> rfrankBivCopula(int n);
    };


	__declspec(dllimportexport) GPDResult fit_GPD_PWM(const std::vector<double>& data);
	__declspec(dllimportexport) GPDResult fit_GPD_MOM(const std::vector<double>& data);
    __declspec(dllimportexport) GPDResult gpd_fit(const std::vector<double>& data, std::optional<double> lower, std::optional<double> upper, int min_obs, std::string method, bool lower_tail, bool double_tail);
    __declspec(dllimportexport) void getwd();
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
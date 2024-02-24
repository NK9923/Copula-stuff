#include "Header.h"
#include "pch.h"

namespace copula {

    std::map<std::string, double> StatsFunctions::fitMoments(const std::vector<double>& data) {
        std::map<std::string, double> result;

        double locHat = this->mean(data);
        double sig2Hat = this->variance(data);

        result["shape"] = this->calculateShape(locHat, sig2Hat);
        result["scale"] = this->calculateScale(locHat, result["shape"]);
        result["loc"] = locHat;

        return result;
    }

    inline double StatsFunctions::calculateShape(double locHat, double sig2Hat) const {
        return (1 - std::pow(locHat, 2) / sig2Hat) / 2;
    }

    inline double StatsFunctions::calculateScale(double locHat, double shapeHat) const {
        return max(locHat * (1 - shapeHat), std::numeric_limits<double>::epsilon());
    }

    inline double StatsFunctions::mean(const std::vector<double>& data) const {
        #if _MSVC_LANG >= 202002L
            if (data.empty()) {
                return 0.0;
            }
            return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        #else
            if (data.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (const double& value : data) {
                sum += value;
            }
            return sum / data.size();
        #endif 
    }

    inline double StatsFunctions::variance(const std::vector<double>& data) const {
        #if _MSVC_LANG >= 202002L
            if (data.empty()) {
                return 0.0;
            }

            double meanValue = this->mean(data);
            return std::accumulate(data.begin(), data.end(), 0.0,
                [meanValue](double accumulator, double value) {
                    double difference = value - meanValue;
                    return accumulator + difference * difference;
                }) / data.size();
        #else
            if (data.empty()) {
                return 0.0;
            }
            double meanValue = this->mean(data);
            double sumSquaredDifferences = 0.0;
            for (const double& value : data) {
                double difference = value - meanValue;
                sumSquaredDifferences += difference * difference;
            }
            return sumSquaredDifferences / data.size();
        #endif 
    }

    inline double StatsFunctions::skewness(const std::vector<double>& data) const {
        double meanValue = this->mean(data);
        double varianceValue = std::sqrt(variance(data));

        double skewness = 0.0;
        for (const double& value : data) {
            double deviation = value - meanValue;
            skewness += std::pow(deviation / varianceValue, 3);
        }
        skewness /= data.size();
        return skewness;
    }

    inline double StatsFunctions::kurtosis(const std::vector<double>& data) const {
        double meanValue = this->mean(data);
        double varianceValue = std::sqrt(variance(data));

        double kurtosis = 0.0;
        for (const double& value : data) {
            double deviation = value - meanValue;
            kurtosis += std::pow(deviation, 4);
        }
        kurtosis /= (data.size() * std::pow(varianceValue, 2));

        return kurtosis;
    }

    std::vector<double> StatsFunctions::generate_uniform(int N_sim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        std::vector<double> randomValues;

        for (int i = 0; i < N_sim; ++i) {
            randomValues.push_back(dis(gen));
        }
        return randomValues;
    }

    std::vector<double> StatsFunctions::generate_gaussian(int N_sim, double mean, double stddev) {

        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> norm(mean, stddev);
        std::vector<double> randomValues;


        for (int i = 0; i < N_sim; ++i) {
            randomValues.push_back(norm(gen));
        }
        return randomValues;
    }

    std::vector<double> StatsFunctions::generate_pareto(int N, double g, double k) {
        if (k <= 0 || g <= 0) {
            throw std::invalid_argument("Both k and g should be greater than 0.");
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        std::vector<double> randomValues;

        for (int i = 0; i < N; ++i) {
            randomValues.push_back(k * std::pow(1 - dis(gen), -1 / g));
        }

        return randomValues;
    }

    std::vector<double> StatsFunctions::generate_cauchy(int N, double location, double scale) {
        std::mt19937 gen(std::random_device{}());
        std::cauchy_distribution<double> dis(location, scale);

        std::vector<double> randomValues;

        for (int i = 0; i < N; ++i) {
            randomValues.push_back(dis(gen));
        }

        return randomValues;
    }

    std::vector<double> StatsFunctions::generate_beta(int N, double alpha, double beta) {
        if (alpha <= 0.0 || beta <= 0.0) {
            throw std::invalid_argument("Both alpha and beta should be greater than 0.");
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        std::vector<double> randomValues;

        for (int i = 0; i < N; ++i) {
            double u1 = dis(gen);
            double u2 = dis(gen);

            double betaValue = std::pow(u1, 1.0 / alpha);
            double invBeta = std::pow(1.0 / u2, 1.0 / beta);

            randomValues.push_back(betaValue / (betaValue + invBeta));
        }

        return randomValues;
    }

    std::pair<double, bool> StatsFunctions::rlogseries_ln1p(double a, double cutoff) {

        if (a <= 0) {
            std::cout << "Warning a needs to be bigger than 0";
            return std::make_pair(0.0, false);
        }

        double val1, val2;
        try {
            val1 = std::log(-std::expm1(-a));

            if (std::isnan(val1)) {
                throw std::invalid_argument("val1 is NaN");
            }
        }
        catch (const std::exception& e) {
            std::cout << "Exception caught while calculating val1: " << e.what() << std::endl;
            val1 = 0;
        }

        try {
            val2 = std::log1p(-std::exp(-a));

            if (std::isnan(val2)) {
                std::cout << "Warning: val2 is not a number" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cout << "Exception caught while calculating val2: " << e.what() << std::endl;
            val2 = 0;
        }
        return std::make_pair((val1 != 0.0) ? val1 : val2, (val1 != 0.0) || (val2 != 0.0));
    }

    float StatsFunctions::erfinv(float x) {
        float tt1, tt2, lnx, sgn;
        sgn = (x < 0) ? -1.0f : 1.0f;

        x = (1 - x) * (1 + x);
        lnx = logf(x);

        tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
        tt2 = 1 / (0.147) * lnx;

        return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
    }

    double StatsFunctions::pnorm(double value) {
        return 0.5 * erfc(-value * sqrt(0.5));
    }

    double StatsFunctions::qnorm(double p, double mean, double sigma) {
        return(mean + sigma * sqrt(2) * erfinv(2 * p - 1));
    }

    double StatsFunctions::qunif(double p, double a, double b) {
        if (p < 0) p = 0;
        if (p > 1) p = 1;
        return a + (b - a) * p;
    }

    // Empirical CDF function 
    ECDF::ECDF(const std::vector<double>& data) {
        sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());

        std::vector<double>::iterator it = std::unique(sorted_data.begin(), sorted_data.end());
        sorted_data.resize(std::distance(sorted_data.begin(), it));

        size_t n = sorted_data.size();
        ecdf_values.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ecdf_values[i] = static_cast<double>(i + 1) / n;
        }
    }

    const std::vector<double>& ECDF::getSortedData() const {
        return sorted_data;
    }

    double ECDF::operator()(double x) const {
        auto it = std::lower_bound(sorted_data.begin(), sorted_data.end(), x);
        if (it == sorted_data.end()) {
            return 1.0;
        }
        else {
            size_t index = std::distance(sorted_data.begin(), it);
            return (index + 1) / static_cast<double>(sorted_data.size());
        }
    }
}
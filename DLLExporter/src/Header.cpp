#include "pch.h"
#include "Header.h"

class StatsFunctions {
    public:
        std::map<std::string, double> fitMoments(const std::vector<double>& data) {
            std::map<std::string, double> result;

            double locHat = mean(data);
            double sig2Hat = variance(data);

            result["shape"] = calculateShape(locHat, sig2Hat);
            result["scale"] = calculateScale(locHat, result["shape"]);
            result["loc"] = locHat;

            return result;
        }

        inline double calculateShape(double locHat, double sig2Hat) const {
            return (1 - std::pow(locHat, 2) / sig2Hat) / 2;
        }

        inline double calculateScale(double locHat, double shapeHat) const {
            return max(locHat * (1 - shapeHat), std::numeric_limits<double>::epsilon());
        }

        inline double mean(const std::vector<double>& data) const {
            if (data.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (const double& value : data) {
                sum += value;
            }
            return sum / data.size();
        }

        inline double variance(const std::vector<double>& data) const {
            if (data.empty()) {
                return 0.0;
            }
            double meanValue = mean(data);
            double sumSquaredDifferences = 0.0;
            for (const double& value : data) {
                double difference = value - meanValue;
                sumSquaredDifferences += difference * difference;
            }
            return sumSquaredDifferences / data.size();
        }

        inline double skewness(const std::vector<double>& data) const {
            double meanValue = mean(data);
            double varianceValue = std::sqrt(variance(data));

            double skewness = 0.0;
            for (const double& value : data) {
                double deviation = value - meanValue;
                skewness += std::pow(deviation / varianceValue, 3);
            }
            skewness /= data.size();
            return skewness;
        }

        inline double kurtosis(const std::vector<double>& data) const {
            double meanValue = mean(data);
            double varianceValue = std::sqrt(variance(data));

            double kurtosis = 0.0;
            for (const double& value : data) {
                double deviation = value - meanValue;
                kurtosis += std::pow(deviation, 4);
            }
            kurtosis /= (data.size() * std::pow(varianceValue, 2));

            return kurtosis;
        }

        template <typename T1, typename T2>
        static inline typename T1::value_type Quantile(const T1& x, T2 q) {
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

        std::vector<double> generate_runif(int N_sim) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            std::vector<double> randomValues;

            for (int i = 0; i < N_sim; ++i) {
                randomValues.push_back(dis(gen));
            }
            return randomValues;
        }    

        std::vector<double> generate_gaussian(int N_sim, double mean, double stddev) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dis(mean, stddev);
            std::vector<double> randomValues;

            for (int i = 0; i < N_sim; ++i) {
                randomValues.push_back(dis(gen));
            }
            return randomValues;
        }

        std::vector<double> generate_pareto(int N, double g, double k) {
            if (k <= 0 || g <= 0) {
                throw std::invalid_argument("Both k and g should be greater than 0.");
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(0.0, 1.0);

            std::vector<double> randomValues;

            for (int i = 0; i < N; ++i) {
                randomValues.push_back(k * std::pow(1 - dis(gen), -1 / g));
            }

            return randomValues;
        }

        std::vector<double> generate_cauchy(int N, double location, double scale) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::cauchy_distribution<double> dis(location, scale);

            std::vector<double> randomValues;

            for (int i = 0; i < N; ++i) {
                randomValues.push_back(dis(gen));
            }

            return randomValues;
        }

        std::vector<double> generate_beta(int N, double alpha, double beta) {
            if (alpha <= 0.0 || beta <= 0.0) {
                throw std::invalid_argument("Both alpha and beta should be greater than 0.");
            }

            std::random_device rd;
            std::mt19937 gen(rd());
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
};

class ECDF {
    public:
        ECDF(const std::vector<double>& data) {
            sorted_data = data;
            std::sort(sorted_data.begin(), sorted_data.end());

            std::vector<double>::iterator it = std::unique(sorted_data.begin(), sorted_data.end());
            sorted_data.resize(std::distance(sorted_data.begin(), it));

            // Berechne die ECDF-Werte
            size_t n = sorted_data.size();
            ecdf_values.resize(n);
            for (size_t i = 0; i < n; ++i) {
                ecdf_values[i] = static_cast<double>(i + 1) / n;
            }
        }

        double operator()(double x) const {
            auto it = std::lower_bound(sorted_data.begin(), sorted_data.end(), x);
            if (it == sorted_data.end()) {
                return 1.0;
            }
            else {
                size_t index = std::distance(sorted_data.begin(), it);
                return ecdf_values[index];
            }
        }

        inline double head_ecdf(int min_obs) const {
            if (min_obs <= 0) {
                return 0.0;
            }
            min_obs = min(min_obs, static_cast<int>(sorted_data.size()));
            return ecdf_values[min_obs - 1];
        }
    private:
        std::vector<double> sorted_data;
        std::vector<double> ecdf_values;
};


copula::GPDResult copula::fit_GPD_PWM(const std::vector<double>& data) {
    GPDResult result;
    return GPDResult();
}

copula::GPDResult copula::fit_GPD_MOM(const std::vector<double>& data) {
    GPDResult result;
    return GPDResult();
}

copula::GPDResult copula::gpd_fit(const std::vector<double>& data, std::optional<double> lower = std::nullopt, std::optional<double> upper = std::nullopt, int min_obs = 150, std::string method = "MLE", bool lower_tail = false, bool double_tail = false) {
    copula::GPDResult result;
    assert(min_obs >= 150);

    if (lower.has_value()) {
        std::cout << "Lower value provided: " << lower.value() << std::endl;
    }
    else {
        std::cout << "Lower value not provided, setting to false." << std::endl;
        lower = false;
    }

    ECDF ecdf(data);
    double lower_threshold = ecdf.head_ecdf(min_obs);

    //std::map<std::string, double> result = StatsFunctions().fitMoments(data);

    if (upper.has_value()) {
        std::cout << "Upper value provided: " << upper.value() << std::endl;
    }
    else {
        std::cout << "Upper value not provided, setting to false." << std::endl;
        upper = false;
    }
    return result;
}

#include "Header.h"
#include "pch.h"

namespace copula {

    // StatsFunctions core

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

    // Random number generation functions

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

    std::vector<double> StatsFunctions::generate_chi_squared(int N, double df) {
        std::random_device rd{};
        std::mt19937 gen{ rd() };
        std::chi_squared_distribution<double> dist{ df };

        std::vector<double> randomValues;

        for (int i = 0; i < N; ++i) {
            randomValues.push_back(dist(gen));
        }

        return randomValues;
    }

    std::vector<double> StatsFunctions::generate_beta(int N, double alpha, double beta) {
        std::mt19937 engine(std::random_device{}());

        std::gamma_distribution<double> x_gamma(alpha);
        std::gamma_distribution<double> y_gamma(beta);

        std::vector<double> randomValues;

        for (int i = 0; i < N; ++i) {
            double x = x_gamma(engine);
            randomValues.push_back(x / (x + y_gamma(engine)));
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

    // Normal distribution

    float StatsFunctions::erfinv(float x) {
        float tt1, tt2, lnx, sgn;
        sgn = (x < 0) ? -1.0f : 1.0f;

        x = (1 - x) * (1 + x);
        lnx = logf(x);

        tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
        tt2 = 1 / (0.147) * lnx;

        return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
    }

    double StatsFunctions::norm_pdf(double value) {
        return 0.5 * erfc(-value * sqrt(0.5));
    }

    double StatsFunctions::norm_cdf(double value) {
		return 0.5 * erfc(-value * sqrt(0.5));
	}

    double StatsFunctions::norm_q(double p, double mean, double sigma) {
        return (mean + sigma * sqrt(2) * erfinv(2 * p - 1));
    }

    // Uniform distribution

    double StatsFunctions::unif_pdf(double x, double a, double b) {
		if (x < a || x > b) {
			return 0;
		}
		return 1 / (b - a);
	}

    double StatsFunctions::unif_cdf(double x, double a, double b) {
		if (x < a) {
			return 0;
		}
		if (x > b) {
			return 1;
		}
		return (x - a) / (b - a);
	}

    double StatsFunctions::unif_q(double p, double a, double b) {
        if (p < 0) p = 0;
        if (p > 1) p = 1;
        return a + (b - a) * p;
    }

    // Trapezoidal rule for numerical integration

    double StatsFunctions::trapezoidal(double a, double b, int n, std::function<double(double)> f) {
        double dx = (b - a) / n;
        double integral = f(a) + f(b);
        for (int i = 1; i <= n - 1; i++) {
            integral += 2.0 * f(a + i * dx);
        }
        integral *= dx / 2.0;
        return integral;
    }

    // Student's t distribution

    double StatsFunctions::t_pdf(double x, int df) {
        double pi = 4.0 * atan(1.0);
        double gamma_term = tgamma(0.5 * (df + 1.0)) / tgamma(0.5 * df) / sqrt(df * pi);
        double expression_term = pow(1.0 + (x * x / df), -0.5 * (df + 1.0));
        return gamma_term * expression_term;
    }

    double StatsFunctions::t_cdf(double x, int df) {
        double pi = 4.0 * atan(1.0);
        std::function<double(double)> pdf_function = [df](double val) { return t_pdf(val, df); };
        double integral = trapezoidal(0.0, x, 1000, pdf_function);
        return 0.5 + (x > 0 ? integral : -integral);
    }

    double StatsFunctions::t_q(double p, int df, double tol, int max_iter) {
        double x = 0.0;
        for (int i = 0; i < max_iter; i++) {
            double cdf = t_cdf(x, df);
            double pdf = t_pdf(x, df);
            if (std::abs(cdf - p) < tol) {
                break;
            }
            else {
                x -= (cdf - p) / pdf;
            }
        }
        return x;
    }

    std::vector<double> StatsFunctions::generate_t(int N, int df){
        std::random_device rd{};
        std::mt19937 gen{ rd() };

        std::normal_distribution<double> standard_normal_distribution(0.0, 1.0);
        std::chi_squared_distribution<double> chi_squared_distribution(df);

        std::vector<double> t_random_values;
        t_random_values.reserve(N);

        for (int i = 0; i < N; ++i) {
            double standard_normal = standard_normal_distribution(gen);
            double chi_square = chi_squared_distribution(gen);
            double t_random = standard_normal / sqrt(chi_square / df);
            t_random_values.push_back(t_random);
        }

        return t_random_values;
    }

    // Gamma distribution

    double StatsFunctions::gamma_pdf(double x, double shape, double scale) {
		if (x < 0) {
			return 0;
		}
		if (shape <= 0 || scale <= 0) {
			return 0;
		}
		return pow(x, shape - 1) * exp(-x / scale) / (tgamma(shape) * pow(scale, shape));
	}

    double StatsFunctions::gamma_cdf(double x, double shape, double scale) {
        if (x < 0 || shape <= 0 || scale <= 0) {
            return 0;
        }

        auto integrand = [shape, scale](double t) {
            return std::pow(t, shape - 1) * std::exp(-t / scale);
        };

        const int n = 1000;
        double result = 0.0;
        double dx = x / n;

        for (int i = 0; i < n; ++i) {
            double x_i = i * dx;
            double x_i_1 = (i + 1) * dx;
            result += 0.5 * (integrand(x_i) + integrand(x_i_1)) * dx;
        }

        return result / std::tgamma(shape);
    }

    // ToFix still an error
    double StatsFunctions::gamma_q(double p, double shape, double scale, double tol, int max_iter) {
        if (p < 0.0 || p > 1.0 || shape <= 0.0 || scale <= 0.0) {
            throw std::invalid_argument("Invalid parameters for gamma distribution quantile function.");
        }

        double lower = 0.0;
        double upper = shape * scale;

        int iter = 0;
        while (upper - lower > tol && iter < max_iter) {
            double x = (lower + upper) / 2;
            double cdf_val = gamma_cdf(x, shape, scale);

            if (std::abs(cdf_val - p) < tol) {
                return x;
            }

            if (cdf_val < p) {
                lower = x;
            }
            else {
                upper = x;
            }
            iter++;
        }
        return (lower + upper) / 2;
    }

    // Exponential distribution

    double StatsFunctions::exp_pdf(double x, double lambda) {
		if (x < 0 || lambda <= 0) {
			return 0;
		}
		return lambda * exp(-lambda * x);
	}

    double StatsFunctions::exp_cdf(double x, double lambda) {
        if (x < 0 || lambda <= 0) {
            return 0;
        }
        return 1 - exp(-lambda * x);
    }

    double StatsFunctions::exp_q(double p, double lambda) {
		return -log(1 - p) / lambda;
	}

    // Beta-distribution with location and scale

    double beta(double x, double y) {
        return tgamma(x) * tgamma(y) / tgamma(x + y);
    }

    double StatsFunctions::beta_pdf(double x, double shape1, double shape2) {
        assert(x >= 0.0 && x <= 1.0);
        assert(shape1 > 0.0 && shape2 > 0.0);
        return pow(x, shape1 - 1) * pow(1 - x, shape2 - 1) / beta(shape1, shape2);
    }

    double StatsFunctions::beta_cdf(double x, double shape1, double shape2) {
        assert(x >= 0.0 && x <= 1.0);
        assert(shape1 > 0.0 && shape2 > 0.0);

        double beta_term = beta(shape1, shape2);

        std::function<double(double)> integrand_function = [shape1, shape2](double t) {
            return pow(t, shape1 - 1) * pow(1 - t, shape2 - 1);
            };

        double integral = trapezoidal(0.0, x, 10000, integrand_function);
        return integral / beta_term;
    }

    double StatsFunctions::beta_q(double p, double shape1, double shape2, double tol, int max_iter) {
        assert(p >= 0.0 && p <= 1.0);
        assert(shape1 > 0.0 && shape2 > 0.0);

        double lower = 0.0;
        double upper = 1.0;

        int iter = 0;
        while (upper - lower > tol && iter < max_iter) {
            double x = (lower + upper) / 2;
            double cdf_val = beta_cdf(x, shape1, shape2);

            if (std::abs(cdf_val - p) < tol) {
                return x;
            }
            if (cdf_val < p) {
                lower = x;
            }
            else {
                upper = x;
            }
            iter++;
        }
        return (lower + upper) / 2;
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
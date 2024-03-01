#include "Header.h"
#include "Reader.h"

#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <map>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include<fstream>
#include<sstream>
#include<typeinfo>
#include<filesystem>
#include<variant>
#include<stdexcept>
#include<ctime>
#include<numeric>
#include<stdio.h>
#include<chrono>
#include<iomanip>
#include<Eigen/Dense>


#ifdef _M_X64
#include <pybind11/embed.h>  
#include <pybind11/stl.h>    
namespace py = pybind11;
py::scoped_interpreter guard{};
using namespace py::literals;
#endif


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
    return std::max(locHat * (1 - shapeHat), std::numeric_limits<double>::epsilon());
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

void ECDF::plotECDF() {
#ifdef _M_X64
    py::module plt = py::module::import("matplotlib.pyplot");

    try {
        plt.attr("plot")(sorted_data, ecdf_values);
        plt.attr("title")("Empirical Cumulative Distribution Function (ECDF)");
        plt.attr("xlabel")("X");
        plt.attr("ylabel")("ECDF");
        plt.attr("show")();
    }
    catch (const std::exception& ex) {
        std::cerr << "C++ Exception: " << ex.what() << std::endl;
    }
#endif
}

template <typename T1, typename T2>
static inline typename T1::value_type StatsFunctions::Quantile(const T1& x, T2 q) {
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

std::vector<EVTCopula::GPDResult> EVTCopula::f_FitGPD(const std::vector<std::vector<double>>& data, std::optional<double> lower, 
    std::optional<double> upper, int min_obs, std::string method, bool lower_tail, bool double_tail) {
    assert(min_obs >= 150);

    size_t numColumns = 2;
    std::vector<EVTCopula::GPDResult> results;

    for (size_t columnIndex = 0; columnIndex < numColumns; ++columnIndex) {
        EVTCopula::GPDResult result;

        std::vector<double> columnValues;
        for (const auto& row : data) {
            columnValues.push_back(row[columnIndex]);
        }

        ECDF ecdf(columnValues);

        //ecdf.plotECDF();

        if (!lower.has_value()) {
            double eval = ecdf.getSortedData()[min_obs - 1];
            lower = ecdf(eval)/2;
        }

        double lower_quant = StatsFunctions().Quantile(columnValues, *lower);

        std::vector<double> excess;

        for (size_t i = 0; i < columnValues.size(); ++i) {
            double val = columnValues[i];

            if (val <= lower_quant) {
                excess.push_back(-val - (-lower_quant));
            }
        }

        auto fit_result = StatsFunctions().fitMoments(excess);

        result.shape = fit_result["shape"];
        result.scale = fit_result["scale"];
        result.threshold = lower_quant;
        result.excesses.insert(result.excesses.end(), excess.begin(), excess.end());
        results.push_back(result);
    }
    return results;
}


std::vector<std::vector<double>> EVTCopula::f_CopulasEmpirical(const std::vector<std::vector<double>>& data, std::vector<EVTCopula::GPDResult>& fit) {
    size_t numColumns = 2;
    std::vector<std::vector<double>> copula;

    for (size_t columnIndex = 0; columnIndex < numColumns; ++columnIndex) {
        EVTCopula::GPDResult result;

        std::vector<double> columnValues;
        for (const auto& row : data) {
            columnValues.push_back(row[columnIndex]);
        }

        auto res = f_FastpSPGPD(columnValues, fit[columnIndex]);
        copula.push_back(res);
    }
    return copula;
}

struct IndexedValue {
    double value;
    size_t index;

    IndexedValue(double val, size_t idx) : value(val), index(idx) {}
};

std::vector<double> EVTCopula::f_FastpSPGPD(const std::vector<double>& data, EVTCopula::GPDResult& fit) {
    
    ECDF ecdf(data);

    double shape = fit.shape;
    double scale = fit.scale;
    double u = fit.threshold;
    const std::vector<double>& excess = fit.excesses;

    auto ecdf_sorted = ecdf.getSortedData();

    int n_u = excess.size();
    int n = ecdf_sorted.size();

    std::vector<IndexedValue> indexedData;
    indexedData.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        indexedData.emplace_back(data[i], i);
    }

    // Split x into lower and upper parts based on the threshold u
    std::vector<double> x_lower, x_upper;
    std::vector<size_t> indices_lower, indices_upper;

    for (const auto& indexedValue : indexedData) {
        double val = indexedValue.value;

        if (val <= u) {
            x_lower.push_back(val);
            indices_lower.push_back(indexedValue.index);
        }
        else {
            x_upper.push_back(val);
            indices_upper.push_back(indexedValue.index);
        }
    }

    // Calculate probabilities for the lower part
    for (int i = 0; i < x_lower.size(); ++i) {
        int most_similar = 0;
        double min_abs_diff = std::abs(x_lower[i] - ecdf_sorted[0]);
        for (int j = 1; j < ecdf_sorted.size(); ++j) {
            double abs_diff = std::abs(x_lower[i] - ecdf_sorted[j]);
            if (abs_diff < min_abs_diff) {
                min_abs_diff = abs_diff;
                most_similar = j;
            }
        }

        int index = (x_lower[i] < ecdf_sorted[most_similar]) ? most_similar - 1 : most_similar;
        x_lower[i] = ((index + 1) - 0.5) / n;
    }

    // Sort x_lower based on the original indices
    std::sort(indices_lower.begin(), indices_lower.end(), [&indexedData](size_t a, size_t b) {
        return indexedData[a].value < indexedData[b].value;
    });

    // Calculate probabilities for the upper part
    for (int i = 0; i < x_upper.size(); ++i) {
        double scaling = static_cast<double>(n_u) / static_cast<double>(n);
        double intermediate_result = std::pow(1.0 + shape * (x_upper[i] - u) / scale, (-1.0 / shape));

        if (std::isfinite(intermediate_result)) {
            double val = 1.0 - scaling * intermediate_result;
            x_upper[i] = val;
        }
        else {
            x_upper[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Sort x_upper based on the original indices
    std::sort(indices_upper.begin(), indices_upper.end(), [&indexedData](size_t a, size_t b) {
        return indexedData[a].value < indexedData[b].value;
    });

    // Combine x_lower and x_upper while preserving the original order
    std::vector<double> result(data.size(), 0.0);

    for (size_t i = 0; i < result.size(); ++i) {
        auto it_lower = std::find(indices_lower.begin(), indices_lower.end(), i);
        auto it_upper = std::find(indices_upper.begin(), indices_upper.end(), i);

        if (it_lower != indices_lower.end()) {
            size_t index_in_indices = std::distance(indices_lower.begin(), it_lower);
            result[i] = x_lower[index_in_indices];
        }
        else if (it_upper != indices_upper.end()) {
            size_t index_in_indices = std::distance(indices_upper.begin(), it_upper);
            result[i] = x_upper[index_in_indices];
        }
    }
    return result;
}


double EVTCopula::f_TailDep(const std::vector<std::vector<double>>& data, double threshold) {
    double tailDepSum = 0.0;

    auto fit = f_FitGPD(data);
    auto emp_copula = f_CopulasEmpirical(data, fit);

    for (const auto& singleVector : data) {
        ECDF ecdf(singleVector);
        auto ecdf_sorted = ecdf.getSortedData();

        int n = ecdf_sorted.size();
        int n_u = 0;
        for (double val : ecdf_sorted) {
            if (val > threshold) {
                n_u++;
            }
        }

        double p = static_cast<double>(n_u) / n;
        double q = 1.0 - p;

        double u = threshold;
        double x = ecdf_sorted[n_u];

        double z = (x - u) / (x - threshold);
        double z2 = z * z;

        double tail_dep = 1.0 / (1.0 - z2) * (1.0 - (p * z2 + q) / (p * z2 + q - 1.0));

        tailDepSum += tail_dep;
    }

    // Calculate the average tail dependence
    double averageTailDep = tailDepSum / data.size();

    return averageTailDep;
}


// numerical integration of f(x) from a to b
double trapezoidal(double a, double b, int n, std::function<double(double)> f) {
    double dx = (b - a) / n;
    double integral = f(a) + f(b);
    for (int i = 1; i <= n - 1; i++) {
        integral += 2.0 * f(a + i * dx);
    }
    integral *= dx / 2.0;
    return integral;
}

# define M_PI 3.14159265358979323846

float erfinv(float x) {
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0) ? -1.0f : 1.0f;

    x = (1 - x) * (1 + x);
    lnx = logf(x);

    tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
    tt2 = 1 / (0.147) * lnx;

    return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

double norm_pdf(double value) {
    return 0.5 * erfc(-value * sqrt(0.5));
}

double norm_cdf(double value) {
    return 0.5 * erfc(-value * sqrt(0.5));
}

double norm_q(double p, double mean, double sigma) {
    return (mean + sigma * sqrt(2) * erfinv(2 * p - 1));
}

// Uniform distribution

double unif_pdf(double x, double a, double b) {
    if (x < a || x > b) {
        return 0;
    }
    return 1 / (b - a);
}

double unif_cdf(double x, double a, double b) {
    if (x < a) {
        return 0;
    }
    if (x > b) {
        return 1;
    }
    return (x - a) / (b - a);
}

double unif_q(double p, double a, double b) {
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    return a + (b - a) * p;
}


double beta(double x, double y) {
    return tgamma(x) * tgamma(y) / tgamma(x + y);
}

double beta_pdf(double x, double shape1, double shape2) {
    assert(x >= 0.0 && x <= 1.0);
    assert(shape1 > 0.0 && shape2 > 0.0);
    return pow(x, shape1 - 1) * pow(1 - x, shape2 - 1) / beta(shape1, shape2);
}

double beta_cdf(double x, double shape1, double shape2) {
    assert(x >= 0.0 && x <= 1.0);
    assert(shape1 > 0.0 && shape2 > 0.0);

    double beta_term = beta(shape1, shape2);

    std::function<double(double)> integrand_function = [shape1, shape2](double t) {
        return pow(t, shape1 - 1) * pow(1 - t, shape2 - 1);
        };

    double integral = trapezoidal(0.0, x, 10000, integrand_function);
    return integral / beta_term;
}

double beta_q(double p, double shape1, double shape2, double tol, int max_iter) {
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


double gamma_pdf(double x, double shape, double scale) {
    if (x < 0) {
        return 0;
    }
    if (shape <= 0 || scale <= 0) {
        return 0;
    }
    return pow(x, shape - 1) * exp(-x / scale) / (tgamma(shape) * pow(scale, shape));
}

double gamma_cdf(double x, double shape, double scale) {
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

double gamma_q(double p, double shape, double scale, double tol = 1e-6, int max_iter = 1000) {
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

double exp_pdf(double x, double lambda) {
    if (x < 0 || lambda <= 0) {
        return 0;
    }
    return lambda * exp(-lambda * x);
}

double exp_cdf(double x, double lambda) {
    if (x < 0 || lambda <= 0) {
        return 0;
    }
    return 1 - exp(-lambda * x);
}

double exp_q(double p, double lambda) {
    return -log(1 - p) / lambda;
}

void writeVectorToFile(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& value : vec) {
            outFile << value << "," << std::endl;
        }
        outFile.close();
    }
    else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

/// <summary>

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

struct MarginalInfo {
    MarginalType type1;
    MarginalType type2;
    DistributionParams params1;
    DistributionParams params2;

    MarginalInfo(MarginalType t1, MarginalType t2, const std::vector<double>& p1, const std::vector<double>& p2)
        : type1(t1), type2(t2), params1({ p1 }), params2({ p2 }) {}
};

struct CopulaInfo {
	std::string type;
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

std::function<double(double)> getQuantileFunction(MarginalType type, const std::vector<double>& parameters) {
    switch (type) {
    case NORMAL:
        return std::bind(norm_q, std::placeholders::_1, parameters[0], parameters[1]);
    case UNIFORM:
        return std::bind(unif_q, std::placeholders::_1, parameters[0], parameters[1]);
    case GAMMA:
        return std::bind(gamma_q, std::placeholders::_1, parameters[0], parameters[1], 1e-6, 1000);
    case EXPONENTIAL:
        return std::bind(exp_q, std::placeholders::_1, parameters[0]);
    case BETA:
        return std::bind(beta_q, std::placeholders::_1, parameters[0], parameters[1], 1e-6, 1000);
    case UNKNOWN:
        throw std::invalid_argument("Unsupported marginal distribution type");
    }
}

#include <random>

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

Eigen::MatrixXd rmvnorm_samples(int n, const double Mean, const double& Sigma) {
    Eigen::VectorXd mean(2);
    mean << Mean, Mean;

    Eigen::MatrixXd sigma(2, 2);
    sigma << 1, Sigma, Sigma, 1;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(sigma);
    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd ev = solver.eigenvectors().real();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < 0) {
            eigenvalues(i) = 0;
        }
    }
    eigenvalues = eigenvalues.cwiseSqrt();
    Eigen::MatrixXd adjustedEv = ev.transpose().array().colwise() * eigenvalues.array();
    ev = (ev * adjustedEv).transpose();

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    Eigen::MatrixXd randomSamples(n, sigma.cols());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < sigma.cols(); ++j) {
            randomSamples(i, j) = distribution(generator);
        }
    }
    randomSamples = randomSamples * ev.transpose();
    randomSamples.rowwise() += mean.transpose();

    return randomSamples;
};


inline std::pair<std::vector<double>, std::vector<double>> rfrankBivCopula(int n, double alpha) {
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


std::vector<std::vector<double>> rCopula(int n, const CopulaInfo& copula, const MarginalInfo& marginals) {
    std::vector<double> u(n);
    std::vector<double> v(n);

    std::vector<std::vector<double>> result;
    result.resize(n, std::vector<double>(2));

    if (copula.type == "independent") {
        for (int i = 0; i < n; ++i) {
            u[i] = generate_uniform<double>(1);
            v[i] = generate_uniform<double>(1);
        }
    }
    else if (copula.type == "normal") {
		Eigen::MatrixXd samples = rmvnorm_samples(n, copula.parameters[0], copula.parameters[1]);
        for (int i = 0; i < n; ++i) {
			u[i] = norm_cdf(samples(i, 0));
			v[i] = norm_cdf(samples(i, 1));
		}
	}
    else if (copula.type == "clayton") {
        for (int i = 0; i < n; ++i) {
            double random1 = generate_uniform<double>(1); // u
            double random2 = generate_uniform<double>(1); // v

            u[i] = random1;
            v[i] = std::pow(std::pow(random1, -copula.parameters[0]) + std::pow(random2, (-copula.parameters[0] / (copula.parameters[0] + 1))) - 1.0, -1.0 / copula.parameters[0]);
        }
    }
    else if (copula.type == "frank") {
        auto samples = rfrankBivCopula(n, copula.parameters[0]);
        std::copy(samples.first.begin(), samples.first.end(), u.begin());
        std::copy(samples.second.begin(), samples.second.end(), v.begin());
    }
    else {
        throw std::invalid_argument("Invalid copula type: " + copula.type);
    }

    std::function<double(double)> qdfExpr1 = getQuantileFunction(marginals.type1, marginals.params1.parameters);
    std::function<double(double)> qdfExpr2 = getQuantileFunction(marginals.type2, marginals.params2.parameters);
    
    for (int i = 0; i < n; ++i) {
        result[i][0] = qdfExpr1(u[i]);
        result[i][1] = qdfExpr2(v[i]);
    }

    return result;
}





int main() {
    using CSVValue = std::variant<int, double, std::string>;
    /*
    MarginalInfo marginalInfo = getMarginalInfo("uniform", { 0.0, 1.0 }, "exponential", { 2.0 });
    CopulaInfo copulaInfo = { "normal", { 0.0, 0.9 } };

    
    std::vector<std::vector<double>> result = rCopula(1000, copulaInfo, marginalInfo);

    std::vector<std::vector<CSVValue>> csvData;
    for (const auto& row : result) {
        std::vector<CSVValue> csvRow(row.begin(), row.end());
        csvData.push_back(csvRow);
    }

    //csv_Reader::write_csv(csvData, "copula.csv");

    // Test 2

    MarginalInfo marginalInfofrank = getMarginalInfo("beta", { 2.0, 5.0 }, "beta", { 3.0, 2.0 });
    CopulaInfo copulaInfoFrank = { "frank", { 4.0 } };

    std::vector<std::vector<double>> resultFrank = rCopula(5000, copulaInfoFrank, marginalInfofrank);

    std::vector<std::vector<CSVValue>> csvData1;
    for (const auto& row : resultFrank) {
        std::vector<CSVValue> csvRow(row.begin(), row.end());
        csvData1.push_back(csvRow);
    }
    */

    //csv_Reader::write_csv(csvData1, "copula.csv");

    // Test 3

    MarginalInfo marginalInfoClayton = getMarginalInfo( "exponential", { 2.0 }, "uniform", { 0.0, 1.0 });
    CopulaInfo copulaInfoClayton = { "clayton", { 2.0 } };

    std::vector<std::vector<double>> resultClayton = rCopula(5000, copulaInfoClayton, marginalInfoClayton);

    std::vector<std::vector<CSVValue>> csvDataClayton;
    for (const auto& row : resultClayton) {
        std::vector<CSVValue> csvRow(row.begin(), row.end());
        csvDataClayton.push_back(csvRow);
    }

    csv_Reader::write_csv(csvDataClayton, "copula.csv");


    // Normal distribution
    std::cout << "Normal PDF at 0: " << norm_pdf(0) << std::endl;
    std::cout << "Normal CDF at 0.5: " << norm_cdf(0.5) << std::endl;
    std::cout << "Normal quantile for 0.2: " << norm_q(0.2, 0, 1) << std::endl;

    // Uniform distribution
    std::cout << "Uniform PDF at 0.3: " << unif_pdf(0.3, 0, 1) << std::endl;
    std::cout << "Uniform CDF at 0.7: " << unif_cdf(0.7, 0, 1) << std::endl;
    std::cout << "Uniform quantile for 0.4: " << unif_q(0.4, 0, 1) << std::endl;

    // Gamma distribution
    std::cout << "Gamma PDF at 2.5: " << gamma_pdf(2.5, 2, 1) << std::endl;
    std::cout << "Gamma CDF at 3.0: " << gamma_cdf(3.0, 2, 1) << std::endl;
    std::cout << "Gamma quantile for 0.8: " << gamma_q(0.8, 2, 1) << std::endl;

    // Exponential distribution
    std::cout << "Exponential PDF at 1.5: " << exp_pdf(1.5, 2) << std::endl;
    std::cout << "Exponential CDF at 2.0: " << exp_cdf(2.0, 2) << std::endl;
    std::cout << "Exponential quantile for 0.6: " << exp_q(0.6, 2) << std::endl;
    
    //writeVectorToFile(random_t_values, "random_t_values.csv");

    /*
    using CSVValue = std::variant<int, double, std::string>;

    size_t csvLineCount;
    bool skipFirstColumn = true;
    std::vector<std::vector<CSVValue>> csvData = csv_Reader::readcsv("test.csv", csvLineCount, skipFirstColumn);
    auto resultVector = csv_Reader::convertToDouble(csvData);

    EVTCopula copula;
    auto result = copula.f_TailDep(resultVector, 0.05);
    */
    return 0;
}



/*

#include <Eigen/Dense>
#include <functional>
#include <iostream>

template<typename VectorType>
class GradientDescent {
    private:
        VectorType gradient(const std::function<double(const VectorType&)>& f, const VectorType& x0, double heps) {
            int n = x0.size();
            VectorType gr(n);

            for (int i = 0; i < n; ++i) {
                VectorType x_plus_h = x0;
                VectorType x_minus_h = x0;

                x_plus_h[i] += heps;
                x_minus_h[i] -= heps;

                gr(i) = (f(x_plus_h) - f(x_minus_h)) / (2 * heps);
            }

            return gr;
        }

    public:
        VectorType gradientDescent(const std::function<double(const VectorType&)>& f, const VectorType& initial_guess, double step_size, double tolerance, double heps, int max_iterations = 1000) {
            VectorType x = initial_guess;
            int iter = 0;
            while (iter < max_iterations) {
                VectorType grad = gradient(f, x, heps);
                x -= step_size * grad;

                if (grad.norm() < tolerance) {
                    break;
                }

                ++iter;
            }
            if (iter == max_iterations) {
                std::cout << "Maximum iterations reached." << std::endl;
            }

            return x;
        }
};

double bealeFunction(const Eigen::VectorXd& x) {
    double term1 = pow(1.5 - x(0) + x(0) * x(1), 2);
    double term2 = pow(2.25 - x(0) + x(0) * pow(x(1), 2), 2);
    double term3 = pow(2.625 - x(0) + x(0) * pow(x(1), 3), 2);
    return term1 + term2 + term3;
}

int main() {
    GradientDescent<Eigen::VectorXd> gd;

    Eigen::VectorXd initial_guess(2);
    initial_guess << 0, 0;

    double step_size = 0.01;
    double tolerance = 1e-6;
    double heps = 1e-10;

    Eigen::VectorXd solution = gd.gradientDescent(bealeFunction, initial_guess, step_size, tolerance, heps);
    std::cout << "Solution: " << solution.transpose() << std::endl;
    std::cout << "Minimum value: " << bealeFunction(solution) << std::endl;

    return 0;
}
*/
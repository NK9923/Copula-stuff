#include "pch.h"
#include "Header.h"

#ifdef _M_X64
#include <pybind11/embed.h>  // py::scoped_interpreter
#include <pybind11/stl.h>    // bindings from C++ STL containers to Python types
namespace py = pybind11;
py::scoped_interpreter guard{};
namespace py = pybind11;
using namespace py::literals;
#endif

# define M_PI 3.14159265358979323846

void copula::getwd() {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::cout << "Current Working Directory: " << currentPath << std::endl;
}

// CLASS Differentiation contains hessian and gradient computation
std::vector<double> copula::differentiation::gradient(double (*f)(const std::vector<double>&), const std::vector<double>& x0, double heps) {
    int n = x0.size();
    std::vector<double> gr(n, 0.0);

    for (int i = 0; i < n; ++i) {
        std::vector<double> x_plus_h = x0;
        std::vector<double> x_minus_h = x0;

        x_plus_h[i] += heps;
        x_minus_h[i] -= heps;

        gr[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * heps);
    }

    return gr;
}

//========================================

// Statsfunctions
std::map<std::string, double> copula::StatsFunctions::fitMoments(const std::vector<double>& data) {
    std::map<std::string, double> result;

    double locHat = this->mean(data);
    double sig2Hat = this->variance(data);

    result["shape"] = this->calculateShape(locHat, sig2Hat);
    result["scale"] = this->calculateScale(locHat, result["shape"]);
    result["loc"] = locHat;

    return result;
}

inline double copula::StatsFunctions::calculateShape(double locHat, double sig2Hat) const {
    return (1 - std::pow(locHat, 2) / sig2Hat) / 2;
}

inline double copula::StatsFunctions::calculateScale(double locHat, double shapeHat) const {
    return max(locHat * (1 - shapeHat), std::numeric_limits<double>::epsilon());
}

inline double copula::StatsFunctions::mean(const std::vector<double>& data) const {
    if (data.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double& value : data) {
        sum += value;
    }
    return sum / data.size();
}

inline double copula::StatsFunctions::variance(const std::vector<double>& data) const {
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
}

inline double copula::StatsFunctions::skewness(const std::vector<double>& data) const {
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

inline double copula::StatsFunctions::kurtosis(const std::vector<double>& data) const {
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

std::vector<double> copula::StatsFunctions::generate_uniform(int N_sim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<double> randomValues;

    for (int i = 0; i < N_sim; ++i) {
        randomValues.push_back(dis(gen));
    }
    return randomValues;
}    

std::vector<double> copula::StatsFunctions::generate_gaussian(int N_sim, double mean, double stddev) {
    
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> norm(mean, stddev);
    std::vector<double> randomValues;


    for (int i = 0; i < N_sim; ++i) {
        randomValues.push_back(norm(gen));
    }
    return randomValues;
}

std::vector<double> copula::StatsFunctions::generate_pareto(int N, double g, double k) {
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

std::vector<double> copula::StatsFunctions::generate_cauchy(int N, double location, double scale) {
    std::mt19937 gen(std::random_device{}());
    std::cauchy_distribution<double> dis(location, scale);

    std::vector<double> randomValues;

    for (int i = 0; i < N; ++i) {
        randomValues.push_back(dis(gen));
    }

    return randomValues;
}

std::vector<double> copula::StatsFunctions::generate_beta(int N, double alpha, double beta) {
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

void copula::StatsFunctions::plotDistribution(std::vector<double>& data) {
    #ifdef _M_X64
        std::vector<double> bins(1000);
        py::module plt = py::module::import("matplotlib.pyplot");

        try {
            plt.attr("hist")(data, "bins"_a = 50);
            plt.attr("show")();
        }
        catch (const std::exception& ex) {
            std::cerr << "C++ Exception: " << ex.what() << std::endl;
        }
    #endif
}

std::pair<double, bool> copula::StatsFunctions::rlogseries_ln1p(double a, double cutoff) {

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

float copula::StatsFunctions::erfinv(float x) {
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0) ? -1.0f : 1.0f;

    x = (1 - x) * (1 + x);
    lnx = logf(x);

    tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
    tt2 = 1 / (0.147) * lnx;

    return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

double copula::StatsFunctions::pnorm(double value) {
    return 0.5 * erfc(-value * sqrt(0.5));
}

double copula::StatsFunctions::qnorm(double p, double mean, double sigma) {
    return(mean + sigma * sqrt(2) * erfinv(2 * p - 1));
}

double copula::StatsFunctions::qunif(double p, double a, double b) {
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    return a + (b - a) * p;
}

//========================================

void copula::GaussCopula::printMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    if (this->debug) {
        std::cout << name << ":\n" << matrix << "\n\n";
    }
}

Eigen::MatrixXd copula::GaussCopula::rmvnorm_samples(int n, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma) {
    if (sigma.rows() != 2 || sigma.cols() != 2) {
        assert("Sigma must be a 2x2 matrix");
    }

    Eigen::EigenSolver<Eigen::MatrixXd> solver(sigma);
    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd ev = solver.eigenvectors().real();
    printMatrix(eigenvalues, "Eigenvalues");
    printMatrix(ev, "Eigenvectors");

    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < 0) {
            eigenvalues(i) = 0;
        }
    }
    eigenvalues = eigenvalues.cwiseSqrt();
    printMatrix(eigenvalues.array(), "eigenvalues.array()");

    Eigen::MatrixXd adjustedEv = ev.transpose().array().colwise() * eigenvalues.array();
    printMatrix(adjustedEv, "adjustedEv");

    ev = (ev * adjustedEv).transpose();
    printMatrix(ev, "Adjusted Cholesky Decomposition");

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

std::pair<std::vector<double>, std::vector<double>> copula::GaussCopula::rGaussCopula(int N_sim, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma, QuantileFunction f1, QuantileFunction f2) {
    Eigen::MatrixXd rmvnorm = GaussCopula::rmvnorm_samples(N_sim, mean, sigma);
    std::vector<double> result1, result2;
    for (int i = 0; i < rmvnorm.rows(); ++i) {
        if (GaussCopula::debug) {
            std::cout << rmvnorm(i, 0) << std::endl;
            std::cout << StatsFunctions::pnorm(rmvnorm(i, 0)) << std::endl;
            std::cout << f1(StatsFunctions::pnorm(rmvnorm(i, 0)));
        }

        result1.push_back(f1(StatsFunctions::pnorm(rmvnorm(i, 0))));
        result2.push_back(f2(StatsFunctions::pnorm(rmvnorm(i, 1))));
    }
    return std::make_pair(result1, result2);
}

void copula::GaussCopula::PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, double cor) {
    #ifdef _M_X64
    try {
        const std::vector<double>& x = copula_data.first;
        const std::vector<double>& y = copula_data.second;

        py::module plt = py::module::import("matplotlib.pyplot");
        std::string cor_str = std::to_string(cor);

        plt.attr("scatter")(x, y, "alpha"_a = 0.5, "color"_a = "orange");
        plt.attr("title")("Scatter Plot of Gaussian Copula Samples - Correlation: " + cor_str);
        plt.attr("xlabel")("X");
        plt.attr("ylabel")("Y");
        plt.attr("show")();
    }
    catch (const std::exception& ex) {
        std::cerr << "C++ Exception: " << ex.what() << std::endl;
    }
    #endif
}

//========================================

// ECDF
copula::ECDF::ECDF(const std::vector<double>& data) {
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

double copula::ECDF::operator()(double x) const {
    auto it = std::lower_bound(sorted_data.begin(), sorted_data.end(), x);
    if (it == sorted_data.end()) {
        return 1.0;
    }
    else {
        size_t index = std::distance(sorted_data.begin(), it);
        return ecdf_values[index];
    }
}

inline double copula::ECDF::head_ecdf(int min_obs) const {
    if (min_obs <= 0) {
        return 0.0;
    }
    min_obs = min(min_obs, static_cast<int>(sorted_data.size()));
    return ecdf_values[min_obs - 1];
}

//========================================

// Frank Copula implementation

inline void copula::FrankCopula::PrintInfo() {
    std::cout << "== Frank Copula ===" << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
}

// CDF of the Frank copula
inline double copula::FrankCopula::cdfExpr(const std::vector<double>& u, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum += -log((exp(-alpha * u[i]) - 1) / (exp(-alpha) - 1));
    }
    return -1.0 / alpha * log(1.0 + exp(-sum) * (exp(-alpha) - 1));
};

// PDF of the Frank copula
inline double copula::FrankCopula::pdfExpr(const std::vector<double>& u) {
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
std::pair<std::vector<double>, std::vector<double>> copula::FrankCopula::rfrankCopula(int n) {
    if (dim == 2) {
        return rfrankBivCopula(n);
    }

    std::vector<double> samples_U;
    std::vector<double> samples_V;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (copula::StatsFunctions().rlogseries_ln1p(2).first == 0) {
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

void copula::FrankCopula::PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data) {

    #ifdef _M_X64
    try {
        const std::vector<double>& x = copula_data.first;
        const std::vector<double>& y = copula_data.second;

        py::module plt = py::module::import("matplotlib.pyplot");

        plt.attr("scatter")(x, y, "alpha"_a = 0.5);
        plt.attr("title")("Scatter Plot of Copula Samples");
        plt.attr("xlabel")("X");
        plt.attr("ylabel")("Y");
        plt.attr("show")();
    }
    catch (const std::exception& ex) {
        std::cerr << "C++ Exception: " << ex.what() << std::endl;
    }
    #endif
}

inline std::pair<std::vector<double>, std::vector<double>> copula::FrankCopula::rfrankBivCopula(int n) {
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

//========================================

//Core Copula stuff
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

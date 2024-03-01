#include <Eigen/Dense>
#include<random>
#include<vector>

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


std::vector<double> generate_chi_squared(int N, double df) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::chi_squared_distribution<double> dist{df};

    std::vector<double> randomValues;

    for (int i = 0; i < N; ++i) {
        randomValues.push_back(dist(gen));
    }

    return randomValues;
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

Eigen::MatrixXd rmvt_samples(int n, double df, double sigma, double mean = 0) {
    auto standard_normal = rmvnorm_samples(n, mean, sigma);
    auto Chi_squared = generate_chi_squared(n, df);
    auto random_t = standard_normal.array() / sqrt(Chi_squared.array() / df);

    random_t.rowwise() += Eigen::VectorXd::Constant(n, mean);

    return random_t;
}

double StatsFunctions::t_pdf(double x, int df) {
        double pi = 4.0 * atan(1.0);
        double gamma_term = tgamma(0.5 * (df + 1.0)) / tgamma(0.5 * df) / sqrt(df * pi);
        double expression_term = pow(1.0 + (x * x / df), -0.5 * (df + 1.0));
        return gamma_term * expression_term;
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
    else if(copula.type == "t") {
        Eigen::MatrixXd samples = rmvt_samples(n, copula.parameters[0], copula.parameters[1], copula.parameters[2]);
        for (int i = 0; i < n; ++i) {
			u[i] = t_pdf(samples(i, 0), copula.parameters[0]);
			v[i] = t_pdf(samples(i, 1), copula.parameters[0]);
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


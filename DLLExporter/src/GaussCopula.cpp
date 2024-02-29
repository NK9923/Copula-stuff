#include "Header.h"
#include "pch.h"

// Gauss Copula implementation

namespace copula {

    void GaussCopula::printMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
        if (this->debug) {
            std::cout << name << ":\n" << matrix << "\n\n";
        }
    }

    Eigen::MatrixXd GaussCopula::rmvnorm_samples(int n, const double Mean, const double& Sigma) {
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

    Eigen::MatrixXd GaussCopula::rmvnorm_samples(int n, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma) {
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

    std::pair<std::vector<double>, std::vector<double>> GaussCopula::rGaussCopula(int N_sim, const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma, QuantileFunction f1, QuantileFunction f2) {
        Eigen::MatrixXd rmvnorm = GaussCopula::rmvnorm_samples(N_sim, mean, sigma);
        std::vector<double> result1, result2;
        for (int i = 0; i < rmvnorm.rows(); ++i) {
            if (GaussCopula::debug) {
                std::cout << rmvnorm(i, 0) << std::endl;
                std::cout << StatsFunctions::norm_pdf(rmvnorm(i, 0)) << std::endl;
                std::cout << f1(StatsFunctions::norm_pdf(rmvnorm(i, 0)));
            }

            result1.push_back(f1(StatsFunctions::norm_pdf(rmvnorm(i, 0))));
            result2.push_back(f2(StatsFunctions::norm_pdf(rmvnorm(i, 1))));
        }
        return std::make_pair(result1, result2);
    }
}

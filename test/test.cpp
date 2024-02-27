#include <iostream>
#include "Header.h"
#include <vector>
#include <fstream>

#ifdef _M_X64
#include <pybind11/embed.h>  // py::scoped_interpreter
#include <pybind11/stl.h>    // bindings from C++ STL containers to Python types
namespace py = pybind11;
#endif

double schaffer_N6(const std::vector<double>& x) {
    double term1 = std::sin(std::sqrt(x[0] * x[0] + x[1] * x[1]));
    double term2 = 1.0 + 0.001 * (x[0] * x[0] + x[1] * x[1]);
    double sum = 0.5 + std::pow(term1, 2) / std::pow(term2, 2);

    return sum;
}

int main()
{
    copula::getwd();

    // Normal distribution
    std::cout << "\n--- Normal Distribution ---" << std::endl;
    std::cout << "Normal PDF at 0: " << copula::StatsFunctions::norm_pdf(0) << std::endl;
    std::cout << "Normal CDF at 0.5: " << copula::StatsFunctions::norm_cdf(0.5) << std::endl;
    std::cout << "Normal quantile for 0.2: " << copula::StatsFunctions::norm_q(0.2, 0, 1) << std::endl;

    // Uniform distribution
    std::cout << "\n--- Uniform Distribution ---" << std::endl;
    std::cout << "Uniform PDF at 0.3: " << copula::StatsFunctions::unif_pdf(0.3, 0, 1) << std::endl;
    std::cout << "Uniform CDF at 0.7: " << copula::StatsFunctions::unif_cdf(0.7, 0, 1) << std::endl;
    std::cout << "Uniform quantile for 0.4: " << copula::StatsFunctions::unif_q(0.4, 0, 1) << std::endl;

    // Gamma distribution
    std::cout << "\n--- Gamma Distribution ---" << std::endl;
    std::cout << "Gamma PDF at 2.5: " << copula::StatsFunctions::gamma_pdf(2.5, 2, 1) << std::endl;
    std::cout << "Gamma CDF at 3.0: " << copula::StatsFunctions::gamma_cdf(3.0, 2, 1) << std::endl;
    std::cout << "Gamma quantile for 0.8: " << copula::StatsFunctions::gamma_q(0.8, 2, 1) << std::endl;

    // Exponential distribution
    std::cout << "\n--- Exponential Distribution ---" << std::endl;
    std::cout << "Exponential PDF at 1.5: " << copula::StatsFunctions::exp_pdf(1.5, 2) << std::endl;
    std::cout << "Exponential CDF at 2.0: " << copula::StatsFunctions::exp_cdf(2.0, 2) << std::endl;
    std::cout << "Exponential quantile for 0.6: " << copula::StatsFunctions::exp_q(0.6, 2) << std::endl;

    // Beta distribution
    std::cout << "\n--- Beta Distribution ---" << std::endl;
    std::cout << "Beta PDF at 0.2: " << copula::StatsFunctions::beta_cdf(0.2, 1, 5) << std::endl;
    std::cout << "Beta CDf at 0.1: " << copula::StatsFunctions::beta_pdf(0.2, 1, 5) << std::endl;
    std::cout << "Beta quantile for 0.2: " << copula::StatsFunctions::beta_q(0.1, 1, 5) << std::endl;

    // Student's t-distribution
    std::cout << "\n--- Student's t Distribution ---" << std::endl;
    std::cout << "Student t PDF at 2: " << copula::StatsFunctions::t_pdf(2, 5) << std::endl;
    std::cout << "Student t CDF at 2: " << copula::StatsFunctions::t_cdf(2, 5) << std::endl;
    std::cout << "Student t quantile for 0.95: " << copula::StatsFunctions::t_q(0.95, 5) << std::endl;

    // Test if gradient computation works
    std::vector<double> result = copula::NumericalDifferentiation::gradient(schaffer_N6, { 1, 2 });

    std::cout << "\nGradient: ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    copula::StatsFunctions statsObj;
    
    std::vector<double> random;

    std::cout << "\n--- Choose a distribution ---" << std::endl;
    std::cout << "1. Uniform Distribution" << std::endl;
    std::cout << "2. Gaussian Distribution" << std::endl;
    std::cout << "3. Pareto Distribution" << std::endl;
    std::cout << "4. Cauchy Distribution" << std::endl;
    std::cout << "5. Beta Distribution" << std::endl;

    int choice;
    std::cin >> choice;

    switch (choice) {
    case 1:
        random = statsObj.generate_uniform(100000);
        break;
    case 2:
        random = statsObj.generate_gaussian(100000, 0, 1);
        break;
    case 3:
        random = statsObj.generate_pareto(100000, 3, 2);
        break;
    case 4:
        random = statsObj.generate_cauchy(100000, 0, 1);
        break;
    case 5:
        random = statsObj.generate_beta(100000, 2, 5);
        break;
    default:
        std::cout << "Invalid choice" << std::endl;
        return 1;
    }

    // Test plot distribution
    #ifdef _M_X64
        statsObj.plotDistribution(random);
    #endif

    // Test Frank Copula
    copula::FrankCopula Frank(25);

    double value = Frank.cdfExpr({ 0.1, 0.2 }, 2);
    std::cout << value << std::endl;

    double val1 = Frank.pdfExpr({ 0.1, 0.2 });
    std::cout << val1 << std::endl;


    std::cout << "\n--- Frank Copula ---" << std::endl;
    auto copulaPair = Frank.rfrankCopula(5000); 
    auto copulaPairParteo = Frank.Frank_paretoMarginals(500, 20, 100, 100, 50, 5);
    
    #ifdef _M_X64
        Frank.PlotCopula(copulaPair);
        Frank.PlotCopula(copulaPairParteo);
    #endif

    Eigen::VectorXd mean(2);
    mean << 0, 0;

    Eigen::MatrixXd sigma(2, 2);
    sigma << 1, 0.7, 0.7, 1;

    int Ntest = 1000;

    copula::GaussCopula Gauss(false);

    std::cout << "\n--- Gauss Copula ---" << std::endl;
    auto result_1 = Gauss.rGaussCopula(1000, mean, sigma, [](double x) { return copula::StatsFunctions::unif_q(x); }, [](double x) { return copula::StatsFunctions::unif_q(x); });

    #ifdef _M_X64
        Gauss.PlotCopula(result_1, sigma(0, 1));
        py::finalize_interpreter();
    #endif
    
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}

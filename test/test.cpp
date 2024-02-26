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

    std::cout << copula::StatsFunctions::q_t(0.95, 5) << std::endl;

    std::cout << copula::StatsFunctions::cdf_t(2, 5) << std::endl;

    std::cout << copula::StatsFunctions::pdf_t(2, 5) << std::endl;

    // Test if gradient computation works
    std::vector<double> result = copula::NumericalDifferentiation::gradient(schaffer_N6, { 1, 2 });

    std::cout << "Gradient: ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    copula::StatsFunctions statsObj;
    
    std::vector<double> random;

    std::cout << "Choose a distribution:" << std::endl;
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

    auto result_1 = Gauss.rGaussCopula(1000, mean, sigma, [](double x) { return copula::StatsFunctions::qunif(x); }, [](double x) { return copula::StatsFunctions::qunif(x); });

    #ifdef _M_X64
        Gauss.PlotCopula(result_1, sigma(0, 1));
        py::finalize_interpreter();
    #endif
    
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}

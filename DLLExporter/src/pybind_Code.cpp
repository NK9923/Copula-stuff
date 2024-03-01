#include "pch.h"
#include "Header.h"

#ifdef _M_X64
#include <pybind11/embed.h>  
#include <pybind11/stl.h>    
namespace py = pybind11;
py::scoped_interpreter guard{};
using namespace py::literals;
#endif

namespace copula {
    void StatsFunctions::plotDistribution(std::vector<double>& data) {
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

    void GaussCopula::PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, double cor) {
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
                plt.attr("savefig")("..\\..\\test\\Figures\\GaussCopula.pdf");
                plt.attr("show")();
            }
            catch (const std::exception& ex) {
                std::cerr << "C++ Exception: " << ex.what() << std::endl;
            }
        #endif
    }

    void FrankCopula::PlotCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data) {
        #ifdef _M_X64
            try {
                const std::vector<double>& x = copula_data.first;
                const std::vector<double>& y = copula_data.second;

                py::module plt = py::module::import("matplotlib.pyplot");

                plt.attr("scatter")(x, y, "alpha"_a = 0.5);
                plt.attr("title")("Scatter Plot of Copula Samples");
                plt.attr("xlabel")("X");
                plt.attr("ylabel")("Y");
                plt.attr("savefig")("..\\..\\test\\Figures\\FrankCopula.pdf");
                plt.attr("show")();
            }
            catch (const std::exception& ex) {
                std::cerr << "C++ Exception: " << ex.what() << std::endl;
            }
        #endif
    }

    void CopulaSampling::PlotRCopula(std::pair<std::vector<double>, std::vector<double>>& copula_data, std::string CopulaTyp, std::string mar1, std::string mar2, std::string filename) {
        #ifdef _M_X64
            try {
                const std::vector<double>& x = copula_data.first;
                const std::vector<double>& y = copula_data.second;

                py::module plt = py::module::import("matplotlib.pyplot");

                plt.attr("scatter")(x, y, "alpha"_a = 0.5);
                plt.attr("title")(py::str(CopulaTyp));
                plt.attr("xlabel")(py::str(mar1));
                plt.attr("ylabel")(py::str(mar2));
                plt.attr("savefig")("..\\..\\test\\Figures\\" + filename + ".pdf");
                plt.attr("show")();
            }
            catch (const std::exception& ex) {
                std::cerr << "C++ Exception: " << ex.what() << std::endl;
            }
        #endif
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
}

#include "Header.h"
#include "pch.h"

// CLASS Differentiation contains hessian and gradient computation

namespace copula {

    std::vector<double> NumericalDifferentiation::gradient(double (*f)(const std::vector<double>&), const std::vector<double>& x0, double heps) {
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

    template<typename VectorType>
    VectorType NumericalDifferentiation::gradient(const std::function<double(const VectorType&)>& f, const VectorType& x0, double heps) {
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
}



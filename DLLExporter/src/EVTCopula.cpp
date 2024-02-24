#include "pch.h"
#include "Header.h"

void copula::getwd() {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::cout << "Current Working Directory: " << currentPath << std::endl;
}

// Extreme Value Theory Copula with Pareto marginals 

namespace copula {

    EVTCopula::GPDResult EVTCopula::fit_GPD_PWM(const std::vector<double>& data) {
        GPDResult result;
        return GPDResult();
    }

    // This is a method to fit the generalized pareto distribution on the lower tails
    std::vector<EVTCopula::GPDResult> EVTCopula::f_FitGPD(const std::vector<std::vector<double>>& data, std::optional<double> lower, std::optional<double> upper, int min_obs, std::string method, bool lower_tail, bool double_tail) {
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
            ecdf.plotECDF();

            if (!lower.has_value()) {
                lower = ecdf(ecdf.getSortedData()[min_obs - 1]);
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

            result.shape += fit_result["shape"];
            result.scale += fit_result["scale"];
            result.threshold = *lower;
            result.excesses.insert(result.excesses.end(), excess.begin(), excess.end());
            results.push_back(result);
        }
        return results;
    }
}



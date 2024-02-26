#include "pch.h"
#include "Header.h"

void copula::getwd() {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::cout << "Current Working Directory: " << currentPath << std::endl;
}

// Extreme Value Theory Copula with Pareto marginals 

namespace copula {

    struct IndexedValue {
        double value;
        size_t index;

        IndexedValue(double val, size_t idx) : value(val), index(idx) {}
    };

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
                lower = ecdf(eval) / 2;
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
    
}



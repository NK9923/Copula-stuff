#pragma once

#define EXPORT

#ifdef EXPORT
#define dllimportexport dllexport
#else
#define dllimportexport dllimport
#endif

#include<vector>
#include<optional>
#include<string>

namespace copula {
    __declspec(dllimportexport) struct GPDResult {
        std::vector<double> excesses;
        double shape;
        double scale;
        double threshold;
    };

	__declspec(dllimportexport) GPDResult fit_GPD_PWM(const std::vector<double>& data);
	__declspec(dllimportexport) GPDResult fit_GPD_MOM(const std::vector<double>& data);
    __declspec(dllimportexport) GPDResult gpd_fit(const std::vector<double>& data, std::optional<double> lower, std::optional<double> upper, int min_obs, std::string method, bool lower_tail, bool double_tail);

}
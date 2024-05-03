#include <iostream>
#include <omp.h>
#include <DataContainers/Utils/TableUtils.h>
#include <MachineLearning/Utils/TimeSeriesForecastingUtils.h>
#include <MachineLearning/DecisionTrees/DecisionTreeRegressor.h>
#include <MachineLearning/Ensembles/RandomForestRegressor.h>
#include <MachineLearning/Ensembles/AdaBoostRegressor.h>

int main() {
    auto trainingDataset = []{
        constexpr int featuresLag = 3;
        constexpr int observationsLag = 2;

        auto table = DataContainers::TableUtils::LoadTableFromFile<double>("dataset.csv", {"Day"});
//    auto table = DataContainers::TableUtils::LoadTableFromFile<double>("daily-total-female-births.csv", {"Date"});
//    auto table = DataContainers::TableUtils::LoadTableFromFile<double>("daily-min-temperatures.csv", {"Date"});

        return MachineLearning::TimeSeriesForecastingUtils::SeriesToSupervised(table, featuresLag, observationsLag);
    }();

//    omp_set_num_threads(6);
//    auto regressor = MachineLearning::DecisionTrees::DecisionTreeRegressor(5, 3, 1.0, 6);

//    auto regressor = MachineLearning::Ensembles::RandomForestRegressor(1000, 0.75, 5, 3, 0.75);
    auto regressor = MachineLearning::Ensembles::AdaBoostRegressor(1000, 4);

    const auto mae = MachineLearning::TimeSeriesForecastingUtils::WalkForwardValidation(regressor, trainingDataset, 10);
    std::cout << "MAE: " << mae << '\n';

    return 0;
}

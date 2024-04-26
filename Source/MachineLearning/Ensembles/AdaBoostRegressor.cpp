#include "AdaBoostRegressor.h"

MachineLearning::Ensembles::AdaBoostRegressor::AdaBoostRegressor(int numOfTrees, int numOfAvailableThreads) {
    if (numOfTrees <= 0)
        throw std::invalid_argument("Number of trees is less than zero");

    m_trees.reserve(numOfTrees);
    for (int i = 0; i < numOfTrees; ++i)
        m_trees.emplace_back(2, 1, 1.0, numOfAvailableThreads);
}

void MachineLearning::Ensembles::AdaBoostRegressor::Fit(const Datasets::SupervisedLearningDatasetView<double> &dataset) {

}

std::vector<double> MachineLearning::Ensembles::AdaBoostRegressor::Predict(const std::vector<double> &features) const {
    return {};
}

DataContainers::Table<double> MachineLearning::Ensembles::AdaBoostRegressor::Predict(const DataContainers::TableView<double> &features) const {
    return {};
}

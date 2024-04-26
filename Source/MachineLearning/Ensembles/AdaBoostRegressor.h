#ifndef ADABOOSTREGRESSOR_H
#define ADABOOSTREGRESSOR_H

#include <vector>
#include <MachineLearning/RegressionModel.h>
#include <MachineLearning/DecisionTrees/DecisionTreeRegressor.h>

namespace MachineLearning::Ensembles {
    class AdaBoostRegressor final : public RegressionModel {
    public:
        explicit AdaBoostRegressor(
            int numOfTrees = 30,
            int maxDepth = 5,
            int minSampleSize = 20,
            double proportionOfFeaturesUsed = 1.0,
            int numOfAvailableThreads = 1
        );

        ~AdaBoostRegressor() override = default;

        void Fit(const Datasets::SupervisedLearningDatasetView<double> &dataset) override;

        [[nodiscard]] std::vector<double> Predict(const std::vector<double> &features) const override;
        [[nodiscard]] DataContainers::Table<double> Predict(const DataContainers::TableView<double> &features) const override;

    private:
        std::vector<DecisionTrees::DecisionTreeRegressor> m_trees;
    };
}

#endif

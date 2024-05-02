#ifndef RANDOMFORESTREGRESSOR_H
#define RANDOMFORESTREGRESSOR_H

#include <vector>
#include <MachineLearning/RegressionModel.h>
#include <MachineLearning/DecisionTrees/DecisionTreeRegressor.h>

namespace MachineLearning::Ensembles {
    class RandomForestRegressor final : public RegressionModel {
    public:
        explicit RandomForestRegressor(
            int numOfTrees = 30,
            double proportionOfRowsUsed = 1.0,
            int maxDepth = 5,
            int minSampleSize = 20,
            double proportionOfFeaturesUsed = 1.0
        );

        ~RandomForestRegressor() override = default;

        void Fit(const Datasets::SupervisedLearningDatasetView<double>& dataset) override;

        [[nodiscard]] std::vector<double> Predict(const std::vector<double>& features) const override;
        [[nodiscard]] DataContainers::Table<double> Predict(const DataContainers::TableView<double>& features) const override;

    private:
        Datasets::SupervisedLearningDatasetView<double> CreateBootstrappedDataset(const Datasets::SupervisedLearningDatasetView<double>& originalDataset) const;

    private:
        const double c_proportionOfRowsUsed;
        int m_numOfPredictedValues;
        std::vector<DecisionTrees::DecisionTreeRegressor> m_trees;
    };
}

#endif

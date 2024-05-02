#ifndef ADABOOSTREGRESSOR_H
#define ADABOOSTREGRESSOR_H

#include <vector>
#include <MachineLearning/RegressionModel.h>
#include <MachineLearning/DecisionTrees/DecisionTreeRegressor.h>

namespace MachineLearning::Ensembles {
    class AdaBoostRegressor final : public RegressionModel {
    public:
        explicit AdaBoostRegressor(
            int maxNumOfTrees = 30,
            int numOfAvailableThreads = 1
        );

        ~AdaBoostRegressor() override = default;

        void Fit(const Datasets::SupervisedLearningDatasetView<double> &dataset) override;

        [[nodiscard]] std::vector<double> Predict(const std::vector<double> &features) const override;
        [[nodiscard]] DataContainers::Table<double> Predict(const DataContainers::TableView<double> &features) const override;

    private:
        void ClearMemory();
        void ReserveMemory();

        [[nodiscard]] static std::vector<double> CalculateSampleProbabilities(const std::vector<double>& sampleWeights);
        [[nodiscard]] Datasets::SupervisedLearningDatasetView<double> CreateBootstrappedDataset(
                const Datasets::SupervisedLearningDatasetView<double>& originalDataset,
                const std::vector<double>& sampleProbabilities);

        [[nodiscard]] static std::vector<double> CalculateSampleLosses(
                const DataContainers::TableView<double>& observations,
                const DataContainers::TableView<double>& predictions);

        [[nodiscard]] std::vector<double> CalculateWeightedMedian(const DataContainers::TableView<double>& predictions) const;

        [[nodiscard]] inline static double CalculateTreeWeight(double beta);
        static void UpdateSampleWeights(std::vector<double>& sampleWeights, const std::vector<double>& sampleLosses, double beta);
        [[nodiscard]] inline static double CalculateBeta(double meanLoss);

    private:
        const int m_maxNumOfTrees;
        const int m_numOfAvailableThreads;
        int m_numOfPredictedValues;
        double m_totalTreesWeight;
        std::mt19937 m_randomGenerator;
        std::vector<DecisionTrees::DecisionTreeRegressor> m_trees;
        std::vector<double> m_treeWeights;
    };
}

#endif

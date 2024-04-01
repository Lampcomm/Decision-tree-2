#ifndef DECISION_TREE_2_DECISIONTREEREGRESSOR_H
#define DECISION_TREE_2_DECISIONTREEREGRESSOR_H

#include <MachineLearning/RegressionModel.h>
#include <memory>
#include <ranges>

namespace MachineLearning::DecisionTrees {
    class DecisionTreeRegressor : public RegressionModel {
    public:
        explicit DecisionTreeRegressor(int maxDepth = 5, int minSampleSize = 20);
        ~DecisionTreeRegressor() override = default;

        void Fit(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset) override;

        [[nodiscard]] std::vector<double> Predict(const std::vector<double>& features) override;
        [[nodiscard]] DataContainers::Table<double> Predict(const DataContainers::TableView<double>& features) override;

    private:
        struct SplittingParameters {
            int BestFeatureIndex = -1;
            double BestValue = 0.0;
        };

        struct ChildNodesTrainingDataset {
            Datasets::SupervisedLearningDatasetView<double> LeftNodeDataset;
            Datasets::SupervisedLearningDatasetView<double> RightNodeDataset;
        };

        DecisionTreeRegressor(int maxDepth, int minSampleSize, int depth);

        void FitImpl(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset, int numOfAvailableThreads);

        [[nodiscard]] static std::vector<double> GetMeanObservations(const DataContainers::TableView<double>& observations);
        [[nodiscard]] double GetMSE(const DataContainers::TableView<double>& observations) const;
        [[nodiscard]] static std::vector<double> GetMovingAverage(const std::vector<double>& column, std::vector<int> sortedColumnElemIndexes);

        [[nodiscard]] SplittingParameters GetSplittingParameters(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset) const;
        [[nodiscard]] ChildNodesTrainingDataset SplitTrainingDataset(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset) const;

        [[nodiscard]] const std::vector<double>& PredictImpl(const std::ranges::random_access_range auto& featureRange);

    private:
        const int c_maxDepth;
        const int c_minSampleSize;
        int m_curDepth = 0;
        double m_nodeMse = 0.0;
        std::vector<double> m_meanObservations;
        std::unique_ptr<DecisionTreeRegressor> m_leftNode;
        std::unique_ptr<DecisionTreeRegressor> m_rightNode;
        SplittingParameters m_splittingParameters;
    };
}

#endif

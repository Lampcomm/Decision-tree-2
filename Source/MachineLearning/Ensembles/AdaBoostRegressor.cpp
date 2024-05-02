#include "AdaBoostRegressor.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <execution>

namespace MachineLearning::Ensembles {
    AdaBoostRegressor::AdaBoostRegressor(int maxNumOfTrees, int numOfAvailableThreads)
            : m_maxNumOfTrees(maxNumOfTrees)
            , m_numOfAvailableThreads(numOfAvailableThreads)
            , m_numOfPredictedValues(0)
            , m_totalTreesWeight(0.)
            , m_randomGenerator(std::random_device{}())
    {
        if (maxNumOfTrees <= 0)
            throw std::invalid_argument("Number of trees is less than zero");
    }

    void AdaBoostRegressor::Fit(const Datasets::SupervisedLearningDatasetView<double> &dataset) {
        ClearMemory();
        ReserveMemory();

        const auto &[features, observations] = dataset;
        m_numOfPredictedValues = observations.GetNumOfColumns();
        std::vector<double> sampleWeights(features.GetNumOfRows(), 1.0);

        for (int i = 0; i < m_maxNumOfTrees; ++i) {
            const auto sampleProbabilities = CalculateSampleProbabilities(sampleWeights);

            auto &tree = m_trees.emplace_back(1, 1, 1.0, m_numOfAvailableThreads);
            tree.Fit(CreateBootstrappedDataset(dataset, sampleProbabilities));
            const auto predictions = tree.Predict(features);

            const auto sampleLosses = CalculateSampleLosses(observations, predictions);
            const double meanLoss = std::transform_reduce(std::execution::par_unseq, sampleLosses.cbegin(), sampleLosses.cend(), sampleProbabilities.cbegin(),
                                                          0.0, std::plus(),
                                                          std::multiplies());
            const double beta = CalculateBeta(meanLoss);
            m_treeWeights.push_back(CalculateTreeWeight(beta));

            if (meanLoss >= 0.5)
                break;

            UpdateSampleWeights(sampleWeights, sampleLosses, beta);
        }

        m_totalTreesWeight = std::reduce(std::execution::par_unseq, m_treeWeights.cbegin(), m_treeWeights.cend(), 0., std::plus());
    }

    std::vector<double> AdaBoostRegressor::Predict(const std::vector<double> &features) const {
        DataContainers::Table<double> predictions;
        predictions.SetNumOfColumns(m_numOfPredictedValues);

        for (const auto& tree : m_trees)
            predictions.PushBackRow(tree.Predict(features));

        return CalculateWeightedMedian(predictions);
    }

    DataContainers::Table<double> AdaBoostRegressor::Predict(const DataContainers::TableView<double> &features) const {
        DataContainers::Table<double> res;
        res.SetNumOfColumns(m_numOfPredictedValues);

        for (int rowIndex = 0; rowIndex < features.GetNumOfRows(); ++rowIndex)
            res.PushBackRow(Predict(features.GetRow(rowIndex) | RangesUtils::to_vector));

        return res;
    }

    void AdaBoostRegressor::ClearMemory() {
        m_trees.clear();
        m_treeWeights.clear();
    }

    void AdaBoostRegressor::ReserveMemory() {
        m_treeWeights.reserve(m_maxNumOfTrees);
        m_trees.reserve(m_maxNumOfTrees);
    }

    std::vector<double> AdaBoostRegressor::CalculateSampleProbabilities(const std::vector<double> &sampleWeights) {
        std::vector<double> res(sampleWeights.size());
        double sum = std::reduce(std::execution::par_unseq, sampleWeights.begin(), sampleWeights.end(), 0.0, std::plus());
        std::transform(std::execution::par_unseq, sampleWeights.cbegin(), sampleWeights.cend(), res.begin(),
                       [sum](double w){ return w / sum; });

        return res;
    }

    Datasets::SupervisedLearningDatasetView<double> AdaBoostRegressor::CreateBootstrappedDataset(
            const Datasets::SupervisedLearningDatasetView<double> &originalDataset,
            const std::vector<double> &sampleProbabilities)
    {
        const auto& [originalFeatures, originalObservations] = originalDataset;
        Datasets::SupervisedLearningDatasetView bootstrappedDataset(originalFeatures.GetViewableTable(), originalObservations.GetViewableTable());

        std::discrete_distribution<int> distribution(sampleProbabilities.begin(), sampleProbabilities.end());
        for (int i = 0; i < originalFeatures.GetNumOfRows(); ++i) {
            const auto rowIndex = distribution(m_randomGenerator);
            bootstrappedDataset.PushBackViewableRowIndex(originalFeatures.GetViewableTableRowIndex(rowIndex));
        }

        return bootstrappedDataset;
    }

    std::vector<double> AdaBoostRegressor::CalculateSampleLosses(
            const DataContainers::TableView<double>& observations,
            const DataContainers::TableView<double>& predictions)
    {
        std::vector<double> sampleLosses(observations.GetNumOfRows());

        for (int rowIndex = 0; rowIndex < observations.GetNumOfRows(); ++rowIndex) {
            const auto observationRow = observations.GetRow(rowIndex);
            const auto predictionRow = predictions.GetRow(rowIndex);
            sampleLosses[rowIndex] = std::transform_reduce(std::execution::par_unseq, observationRow.cbegin(), observationRow.cend(), predictionRow.cbegin(),
                                                           0.0, std::plus(),
                                                           [](double a, double b){ return (a - b) * (a - b); });
            sampleLosses[rowIndex] = std::sqrt(sampleLosses[rowIndex]);
        }

        std::transform(std::execution::par_unseq, sampleLosses.cbegin(), sampleLosses.cend(), sampleLosses.begin(),
                       [maxLoss = *std::ranges::max_element(sampleLosses)]
                       (double loss){ return 1. - std::exp(-loss / maxLoss); });

        return sampleLosses;
    }

    std::vector<double> AdaBoostRegressor::CalculateWeightedMedian(const DataContainers::TableView<double>& predictions) const {
        std::vector<int> sampleIndexes(predictions.GetNumOfRows());
        std::iota(sampleIndexes.begin(), sampleIndexes.end(), 0);

        std::vector<double> sampleLengthsSquares(sampleIndexes.size());
        for (int i = 0; i < std::ssize(sampleLengthsSquares); ++i) {
            const auto row = predictions.GetRow(i);
            sampleLengthsSquares[i] = std::transform_reduce(row.cbegin(), row.cend(),
                                                     0., std::plus(),
                                                     [](double val){ return val * val; });
        }

        std::ranges::sort(sampleIndexes, [&sampleLengthsSquares](int a, int b){ return sampleLengthsSquares[a] < sampleLengthsSquares[b]; });
        const auto sortedTreeWeights = sampleIndexes | std::ranges::views::transform([this](int i){ return m_treeWeights[i]; });

        int k = 0;
        double sumOfWeights = m_totalTreesWeight - sortedTreeWeights[0];

        while(sumOfWeights > m_totalTreesWeight / 2.)
            sumOfWeights -= sortedTreeWeights[++k];

        return predictions.GetRow(sampleIndexes[k]) | RangesUtils::to_vector;
    }

    double AdaBoostRegressor::CalculateTreeWeight(double beta) {
        return std::log(1. / beta);
    }

    void AdaBoostRegressor::UpdateSampleWeights(
            std::vector<double>& sampleWeights,
            const std::vector<double>& sampleLosses,
            double beta)
    {
        for (int i = 0; i < std::ssize(sampleWeights); ++i)
            sampleWeights[i] *= std::pow(beta, 1. - sampleLosses[i]);
    }

    double AdaBoostRegressor::CalculateBeta(double meanLoss) {
        return meanLoss / (1. - meanLoss);
    }
}
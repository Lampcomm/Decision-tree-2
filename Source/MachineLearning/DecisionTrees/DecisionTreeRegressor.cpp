#include "DecisionTreeRegressor.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <RangesUtils/ToVectorRangeAdaptor.h>

namespace {
    constexpr int WindowSize = 2;
}

namespace MachineLearning::DecisionTrees {
    DecisionTreeRegressor::DecisionTreeRegressor(int maxDepth, int minSampleSize)
        : c_maxDepth(maxDepth)
        , c_minSampleSize(minSampleSize)
    {}

    DecisionTreeRegressor::DecisionTreeRegressor(int maxDepth, int minSampleSize, int depth)
        : c_maxDepth(maxDepth)
        , c_minSampleSize(minSampleSize)
        , m_curDepth(depth)
    {}

    void DecisionTreeRegressor::Fit(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset)
    {
        #pragma omp parallel
        {
            #pragma omp single
            FitImpl(trainingDataset, omp_get_num_threads());
        }
    }

    void DecisionTreeRegressor::FitImpl(const Datasets::SupervisedLearningDatasetView<double> &trainingDataset, int numOfAvailableThreads) {
        const auto& [features, observations] = trainingDataset;

        m_meanObservations = GetMeanObservations(observations);
        m_nodeMse = GetMSE(observations);

        if (m_curDepth >= c_maxDepth || features.GetNumOfRows() < c_minSampleSize)
            return;

        m_splittingParameters = GetSplittingParameters(trainingDataset);
        if (m_splittingParameters.BestFeatureIndex == -1)
            return;

        auto [leftNodeDataset, rightNodeDataset] = SplitTrainingDataset(trainingDataset);

        const double threadsDistributionCoeff = (double)leftNodeDataset.Features.GetNumOfRows() / (double)rightNodeDataset.Features.GetNumOfRows();
        const int numOfLeftNodeThreads = std::round(threadsDistributionCoeff * (double)numOfAvailableThreads / (1. + threadsDistributionCoeff));
        const int numOfRightNodeThreads = numOfAvailableThreads - numOfLeftNodeThreads;

        m_leftNode.reset(new DecisionTreeRegressor(c_maxDepth, c_minSampleSize, m_curDepth + 1));
        m_rightNode.reset(new DecisionTreeRegressor(c_maxDepth, c_minSampleSize, m_curDepth + 1));

        if (numOfAvailableThreads <= 1)
        {
            m_leftNode->FitImpl(leftNodeDataset, numOfLeftNodeThreads);
            m_rightNode->FitImpl(rightNodeDataset, numOfRightNodeThreads);
            return;
        }
        #pragma omp task
        m_leftNode->FitImpl(leftNodeDataset, numOfLeftNodeThreads);

        #pragma omp task
        m_rightNode->FitImpl(rightNodeDataset, numOfRightNodeThreads);
    }

    std::vector<double> DecisionTreeRegressor::Predict(const std::vector<double>& features) {
        return PredictImpl(features);
    }

    DataContainers::Table<double> DecisionTreeRegressor::Predict(const DataContainers::TableView<double>& features) {
        DataContainers::Table<double> res;
        res.SetNumOfColumns(std::ssize(m_meanObservations));

        for (int rowIndex = 0; rowIndex < features.GetNumOfRows(); ++rowIndex)
            res.PushBackRow(PredictImpl(features.GetRow(rowIndex)));

        return res;
    }

    std::vector<double> DecisionTreeRegressor::GetMeanObservations(const DataContainers::TableView<double>& observations) {
        std::vector<double> meanObservations(observations.GetNumOfColumns());
        const double numOfRows = observations.GetNumOfRows();
        for (int columnIndex = 0; columnIndex < observations.GetNumOfColumns(); ++columnIndex) {
            const auto column = observations.GetColumn(columnIndex);
            meanObservations[columnIndex] = std::accumulate(column.begin(), column.end(), 0.0,
                                                        [numOfRows](double res, double val){ return res + val / numOfRows;});
        }

        return meanObservations;
    }

    double DecisionTreeRegressor::GetMSE(const DataContainers::TableView<double> &observations) const {
        double mse = 0.0;
        const auto n = static_cast<double>(observations.GetNumOfRows() * std::ssize(m_meanObservations));

        for (int rowIndex = 0; rowIndex < observations.GetNumOfRows(); ++rowIndex) {
            auto row = observations.GetRow(rowIndex);
            mse += std::transform_reduce(row.begin(), row.end(), m_meanObservations.begin(), 0.0,
                                         std::plus(),
                                         [n](double observation, double prediction){ return (observation - prediction) / n * (observation - prediction); } );
        }

        return mse;
    }

    DecisionTreeRegressor::SplittingParameters
    DecisionTreeRegressor::GetSplittingParameters(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset) const {
        const auto& [features, observations] = trainingDataset;

        const auto n = static_cast<double>(observations.GetNumOfColumns() * observations.GetNumOfRows());
        const double sqrtOfN = std::sqrt(n);
        double observationMeanSquareSum = 0.0;
        std::vector<double> observationsMeanSums(observations.GetNumOfColumns(), 0.0);
        for (int columnIndex = 0; columnIndex < observations.GetNumOfColumns(); ++columnIndex) {
            for (auto value : observations.GetColumn(columnIndex)) {
                observationMeanSquareSum += value / n * value;
                observationsMeanSums[columnIndex] += value / sqrtOfN;
            }
        }
        double bestMse = m_nodeMse;
        SplittingParameters res;

        for (int columnIndex = 0; columnIndex < features.GetNumOfColumns(); ++columnIndex) {
            std::vector<double> leftMeanSums(observationsMeanSums.size(), 0.0);
            std::vector<double> rightMeanSums(observationsMeanSums);
            int numOfLeftObservations = 0;
            int numOfRightObservations = observations.GetNumOfRows();

            const auto featuresColumn = features.GetColumn(columnIndex) | RangesUtils::to_vector;
            std::vector<int> rowIndexes(featuresColumn.size());
            std::iota(rowIndexes.begin(), rowIndexes.end(), 0);
            std::ranges::sort(rowIndexes,[&featuresColumn](int a, int b){ return featuresColumn[a] < featuresColumn[b]; });
            auto sortedFeaturesColumn = rowIndexes | std::views::transform(
                    [&featuresColumn](int i){ return featuresColumn[i]; });

            for (auto value : GetMovingAverage(featuresColumn, rowIndexes)) {
                for(;numOfLeftObservations < std::ssize(featuresColumn) - 1 && sortedFeaturesColumn[numOfLeftObservations] < value; ++numOfLeftObservations, --numOfRightObservations) {
                    const auto row = observations.GetRow(rowIndexes[numOfLeftObservations]);
                    for (int i = 0; i < row.GetSize(); ++i) {
                        const double val = row.At(i);
                        leftMeanSums[i] += val / sqrtOfN;
                        rightMeanSums[i] -= val / sqrtOfN;
                    }
                }
                const double leftMeanSumSquared = std::accumulate(leftMeanSums.begin(), leftMeanSums.end(), 0.0,
                                                              [numOfLeftObservations](double res, double val){ return res + val / numOfLeftObservations * val; });
                const double rightMeanSumSquared = std::accumulate(rightMeanSums.begin(), rightMeanSums.end(), 0.0,
                                                               [numOfRightObservations](double res, double val){ return res + val / numOfRightObservations * val; });
                const double newMse = observationMeanSquareSum - leftMeanSumSquared - rightMeanSumSquared;
                if (newMse < bestMse) {
                    res = {columnIndex, value};
                    bestMse = newMse;
                }
            }
        }

        return res;
    }

    DecisionTreeRegressor::ChildNodesTrainingDataset
    DecisionTreeRegressor::SplitTrainingDataset(const Datasets::SupervisedLearningDatasetView<double>& trainingDataset) const {
        const auto& [features, observations] = trainingDataset;

        Datasets::SupervisedLearningDatasetView<double> leftNodeData(features.GetViewableTable(),observations.GetViewableTable());
        auto rightNodeData = leftNodeData;
        const auto [bestFeatureIndex, bestValue] = m_splittingParameters;

        auto addRowToTables = [f = &features, o = &observations](auto& nodeData, int rowIndex){
            nodeData.Features.PushBackViewableRowIndex(f->GetViewableTableRowIndex(rowIndex));
            nodeData.Observations.PushBackViewableRowIndex(o->GetViewableTableRowIndex(rowIndex));
        };
        for (int rowIndex = 0; rowIndex < features.GetNumOfRows(); ++rowIndex)
            addRowToTables(features.At(rowIndex, bestFeatureIndex) > bestValue ? rightNodeData : leftNodeData, rowIndex);

        return {leftNodeData, rightNodeData};
    }

    const std::vector<double>& DecisionTreeRegressor::PredictImpl(const std::ranges::random_access_range auto& featureRange) {
        auto featureIterator = std::ranges::begin(featureRange);
        const auto* curNode = this;

        while (true) {
            auto [bestFeatureIndex, bestValue] = curNode->m_splittingParameters;
            if (bestFeatureIndex == -1)
                return curNode->m_meanObservations;

            curNode = featureIterator[bestFeatureIndex] > bestValue ? curNode->m_rightNode.get() : curNode->m_leftNode.get();
        }
    }

    std::vector<double> DecisionTreeRegressor::GetMovingAverage(const std::vector<double>& column, std::vector<int> sortedColumnElemIndexes) {
        const auto rubbish = std::ranges::unique(sortedColumnElemIndexes, [&column](int a, int b){ return column[a] == column[b]; });
        const auto numOfUniqueElements = std::ranges::distance(std::ranges::begin(sortedColumnElemIndexes), std::ranges::begin(rubbish));
        const auto uniqueColumnElem =   sortedColumnElemIndexes
                                        | std::views::take(numOfUniqueElements)
                                        | std::views::transform([&column](int i){ return column[i]; });

        std::vector<double> res(numOfUniqueElements - WindowSize + 1);
        for (int i = 0; i < std::ssize(res); ++i) {
            auto elemBunch = std::views::counted(std::ranges::begin(uniqueColumnElem) + i, WindowSize);
            res[i] = (std::accumulate(std::ranges::begin(elemBunch), std::ranges::end(elemBunch), 0.0,
                                      [](double res, double val){ return res + val / WindowSize; }));
        }

        return res;
    }
}
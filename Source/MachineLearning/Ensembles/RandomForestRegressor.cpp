#include "RandomForestRegressor.h"

#include <algorithm>
#include <RandomGenerators/ThreadSafeRandom.h>
#include <RangesUtils/ToVectorRangeAdaptor.h>

namespace MachineLearning::Ensembles {
    RandomForestRegressor::RandomForestRegressor(int numOfTrees, double proportionOfRowsUsed, int maxDepth, int minSampleSize,
        double proportionOfFeaturesUsed)
        : c_proportionOfRowsUsed(proportionOfRowsUsed)
        , m_numOfPredictedValues(0)
    {
        if (numOfTrees <= 0)
            throw std::invalid_argument("Number of trees is less than zero");

        if (c_proportionOfRowsUsed <= 0. || c_proportionOfRowsUsed > 1.)
            throw std::invalid_argument("Invalid proportion of rows used");

        m_trees.reserve(numOfTrees);
        for (int i = 0; i < numOfTrees; ++i)
           m_trees.emplace_back(maxDepth, minSampleSize, proportionOfFeaturesUsed);
    }

    void RandomForestRegressor::Fit(const Datasets::SupervisedLearningDatasetView<double>& dataset) {
        m_numOfPredictedValues = dataset.Observations.GetNumOfColumns();

        #pragma omp parallel for
        for (auto &tree : m_trees)
            tree.Fit(CreateBootstrappedDataset(dataset));
    }

    std::vector<double> RandomForestRegressor::Predict(const std::vector<double>& features) const {
        std::vector<double> res(m_numOfPredictedValues, 0.);
        const auto numOfTrees = static_cast<double>(m_trees.size());

        for (const auto& tree : m_trees) {
            const auto predictedValues = tree.Predict(features);
            std::ranges::transform(res, predictedValues, res.begin(), [numOfTrees](double res, double val){ return res + val / numOfTrees; });
        }

        return res;
    }

    DataContainers::Table<double> RandomForestRegressor::Predict(const DataContainers::TableView<double>& features) const {
        DataContainers::Table<double> res;
        res.SetNumOfColumns(m_numOfPredictedValues);

        for (int rowIndex = 0; rowIndex < features.GetNumOfRows(); ++rowIndex)
            res.PushBackRow(Predict(features.GetRow(rowIndex) | RangesUtils::to_vector));

        return res;
    }

    Datasets::SupervisedLearningDatasetView<double> RandomForestRegressor::CreateBootstrappedDataset(
        const Datasets::SupervisedLearningDatasetView<double> &originalDataset) const {
        const auto& [originalFeatures, originalObservations] = originalDataset;
        Datasets::SupervisedLearningDatasetView bootstrappedDataset(originalFeatures.GetViewableTable(), originalObservations.GetViewableTable());

        std::uniform_int_distribution distribution(0, originalFeatures.GetNumOfRows() - 1);
        const auto numOfBootstrappedRows = std::max(1, static_cast<int>((double)originalFeatures.GetNumOfRows() * c_proportionOfRowsUsed));

        for (int i = 0; i < numOfBootstrappedRows; ++i) {
            const auto rowIndex = distribution(RandomGenerators::ThreadSafeRandom::Generator);
            bootstrappedDataset.PushBackViewableRowIndex(originalFeatures.GetViewableTableRowIndex(rowIndex));
        }

        return bootstrappedDataset;
    }
}

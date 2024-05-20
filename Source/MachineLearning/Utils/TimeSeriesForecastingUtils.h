#ifndef DECISION_TREE_2_TIMESERIESFORECASTINGUTILS_H
#define DECISION_TREE_2_TIMESERIESFORECASTINGUTILS_H

#include <DataContainers/Table.h>
#include <MachineLearning/RegressionModel.h>
#include <MachineLearning/Datasets/SupervisedLearningDataset.h>
#include <numeric>

namespace MachineLearning::TimeSeriesForecastingUtils {
    template<class StoredType>
    [[nodiscard]] Datasets::SupervisedLearningDataset<StoredType> SeriesToSupervised(const DataContainers::Table<StoredType>& series, int featuresLag, int observationsLag) {
        if (featuresLag <= 0)
            throw std::invalid_argument("Features lag is less than or equal to zero");

        if (observationsLag <= 0)
            throw std::invalid_argument("Observations lag is less than or equal to zero");

        const int windowSize = featuresLag + observationsLag;
        if (windowSize > series.GetNumOfRows())
            throw std::invalid_argument("Observations or Features lag is too long");

        const int superviseNumOfRows = series.GetNumOfRows() - windowSize + 1;
        DataContainers::Table<StoredType> features(superviseNumOfRows, featuresLag * series.GetNumOfColumns());
        DataContainers::Table<StoredType> observations(superviseNumOfRows, observationsLag * series.GetNumOfColumns());
        for (int shift = 0; shift < superviseNumOfRows; ++shift)
            for (int seriesRowIndex = 0, superviseColumnIndex = 0; seriesRowIndex < windowSize; ++seriesRowIndex)
                for (int seriesColumnIndex = 0; seriesColumnIndex < series.GetNumOfColumns(); ++seriesColumnIndex, ++superviseColumnIndex)
                    if (superviseColumnIndex < featuresLag)
                        features.At(shift, superviseColumnIndex) = series.At(seriesRowIndex + shift, seriesColumnIndex);
                    else
                        observations.At(shift, superviseColumnIndex - featuresLag) = series.At(seriesRowIndex + shift, seriesColumnIndex);

        return {features, observations};
    }

    template<class StoredType, class FuncType>
    [[nodiscard]] double CalculateMeanError(
        const DataContainers::TableView<StoredType>& observations,
        const DataContainers::TableView<StoredType>& predictions,
        FuncType calculateElementError)
    {
        if (observations.GetNumOfRows() != predictions.GetNumOfRows()
          || observations.GetNumOfColumns() != predictions.GetNumOfColumns())
        {
            throw std::invalid_argument("Size of observations and predictions don't coincide");
        }

        double meanError = 0.0;
        const auto n = static_cast<double>(observations.GetNumOfRows() * observations.GetNumOfColumns());
        for (int rowIndex = 0; rowIndex < observations.GetNumOfRows(); ++rowIndex) {
            auto observationRow = observations.GetRow(rowIndex);
            auto predictionRow = predictions.GetRow(rowIndex);

            meanError += std::transform_reduce(observationRow.begin(), observationRow.end(), predictionRow.begin(), 0.0,
                                         std::plus(),
                                         [n, &calculateElementError](double observation, double prediction){ return calculateElementError(observation, prediction) / n; });
        }

        return meanError;
    }

    template<class StoredType>
    [[nodiscard]] double CalculateMAE(const DataContainers::TableView<StoredType>& observations, const DataContainers::TableView<StoredType>& predictions) {
        return CalculateMeanError(observations, predictions,
            [](double observation, double prediction){ return std::abs(observation - prediction); });
    }

    template<class StoredType>
    [[nodiscard]] double CalculateMRPE(const DataContainers::TableView<StoredType>& observations, const DataContainers::TableView<StoredType>& predictions) {
        return CalculateMeanError(observations, predictions,
            [](double observation, double prediction){ return std::abs(observation - prediction) / observation; }) * 100;
    }

    [[nodiscard]] double WalkForwardValidation(RegressionModel& regressor, const Datasets::SupervisedLearningDataset<double>& dataset, int numOfTests);
}

#endif

#include "TimeSeriesForecastingUtils.h"
#include <MachineLearning/Utils/SupervisedLearningUtils.h>
#include <RangesUtils/ToVectorRangeAdaptor.h>
#include <DataContainers/Utils/TableUtils.h>
#include <iostream>

#ifdef PrintTrainingTime
#include <chrono>
#endif

namespace MachineLearning::TimeSeriesForecastingUtils {
    double WalkForwardValidation(MachineLearning::RegressionModel &regressor, const Datasets::SupervisedLearningDataset<double>& dataset, int numOfTests) {
        if (numOfTests <= 0)
            throw std::invalid_argument("Number of tests is less than or equal to zero");

        auto&& [trainingDataset, testDataset] = SupervisedLearningUtils::SplitDatasetIntoTestAndTraining(dataset,1.f - (float)numOfTests / (float)dataset.Features.GetNumOfRows());
        DataContainers::Table<double> predictions(0, trainingDataset.Observations.GetNumOfColumns());

    #ifdef PrintTrainingTime
        auto totalTrainingTime = std::chrono::microseconds(0);
    #endif

        for (int testNum = 0; testNum < numOfTests; ++testNum) {
        #ifdef PrintTrainingTime
            auto start = std::chrono::high_resolution_clock::now();
            regressor.Fit(trainingDataset);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::microseconds oneTrainingTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            totalTrainingTime += oneTrainingTime;
        #else
            regressor.Fit(trainingDataset);
        #endif
            predictions.PushBackRow(regressor.Predict(testDataset.Features.GetRow(testNum) | RangesUtils::to_vector));
            trainingDataset.PushBackViewableRowIndex(testDataset.Features.GetViewableTableRowIndex(testNum));

            std::cout   << ">expected=" << testDataset.Observations.GetRow(testNum)
                        << " predicted=" << predictions.GetRow(testNum) << '\n';
        #ifdef PrintTrainingTime
            std::cout << "Time of one training: " << (double)oneTrainingTime.count() * 1e-6 << " s.\n\n";
        #endif
        }

#ifdef PrintTrainingTime
    std::cout << "Total training time: " << (double)totalTrainingTime.count() * 1e-6 << " s.\n";
    std::cout << "Mean training time: " << (double)totalTrainingTime.count() / (double)numOfTests * 1e-6 << " s.\n";
#endif

        return CalcMAE(testDataset.Observations, DataContainers::TableView(predictions));
    }

}

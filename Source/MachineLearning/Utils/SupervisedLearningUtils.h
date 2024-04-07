#ifndef DECISION_TREE_2_SUPERVISEDLEARNINGUTILS_H
#define DECISION_TREE_2_SUPERVISEDLEARNINGUTILS_H

#include <MachineLearning/Datasets/SupervisedLearningDatasetView.h>

namespace MachineLearning::SupervisedLearningUtils {
    template<class StoredType>
    struct TrainingAndTestDatasets {
        Datasets::SupervisedLearningDatasetView<StoredType> TrainingDataset;
        Datasets::SupervisedLearningDatasetView<StoredType> TestDataset;
    };

    template<class StoredType>
    TrainingAndTestDatasets<StoredType>
    SplitDatasetIntoTestAndTraining(const Datasets::SupervisedLearningDataset<StoredType>& dataset, double proportionOfTrainingData) {
        if (proportionOfTrainingData <= 0 || proportionOfTrainingData > 1.0)
            throw std::invalid_argument("Invalid proportion of training data");

        Datasets::SupervisedLearningDatasetView<StoredType> trainingDataset(dataset);
        Datasets::SupervisedLearningDatasetView<StoredType> testDataset(dataset);

        const int trainingDatasetSize = dataset.Features.GetNumOfRows() * proportionOfTrainingData;
        for (int rowIndex = 0; rowIndex < trainingDatasetSize; ++rowIndex)
            trainingDataset.PushBackViewableRowIndex(rowIndex);

        for (int rowIndex = trainingDatasetSize; rowIndex < dataset.Features.GetNumOfRows(); ++rowIndex)
            testDataset.PushBackViewableRowIndex(rowIndex);

        return {trainingDataset, testDataset};
    }
}

#endif

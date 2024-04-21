#ifndef DECISION_TREE_2_SUPERVISEDLEARNINGDATASETVIEW_H
#define DECISION_TREE_2_SUPERVISEDLEARNINGDATASETVIEW_H

#include <MachineLearning/Datasets/SupervisedLearningDataset.h>
#include <DataContainers/TableView.h>

namespace MachineLearning::Datasets {
    template<class StoredType>
    struct SupervisedLearningDatasetView {
        SupervisedLearningDatasetView(const DataContainers::Table<StoredType>& features,
                                      const DataContainers::Table<StoredType>& observations)
            : Features(features)
            , Observations(observations)
        {}

        SupervisedLearningDatasetView(const DataContainers::TableView<StoredType>& features,
                                      const DataContainers::TableView<StoredType>& observations)
                : Features(features)
                , Observations(observations)
        {}

        explicit SupervisedLearningDatasetView(const SupervisedLearningDataset<StoredType>& dataset)
                : Features(dataset.Features)
                , Observations(dataset.Observations)
        {}
        SupervisedLearningDatasetView(SupervisedLearningDataset<StoredType>&& dataset) = delete;

        void PushBackViewableRowIndex(int rowIndex) {
            Features.PushBackViewableRowIndex(rowIndex);
            Observations.PushBackViewableRowIndex(rowIndex);
        }

        DataContainers::TableView<StoredType> Features;
        DataContainers::TableView<StoredType> Observations;
    };
}

#endif

#ifndef DECISION_TREE_2_SUPERVISEDLEARNINGDATASET_H
#define DECISION_TREE_2_SUPERVISEDLEARNINGDATASET_H

#include <DataContainers/Table.h>

namespace MachineLearning::Datasets {
    template<class StoredType>
    struct SupervisedLearningDataset {
        DataContainers::Table<StoredType> Features;
        DataContainers::Table<StoredType> Observations;
    };
}

#endif

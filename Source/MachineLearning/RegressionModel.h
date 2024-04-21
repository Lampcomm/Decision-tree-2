#ifndef DECISION_TREE_2_REGRESSIONMODEL_H
#define DECISION_TREE_2_REGRESSIONMODEL_H

#include <vector>
#include <MachineLearning/Datasets/SupervisedLearningDatasetView.h>

namespace MachineLearning {
    class RegressionModel {
    public:
        virtual void Fit(const Datasets::SupervisedLearningDatasetView<double>& dataset) = 0;

       [[nodiscard]] virtual std::vector<double> Predict(const std::vector<double>& features) const = 0;
       [[nodiscard]] virtual DataContainers::Table<double> Predict(const DataContainers::TableView<double>& features) const = 0;

        virtual ~RegressionModel() = 0;
    };
}

#endif

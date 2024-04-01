#ifndef DECISION_TREE_2_TABLECOLUMN_H
#define DECISION_TREE_2_TABLECOLUMN_H

#include <DataContainers/TableTraits/AbstractTableRowAndColumn.h>

namespace DataContainers {
    template<class T>
    class Table;

    template<class T>
    class TableView;
}

namespace DataContainers::TableTraits {
    template<class TableType, class ValueType>
    class TableColumn : public AbstractTableRowAndColumn<ValueType> {
        template<class T>
        friend class DataContainers::Table;

        template<class T>
        friend class DataContainers::TableView;

    public:
        [[nodiscard]] int GetSize() const override { return m_table->GetNumOfRows(); }
        [[nodiscard]] ValueType& At(int i) override { return m_table->At(i, m_columnIndex); }
        [[nodiscard]] const ValueType& At(int i) const override { return m_table->At(i, m_columnIndex); }

   private:
        TableColumn(int columnIndex, TableType& table)
                : m_columnIndex(columnIndex), m_table(&table) {}

    private:
        int m_columnIndex = 0;
        TableType* m_table;
    };
}

#endif

#ifndef DECISION_TREE_2_TABLEROW_H
#define DECISION_TREE_2_TABLEROW_H

#include <DataContainers/TableTraits/AbstractTableRowAndColumn.h>

namespace DataContainers {
    template<class T>
    class Table;

    template<class T>
    class TableView;
}

namespace DataContainers::TableTraits {
    template<class TableType, class ValueType>
    class TableRow : public AbstractTableRowAndColumn<ValueType> {
        template<class T>
        friend class DataContainers::Table;

        template<class T>
        friend class DataContainers::TableView;

    public:
        [[nodiscard]] int GetSize() const override { return m_table->GetNumOfColumns(); }
        [[nodiscard]] ValueType& At(int i) override { return m_table->At(m_rowIndex, i); }
        [[nodiscard]] const ValueType& At(int i) const override { return m_table->At(m_rowIndex, i); }

    private:
        TableRow(int rowIndex, TableType& table)
                : m_rowIndex(rowIndex), m_table(&table) {}
    private:
        int m_rowIndex = 0;
        TableType* m_table;
    };
}

#endif

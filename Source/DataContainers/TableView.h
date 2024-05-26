#ifndef DECISION_TREE_2_TABLEVIEW_H
#define DECISION_TREE_2_TABLEVIEW_H

#include <vector>
#include <DataContainers/Table.h>
#include <DataContainers/TableTraits/TableRow.h>
#include <DataContainers/TableTraits/TableColumn.h>
#include <DataContainers/TableTraits/TableTag.h>

namespace DataContainers {
    template<class StoredType>
    class TableView : TableTraits::TableTag {
    public:
        TableView(const Table<StoredType>& table)
            : m_viewableTable(&table){}

        TableView(Table<StoredType>&& table) = delete;

        const Table<StoredType>& GetViewableTable() const { return *m_viewableTable; };

        [[nodiscard]] const StoredType& At(int rowIndex, int columnIndex) const {
            return m_viewableTable->At(GetViewableTableRowIndex(rowIndex), GetViewableTableColumnIndex(columnIndex));
        }

        [[nodiscard]] int GetNumOfRows() const { return m_viewableRows.empty() ? m_viewableTable->GetNumOfRows() : m_viewableRows.size(); }
        [[nodiscard]] int GetNumOfColumns() const { return  m_viewableColumns.empty() ? m_viewableTable->GetNumOfColumns() : m_viewableColumns.size(); }

        void PushBackViewableRowIndex(int viewableTableRowIndex) {
            if (viewableTableRowIndex < 0 || viewableTableRowIndex >= m_viewableTable->GetNumOfRows())
                throw std::out_of_range("Viewable table row index is out of range");
            m_viewableRows.push_back(viewableTableRowIndex);
        }

        template<std::weakly_incrementable Out>
        void GetRow(int rowIndex, Out&& outRange) const {
            for (int columnIndex = 0; columnIndex < GetNumOfColumns(); ++columnIndex, ++outRange)
                *outRange = At(rowIndex, columnIndex);
        }

        auto GetRow(int rowIndex) const { return TableTraits::TableRow(rowIndex, *this); }

        void PushBackViewableColumnIndex(int viewableTableColumnIndex) {
            if (viewableTableColumnIndex < 0 || viewableTableColumnIndex >= m_viewableTable->GetNumOfColumns())
                throw std::out_of_range("Viewable table column index is out of range");
            m_viewableColumns.push_back(viewableTableColumnIndex);
        }

        template<std::weakly_incrementable Out>
        void GetColumn(int columnIndex, Out&& outRange) const {
            for (int rowIndex = 0; rowIndex < GetNumOfRows(); ++rowIndex, ++outRange)
                *outRange = At(rowIndex, columnIndex);
        }

        auto GetColumn(int columnIndex) const { return TableTraits::TableColumn(columnIndex, *this); }

        void ClearViewableRows() { m_viewableRows.clear(); }
        void ClearViewableColumns() { m_viewableColumns.clear(); }

        [[nodiscard]] int GetViewableTableRowIndex(int viewRowIndex) const {
            return m_viewableRows.empty() ? viewRowIndex : m_viewableRows[viewRowIndex];
        }

        [[nodiscard]] int GetViewableTableColumnIndex(int viewColumnIndex) const {
            return m_viewableColumns.empty() ? viewColumnIndex : m_viewableColumns[viewColumnIndex];
        }

    private:
        std::vector<int> m_viewableRows;
        std::vector<int> m_viewableColumns;
        const Table<StoredType>* m_viewableTable = nullptr;
    };

}

#endif

#ifndef DECISION_TREE_2_TABLE_H
#define DECISION_TREE_2_TABLE_H

#include <vector>
#include <ostream>
#include <DataContainers/TableTraits/TableRow.h>
#include <DataContainers/TableTraits/TableColumn.h>
#include <DataContainers/TableTraits/TableTag.h>

namespace DataContainers {
    template<class StoredType>
    class Table : TableTraits::TableTag {
    public:
        Table() = default;
        Table(int numOfRows, int numOfColumns) {
            SetNumOfRows(numOfRows);
            SetNumOfColumns(numOfColumns);
        }

        [[nodiscard]] int GetNumOfRows() const { return m_numOfRows; };
        void SetNumOfRows(int numOfRows) {
            if (numOfRows < 0)
                throw std::invalid_argument("Number of rows is less than zero");

            if (m_numOfRows > numOfRows)
                RemoveRowsFromEnd(m_numOfRows - numOfRows);
            else if (m_numOfRows < numOfRows)
                AddRowsFromEnd(numOfRows - m_numOfRows);

            m_numOfRows = numOfRows;
        }

        [[nodiscard]] int GetNumOfColumns() const { return m_numOfColumns; };
        void SetNumOfColumns(int numOfColumns) {
            if (numOfColumns < 0)
                throw std::invalid_argument("Number of columns is less than zero");

            if (m_numOfColumns > numOfColumns)
                RemoveColumnsFromEnd(m_numOfColumns - numOfColumns);
            else if (m_numOfColumns < numOfColumns)
                AddColumnsFromEnd(numOfColumns - m_numOfColumns);

            m_numOfColumns = numOfColumns;
        }

        template<std::weakly_incrementable Out>
        void GetRow(int rowIndex, Out&& outRange) const {
            RowIndexCheck(rowIndex);

            for (int columnIndex = 0; columnIndex < m_numOfColumns; ++columnIndex, ++outRange)
                *outRange = m_tableData[GetElementIndexInTableData(rowIndex, columnIndex)];
        }

        auto GetRow(int rowIndex) { return TableTraits::TableRow(rowIndex, *this); }
        auto GetRow(int rowIndex) const { return TableTraits::TableRow(rowIndex, *this); }

        template<std::ranges::input_range Range>
        void PushBackRow(Range&& inputRange) {
            auto inputRangeBegin = std::ranges::begin(inputRange);
            auto rangeSize = std::distance(inputRangeBegin, std::ranges::end(inputRange));
            if (rangeSize != m_numOfColumns)
                throw std::invalid_argument("Range size is not equal to number of columns");

            SetNumOfRows(m_numOfRows + 1);
            for (int columnIndex = 0; columnIndex < m_numOfColumns; ++columnIndex, ++inputRangeBegin)
                m_tableData[GetElementIndexInTableData(m_numOfRows - 1, columnIndex)] = *inputRangeBegin;
        }

        template<std::weakly_incrementable Out>
        void GetColumn(int columnIndex, Out&& outRange) const {
            ColumnIndexCheck(columnIndex);

            for (int rowIndex = 0; rowIndex < m_numOfRows; ++columnIndex, ++outRange)
                *outRange = m_tableData[GetElementIndexInTableData(rowIndex, columnIndex)];
        }

        auto GetColumn(int columnIndex) { return TableTraits::TableColumn(columnIndex, *this); }
        auto GetColumn(int columnIndex) const { return TableTraits::TableColumn(columnIndex, *this); }

        template<std::ranges::input_range Range>
        void PushBackColumn(Range&& inputRange) {
            auto inputRangeBegin = std::ranges::begin(inputRange);
            auto rangeSize = std::distance(inputRangeBegin, std::end(inputRange));
            if (rangeSize != m_numOfRows)
                throw std::invalid_argument("Range size is not equal to number of rows");

            SetNumOfColumns(m_numOfColumns + 1);
            for (int rowIndex = 0; rowIndex < m_numOfRows; ++rowIndex, ++inputRangeBegin)
                m_tableData[GetElementIndexInTableData(rowIndex, m_numOfColumns - 1)] = *inputRangeBegin;
        }

        [[nodiscard]] StoredType& At(int rowIndex, int columnIndex) {
            RowIndexCheck(rowIndex);
            ColumnIndexCheck(columnIndex);

            return m_tableData[GetElementIndexInTableData(rowIndex, columnIndex)];
        }

        [[nodiscard]] const StoredType& At(int rowIndex, int columnIndex) const {
            RowIndexCheck(rowIndex);
            ColumnIndexCheck(columnIndex);

            return m_tableData[GetElementIndexInTableData(rowIndex, columnIndex)];
        }

    private:
        inline void RowIndexCheck(int rowIndex) const {
            if (rowIndex < 0 || rowIndex >= m_numOfRows)
                throw std::out_of_range("Row index out of range");
        }

        inline void ColumnIndexCheck(int columnIndex) const {
            if (columnIndex < 0 || columnIndex >= m_numOfColumns)
                throw std::out_of_range("Column index out of range");
        }

        [[nodiscard]] inline int GetElementIndexInTableData(int rowIndex, int columnIndex) const { return columnIndex * m_numOfRows + rowIndex; }

        void RemoveRowsFromEnd(int numOfRemoveRows) {
            auto endRowIt = std::next(m_tableData.begin(), m_numOfRows);
            for (int i = 0; i < m_numOfColumns; ++i, endRowIt += m_numOfRows)
                endRowIt = m_tableData.erase(std::prev(endRowIt, numOfRemoveRows), endRowIt);
        }

        void AddRowsFromEnd(int numOfAddRows) {
            auto endRowIt = std::next(m_tableData.begin(), m_numOfRows);
            for (int i = 0; i < m_numOfColumns; ++i, endRowIt += m_numOfRows + numOfAddRows)
                endRowIt = m_tableData.insert(endRowIt, numOfAddRows, 0.0);
        }

        void RemoveColumnsFromEnd(int numOfRemoveColumns) {
            m_tableData.erase(std::prev(m_tableData.end(), m_numOfRows * numOfRemoveColumns), m_tableData.end());
        }
        void AddColumnsFromEnd(int numOfAddColumns) {
            m_tableData.resize(m_numOfRows * (m_numOfColumns + numOfAddColumns));
        }

    private:
        int m_numOfRows = 0;                   ///< Number of rows in the table
        int m_numOfColumns = 0;                ///< Number of columns in the table
        std::vector<StoredType> m_tableData;   ///< Table data
    };
}

#endif

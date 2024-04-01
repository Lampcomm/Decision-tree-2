#ifndef DECISION_TREE_2_TABLEUTILS_H
#define DECISION_TREE_2_TABLEUTILS_H

#include <DataContainers/Table.h>
#include <string>
#include <unordered_set>
#include <RangesUtils/ToVectorRangeAdaptor.h>
#include <StringUtils/Trim.h>
#include <sstream>
#include <fstream>
#include <ranges>
#include <type_traits>

namespace DataContainers::TableUtils {
    template<class StoredType>
    Table<StoredType> LoadTableFromFile(const std::string &fileName, const std::unordered_set<std::string> &ignoredColumns, char delim = ',') {
        std::ifstream inp(fileName);
        if (!inp.is_open())
            throw std::invalid_argument("Failed to open file");

        const auto columnNames = [&inp, delim](){
            std::string columnNamesStr;
            std::getline(inp, columnNamesStr);
            return columnNamesStr
                   | std::views::split(delim)
                   | std::views::transform([](const std::span<char> &sp) { return StringUtils::TrimCopy(std::string(sp.begin(), sp.end())); })
                   | RangesUtils::to_vector;
        }();

        if (std::ssize(ignoredColumns) >= std::ssize(columnNames))
            throw std::invalid_argument("The number of columns to be ignored is greater or equal than the total number of columns");

        Table<StoredType> table;
        table.SetNumOfColumns(std::ssize(columnNames) - std::ssize(ignoredColumns));

        for (std::string line; getline(inp, line);)
        {
            if (line.empty())
                continue;

            auto data = line
                        | std::views::split(delim)
                        | std::views::filter([index = 0, &ignoredColumns, &columnNames](auto&&) mutable { return !ignoredColumns.contains(columnNames[index++]); })
                        | std::views::transform([](const std::span<char>& rawObj) {
                            std::stringstream ss;
                            ss << std::string_view(rawObj);
                            StoredType obj;
                            ss >> obj;
                            return obj;
                        });
            table.PushBackRow(data);
        }

        return table;
    }

    template<class StoredType, std::invocable<int, int> Func>
    Table<StoredType> LoadTableFromArray(StoredType* array, int numOfRows, int numOfColumns, Func getElementIndexInArray) {
        Table<StoredType> table(numOfRows, numOfColumns);

        for (int i = 0; i < numOfRows; ++i)
            for (int j = 0; j < numOfColumns; ++j)
                table.At(i, j) = array[getElementIndexInArray(i, j)];

        return table;
    }
}

template<class TableType>
requires std::is_base_of_v<DataContainers::TableTraits::TableTag, TableType>
std::ostream& operator<<(std::ostream &out, TableType&& table)
{
    for (int i = 0; i < table.GetNumOfRows(); ++i)
    {
        for (int j = 0; j < table.GetNumOfColumns(); ++j)
            out << table.At(i, j) << '\t';
        out << '\n';
    }
    return out;
}

template<class StoredType>
std::ostream& operator<<(std::ostream &out, const DataContainers::TableTraits::AbstractTableRowAndColumn<StoredType>& range) {
    for (auto&& elm : range)
        out << elm << " ";
    return out;
}

#endif

#ifndef DECISION_TREE_2_TRIM_H
#define DECISION_TREE_2_TRIM_H

#include <string>

namespace StringUtils {
    void Trim(std::string& str);
    std::string TrimCopy(std::string str);

    void TrimLeft(std::string& str);
    std::string TrimLeftCopy(std::string str);

    void TrimRight(std::string& str);
    std::string TrimRightCopy(std::string str);
}

#endif

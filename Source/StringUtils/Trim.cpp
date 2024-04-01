#include "Trim.h"
#include <algorithm>

namespace StringUtils {
    void Trim(std::string &str) {
        TrimLeft(str);
        TrimRight(str);
    }

    std::string TrimCopy(std::string str) {
        Trim(str);
        return str;
    }

    void TrimLeft(std::string &str) {
        auto firstChar = std::find_if(str.begin(), str.end(),
                                      [](auto c) { return !std::isspace(c); });
        str.erase(str.begin(), firstChar);
    }

    std::string TrimLeftCopy(std::string str) {
        TrimLeft(str);
        return str;
    }

    void TrimRight(std::string &str) {
        auto lastChar = std::find_if(str.rbegin(), str.rend(),
                                     [](auto c) { return !std::isspace(c); });
        str.erase(lastChar.base(), str.end());
    }

    std::string TrimRightCopy(std::string str) {
        TrimRight(str);
        return str;
    }
}

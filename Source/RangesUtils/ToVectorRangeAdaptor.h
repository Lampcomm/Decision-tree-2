#ifndef DECISION_TREE_2_TOVECTORRANGEADAPTOR_H
#define DECISION_TREE_2_TOVECTORRANGEADAPTOR_H

#include <ranges>
#include <vector>

namespace RangesUtils {
    struct to_vector_closure {
        template<std::ranges::viewable_range Range, typename Self>
        friend constexpr auto operator|(Range &&r, Self &&s) { return std::forward<Self>(s)(std::forward<Range>(r)); }
    };

    struct to_vector_adapter : to_vector_closure {
        template<std::ranges::viewable_range Range>
        constexpr auto operator()(Range &&r) const {
            auto r_common = std::views::common(std::forward<Range>(r));
            return std::vector(r_common.begin(), r_common.end());
        }
    };

    constexpr inline to_vector_adapter to_vector;
}

#endif

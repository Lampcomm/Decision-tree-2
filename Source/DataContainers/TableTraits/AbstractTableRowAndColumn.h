#ifndef DECISION_TREE_2_ABSTRACTTABLEROWANDCOLUMN_H
#define DECISION_TREE_2_ABSTRACTTABLEROWANDCOLUMN_H

#include <iterator>
#include <iterator>

namespace DataContainers::TableTraits {
    template<class ValueType>
    class AbstractTableRowAndColumn {
    public:
        template<class IterRefType, class OwnerType>
        struct IteratorTemplate {
            using iterator_category = std::random_access_iterator_tag;
            using difference_type   = int;
            using value_type        = std::remove_const_t<IterRefType>;
            using pointer           = IterRefType*;
            using reference         = IterRefType&;

            IteratorTemplate() = default;
            IteratorTemplate(const IteratorTemplate&) = default;
//            IteratorTemplate(IteratorTemplate&&) = default;
            IteratorTemplate(int index, OwnerType* owner)
                    : m_curIndex(index) , m_owner(owner) {}

            reference operator*() const { return m_owner->At(m_curIndex); }
            pointer operator->() const { return &m_owner->At(m_curIndex); }
            reference operator[](difference_type i) const { return m_owner->At(i); };

            IteratorTemplate& operator++() { ++m_curIndex; return *this; }
            IteratorTemplate operator++(int) { auto tmp = *this; ++(*this); return tmp; }
            IteratorTemplate& operator+=(difference_type n) { m_curIndex += n; return *this; }
            IteratorTemplate& operator--() { --m_curIndex; return *this; }
            IteratorTemplate operator--(int) { auto tmp = *this; --(*this); return tmp; }
            IteratorTemplate& operator-=(difference_type n) { m_curIndex -= n; return *this; }

            friend bool operator==(const IteratorTemplate& a, const IteratorTemplate& b) {
                return a.m_curIndex == b.m_curIndex;
            }
            friend bool operator!=(const IteratorTemplate& a, const IteratorTemplate& b) { return !(a == b); }
            friend bool operator<(const IteratorTemplate& a, const IteratorTemplate& b) { return a.m_curIndex < b.m_curIndex; }
            friend bool operator>(const IteratorTemplate& a, const IteratorTemplate& b) { return a.m_curIndex > b.m_curIndex; }
            friend bool operator<=(const IteratorTemplate& a, const IteratorTemplate& b) { return a.m_curIndex <= b.m_curIndex; }
            friend bool operator>=(const IteratorTemplate& a, const IteratorTemplate& b) { return a.m_curIndex >= b.m_curIndex; }

            friend IteratorTemplate operator+(const IteratorTemplate& a, difference_type n) { return Iterator(a.m_curIndex + n, a.m_owner); }
            friend IteratorTemplate operator+(difference_type n, const IteratorTemplate& a) { return a - n; }
            friend IteratorTemplate operator-(const IteratorTemplate& a, difference_type n) { return Iterator(a.m_curIndex - n, a.m_owner); }
            friend difference_type operator-(const IteratorTemplate& a, const IteratorTemplate& b) { return a.m_curIndex - b.m_curIndex; }

        private:
            int m_curIndex = -1;
            OwnerType* m_owner = nullptr;
        };

        using Iterator = IteratorTemplate<ValueType, AbstractTableRowAndColumn>;
        using ConstIterator = IteratorTemplate<const ValueType, const AbstractTableRowAndColumn>;

        [[nodiscard]] virtual int GetSize() const = 0;
        [[nodiscard]] virtual ValueType& At(int i) = 0;
        [[nodiscard]] virtual const ValueType& At(int i) const = 0;

        auto begin() { return Iterator(0, this); }
        auto end() { return Iterator(GetSize(), this); };
        auto begin() const { return ConstIterator(0, this); }
        auto end() const { return ConstIterator(GetSize(), this); };

        auto cbegin() const { return ConstIterator(0, this); }
        auto cend() const { return ConstIterator(GetSize(), this); };
    };
}

#endif
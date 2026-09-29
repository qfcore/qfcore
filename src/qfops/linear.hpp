// Exact sparse F2 homology with representatives, kept simple on purpose.
// The same code is used for QF and GUDHI-backed chains. Fill limits bound the
// growth of sparse columns.
#pragma once
#include "core.hpp"
#include <map>
namespace ops {
using qf::i64;
using qf::u32;
using V = std::vector<u32>;
using M = std::vector<V>;
inline void check_size(std::size_t n, std::size_t limit) {
    if (n > limit)
        throw std::length_error("sparse fill limit exceeded");
}
inline V add(const V &a, const V &b, std::size_t limit) {
    V out;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    check_size(out.size(), limit);
    return out;
}
inline V parity(V v) {
    std::sort(v.begin(), v.end());
    V out;
    for (std::size_t i = 0; i < v.size();) {
        std::size_t j = i + 1;
        while (j < v.size() && v[j] == v[i])
            ++j;
        if ((j - i) % 2)
            out.push_back(v[i]);
        i = j;
    }
    return out;
}
inline V col(const qf::Boundary &b, std::size_t j) {
    return V(b.idx.begin() + b.off.at(j), b.idx.begin() + b.off.at(j + 1));
}
inline V linear_apply(const qf::Boundary &b, const V &v, std::size_t limit) {
    V x;
    for (u32 j : v)
        x = add(x, col(b, j), limit);
    return x;
}
inline V linear_apply(const M &m, const V &v, std::size_t limit) {
    V x;
    for (u32 j : v)
        x = add(x, m.at(j), limit);
    return x;
}
inline V linear_apply(const std::vector<i64> &m, const V &v) {
    V x;
    for (u32 j : v)
        if (m.at(j) >= 0)
            x.push_back(static_cast<u32>(m[j]));
    return parity(std::move(x));
}
struct Entry {
    V vector, tag;
};
struct Span {
    std::map<u32, Entry> pivots;
    std::size_t entries = 0, limit;
    explicit Span(std::size_t lim) : limit(lim) {}
    std::pair<V, V> reduce(V x, V tag = {}) const {
        while (!x.empty()) {
            auto it = pivots.find(x.back());
            if (it == pivots.end())
                break;
            x = add(x, it->second.vector, limit);
            tag = add(tag, it->second.tag, limit);
        }
        return {std::move(x), std::move(tag)};
    }
    bool insert(V x, V tag = {}) {
        auto r = reduce(std::move(x), std::move(tag));
        if (r.first.empty())
            return false;
        store(std::move(r.first), std::move(r.second));
        return true;
    }
    void store(V x, V tag) {
        if (x.empty())
            throw std::logic_error("empty pivot");
        entries += x.size() + tag.size();
        check_size(entries, limit);
        u32 p = x.back();
        if (!pivots.emplace(p, Entry{std::move(x), std::move(tag)}).second)
            throw std::logic_error("duplicate pivot");
    }
};
// Kernel generators of a matrix, with source column transformations retained.
inline M kernel(const M &m, std::size_t limit) {
    Span s(limit);
    M out;
    std::size_t used = 0;
    for (std::size_t j = 0; j < m.size(); ++j) {
        auto r = s.reduce(m[j], V{static_cast<u32>(j)});
        if (r.first.empty()) {
            used += r.second.size();
            check_size(used, limit);
            out.push_back(std::move(r.second));
        } else
            s.store(std::move(r.first), std::move(r.second));
    }
    return out;
}
inline std::size_t rank(const M &m, std::size_t limit) {
    Span s(limit);
    for (const auto &v : m)
        s.insert(v);
    return s.pivots.size();
}
struct Degree {
    Span span;
    M cycles;
    std::size_t boundary_rank = 0;
    explicit Degree(std::size_t lim) : span(lim) {}
};
struct Homology {
    qf::CPtr chain;
    std::vector<Degree> degrees;
    std::size_t limit;
    Homology(qf::CPtr c, std::size_t lim) : chain(std::move(c)), limit(lim) {
        if (!lim)
            throw std::invalid_argument("fill limit must be positive");
        for (std::size_t d = 0; d < chain->counts.size(); ++d) {
            degrees.emplace_back(limit);
            auto &h = degrees.back();
            if (d + 1 < chain->counts.size()) {
                auto &b = chain->boundaries[d + 1];
                for (i64 j = 0; j < b.cols(); ++j)
                    h.span.insert(col(b, j));
            }
            h.boundary_rank = h.span.pivots.size();
            Span lower(limit);
            std::size_t cycle_entries = 0;
            auto &b = chain->boundaries[d];
            for (i64 j = 0; j < b.cols(); ++j) {
                auto r = lower.reduce(col(b, j), V{static_cast<u32>(j)});
                if (!r.first.empty()) {
                    lower.store(std::move(r.first), std::move(r.second));
                    continue;
                }
                // Normalize a cycle modulo previously inserted boundaries and homology cycles.
                auto z = h.span.reduce(std::move(r.second)).first;
                if (!z.empty()) {
                    u32 id = h.cycles.size();
                    cycle_entries += z.size();
                    check_size(cycle_entries, limit);
                    h.cycles.push_back(z);
                    h.span.store(std::move(z), V{id});
                }
            }
        }
    }
    std::vector<i64> betti() const {
        std::vector<i64> b;
        for (const auto &d : degrees)
            b.push_back(d.cycles.size());
        return b;
    }
    std::vector<M> cycles() const {
        std::vector<M> v;
        for (const auto &d : degrees)
            v.push_back(d.cycles);
        return v;
    }
    std::size_t payload() const {
        std::size_t n = chain->payload_bytes();
        for (const auto &d : degrees) {
            n += d.span.entries * sizeof(u32);
            for (const auto &z : d.cycles)
                n += z.size() * sizeof(u32);
        }
        return n;
    }
    V coordinates(std::size_t d, V z) const {
        auto r = degrees.at(d).span.reduce(std::move(z));
        if (!r.first.empty())
            throw std::logic_error("image is not in the target cycle span");
        return r.second;
    }
};
using Images = std::vector<std::vector<i64>>;
inline bool chain_map_valid(const qf::ChainComplex &a, const qf::ChainComplex &b, const Images &f,
                            std::size_t limit) {
    if (a.counts.size() != b.counts.size() || f.size() != a.counts.size())
        return false;
    for (std::size_t d = 0; d < f.size(); ++d) {
        if (f[d].size() != static_cast<std::size_t>(a.counts[d]))
            return false;
        for (i64 j : f[d])
            if (j < -1 || j >= b.counts[d])
                return false;
    }
    for (std::size_t d = 1; d < f.size(); ++d)
        for (std::size_t j = 0; j < f[d].size(); ++j) {
            V l = f[d][j] < 0 ? V{} : col(b.boundaries[d], f[d][j]);
            V r = linear_apply(f[d - 1], col(a.boundaries[d], j));
            if (l != r)
                return false;
        }
    return true;
}
inline std::vector<M> induced(const Homology &a, const Homology &b, const Images &f) {
    if (f.size() != a.degrees.size() || f.size() != b.degrees.size())
        throw std::invalid_argument("map dimension mismatch");
    std::vector<M> out(f.size());
    for (std::size_t d = 0; d < f.size(); ++d) {
        if (f[d].size() != static_cast<std::size_t>(a.chain->counts[d]))
            throw std::invalid_argument("map shape mismatch");
        for (auto j : f[d])
            if (j < -1 || j >= b.chain->counts[d])
                throw std::invalid_argument("map image out of bounds");
        for (const auto &z : a.degrees[d].cycles)
            out[d].push_back(b.coordinates(d, linear_apply(f[d], z)));
    }
    return out;
}
inline std::vector<M> compose(const std::vector<M> &f, const std::vector<M> &g, std::size_t lim) {
    if (f.size() != g.size())
        throw std::invalid_argument("composition degree mismatch");
    std::vector<M> out(f.size());
    for (std::size_t d = 0; d < f.size(); ++d)
        for (const auto &c : f[d])
            out[d].push_back(linear_apply(g[d], c, lim));
    return out;
}
inline Images compose_images(const Images &f, const Images &g) {
    if (f.size() != g.size())
        throw std::invalid_argument("composition degree mismatch");
    Images h(f.size());
    for (std::size_t d = 0; d < f.size(); ++d)
        for (i64 j : f[d])
            h[d].push_back(j < 0 ? -1 : g[d].at(j));
    return h;
}
inline bool dd_zero(const qf::ChainComplex &c, std::size_t lim) {
    for (std::size_t d = 2; d < c.boundaries.size(); ++d)
        for (i64 j = 0; j < c.boundaries[d].cols(); ++j)
            if (!linear_apply(c.boundaries[d - 1], col(c.boundaries[d], j), lim).empty())
                return false;
    return true;
}
} // namespace ops

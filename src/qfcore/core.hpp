// Native, immutable flat simplicial complexes and F2 chain complexes.
// Flat arrays of simplices; the QF trie is in qftree.hpp.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qf {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using Simplex = std::vector<u32>;
constexpr std::size_t no_index = std::numeric_limits<std::size_t>::max();
inline std::string key(const u32 *v, std::size_t n) {
    return n ? std::string(reinterpret_cast<const char *>(v), n * sizeof(u32)) : std::string{};
}
inline std::string key(const Simplex &v) {
    return key(v.data(), v.size());
}
inline bool simplex_less(const Simplex &a, const Simplex &b) {
    return a.size() == b.size() ? a < b : a.size() < b.size();
}
inline void canonical(Simplex &s) {
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
}
struct Complex {
    std::vector<u32> verts;
    std::vector<i64> off{0};
    std::vector<i32> dims;
    std::vector<std::size_t> starts;
    i32 dimension = -1;
    std::size_t size() const { return dims.size(); }
    std::pair<std::size_t, std::size_t> range(i32 d) const {
        if (d < 0 || d > dimension)
            return {0, 0};
        return {starts[d], starts[d + 1]};
    }
    std::size_t count(i32 d) const {
        auto r = range(d);
        return r.second - r.first;
    }
    Simplex simplex(std::size_t i) const {
        if (i >= size())
            throw std::out_of_range("simplex index out of range");
        return Simplex(verts.begin() + off[i], verts.begin() + off[i + 1]);
    }
    std::size_t vertex_index(u32 v) const {
        const auto n = count(0);
        auto it = std::lower_bound(verts.begin(), verts.begin() + n, v);
        return it != verts.begin() + n && *it == v ? static_cast<std::size_t>(it - verts.begin()) : no_index;
    }
    std::unordered_map<std::string, std::size_t> index() const {
        std::unordered_map<std::string, std::size_t> out;
        out.reserve(size());
        for (std::size_t s = 0; s < size(); ++s)
            out.emplace(key(verts.data() + off[s], off[s + 1] - off[s]), s);
        return out;
    }
    void finish() {
        dimension = dims.empty() ? -1 : dims.back();
        starts.assign(static_cast<std::size_t>(dimension + 2), size());
        std::size_t s = 0;
        for (i32 d = 0; d <= dimension; ++d) {
            starts[d] = s;
            while (s < size() && dims[s] == d)
                ++s;
        }
    }
    std::size_t payload_bytes() const {
        return verts.size() * sizeof(u32) + off.size() * sizeof(i64) + dims.size() * sizeof(i32) +
               starts.size() * sizeof(std::size_t);
    }
    std::size_t allocated_bytes() const {
        return sizeof(Complex) + verts.capacity() * sizeof(u32) + off.capacity() * sizeof(i64) +
               dims.capacity() * sizeof(i32) + starts.capacity() * sizeof(std::size_t);
    }
};
using KPtr = std::shared_ptr<Complex>;
inline void check_limit(std::size_t n, std::size_t limit) {
    if (n > limit)
        throw std::length_error("max_simplices limit exceeded; raise it explicitly for a larger complex");
    if (n > std::numeric_limits<u32>::max())
        throw std::length_error("32-bit cell index limit exceeded");
}
inline void check_simplex_budget(std::size_t dimension_plus_one, std::size_t limit) {
    // Any simplex forces all its faces; reject impossible expansions before
    // descending deeply through a huge clique / inserted simplex.
    if (dimension_plus_one >= 64 ||
        ((u64(1) << dimension_plus_one) - 1) > std::min<u64>(limit, std::numeric_limits<u32>::max()))
        throw std::length_error("simplex face count exceeds max_simplices / 32-bit index limit");
}
inline KPtr pack(std::vector<Simplex> cells, std::size_t limit, bool validate = true) {
    for (auto &s : cells) {
        if (s.empty())
            throw std::invalid_argument("empty simplices are not stored");
        if (!std::is_sorted(s.begin(), s.end()) || std::adjacent_find(s.begin(), s.end()) != s.end())
            throw std::invalid_argument("simplex vertices must be strictly increasing");
    }
    std::sort(cells.begin(), cells.end(), simplex_less);
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    check_limit(cells.size(), limit);
    auto k = std::make_shared<Complex>();
    std::size_t nv = 0;
    for (const auto &s : cells)
        nv += s.size();
    k->verts.reserve(nv);
    k->off.reserve(cells.size() + 1);
    k->dims.reserve(cells.size());
    for (const auto &s : cells) {
        k->verts.insert(k->verts.end(), s.begin(), s.end());
        k->off.push_back(static_cast<i64>(k->verts.size()));
        k->dims.push_back(static_cast<i32>(s.size() - 1));
    }
    k->finish();
    if (validate) {
        const auto ix = k->index();
        Simplex f;
        for (const auto &s : cells)
            if (s.size() > 1) {
                for (std::size_t i = 0; i < s.size(); ++i) {
                    f = s;
                    f.erase(f.begin() + i);
                    if (!ix.count(key(f)))
                        throw std::invalid_argument("complex is not closed under faces");
                }
            }
    }
    return k;
}
inline KPtr from_arrays(const std::vector<u32> &verts, const std::vector<i64> &off, std::size_t limit) {
    if (off.empty() || off.front() != 0 || off.back() != static_cast<i64>(verts.size()))
        throw std::invalid_argument("offsets must start at zero and end at len(vertices)");
    check_limit(off.size() - 1, limit);
    std::vector<Simplex> cells;
    cells.reserve(off.size() - 1);
    for (std::size_t s = 0; s + 1 < off.size(); ++s) {
        if (off[s] < 0 || off[s + 1] <= off[s] || off[s + 1] > static_cast<i64>(verts.size()))
            throw std::invalid_argument("offsets must delimit nonempty, in-bounds simplices");
        cells.emplace_back(verts.begin() + off[s], verts.begin() + off[s + 1]);
    }
    // Duplicate cells are rejected here, not silently assigned different IDs.
    auto copy = cells;
    std::sort(copy.begin(), copy.end(), simplex_less);
    if (std::adjacent_find(copy.begin(), copy.end()) != copy.end())
        throw std::invalid_argument("duplicate simplex");
    return pack(std::move(cells), limit, true);
}
inline KPtr from_graph(std::size_t n, const std::vector<u32> &edges, i32 cap, std::size_t limit) {
    if (cap < -1)
        throw std::invalid_argument("max_dimension must be nonnegative or None");
    if (edges.size() % 2)
        throw std::invalid_argument("edges must have two endpoints");
    check_limit(n, limit);
    std::vector<std::vector<u32>> adj(n);
    for (std::size_t i = 0; i < edges.size(); i += 2) {
        u32 a = edges[i], b = edges[i + 1];
        if (a >= n || b >= n)
            throw std::invalid_argument("edge endpoint outside 0..n_vertices-1");
        if (a == b)
            throw std::invalid_argument("self-loops are not simplicial edges");
        if (a > b)
            std::swap(a, b);
        adj[a].push_back(b); // Only higher neighbors are needed by the enumeration.
    }
    for (auto &row : adj)
        canonical(row);
    const std::size_t max_size = cap < 0 ? n : static_cast<std::size_t>(cap) + 1;
    std::vector<Simplex> cells;
    Simplex prefix;
    std::function<void(const Simplex &)> visit = [&](const Simplex &cand) {
        for (std::size_t i = 0; i < cand.size(); ++i) {
            const u32 v = cand[i];
            prefix.push_back(v);
            check_simplex_budget(prefix.size(), limit);
            check_limit(cells.size() + 1, limit);
            cells.push_back(prefix);
            if (prefix.size() < max_size) {
                Simplex next;
                std::set_intersection(cand.begin() + i + 1, cand.end(), adj[v].begin(), adj[v].end(),
                                      std::back_inserter(next));
                if (!next.empty())
                    visit(next);
            }
            prefix.pop_back();
        }
    };
    Simplex all(n);
    std::iota(all.begin(), all.end(), u32(0));
    if (max_size)
        visit(all);
    return pack(std::move(cells), limit, false);
}
inline KPtr from_points(const std::vector<double> &pts, std::size_t n, std::size_t p, double radius, i32 cap,
                        bool torus, std::size_t limit) {
    if ((n && p == 0) || (p && n > pts.size() / p) || n * p != pts.size())
        throw std::invalid_argument("points shape does not match buffer");
    if (!std::isfinite(radius) || radius < 0)
        throw std::invalid_argument("radius must be finite and nonnegative");
    for (double x : pts)
        if (!std::isfinite(x))
            throw std::invalid_argument("points must be finite");
    check_limit(n, limit);
    if (cap == 0)
        return from_graph(n, {}, 0, limit);
    std::vector<u32> edges;
    const long double r2 = static_cast<long double>(radius) * radius;
    // The distance predicate is shared by the grid and brute-force paths.
    // In particular, keep the original long-double arithmetic and <= boundary.
    auto consider_pair = [&](std::size_t i, std::size_t j) {
        long double ss = 0;
        for (std::size_t a = 0; a < p; ++a) {
            long double d = static_cast<long double>(pts[i * p + a]) - pts[j * p + a];
            if (torus)
                d -= std::round(d);
            ss += d * d;
        }
        if (ss <= r2) {
            check_limit(n + edges.size() / 2 + 1, limit);
            edges.push_back(static_cast<u32>(i));
            edges.push_back(static_cast<u32>(j));
        }
    };
    // Exact acceleration for points in the periodic unit square.
    // Width >= 2*r leaves a substantial margin at bin boundaries: any edge
    // lies in the same or an adjacent bin in each periodic coordinate.
    // The grid has at most n bins. There is no dense n-by-n distance matrix.
    // Other dimensions/coordinate domains retain the general reference path.
    std::size_t side = 0;
    if (torus && p == 2 && n >= 64 && radius > 0) {
        bool unit_square = true;
        for (double x : pts)
            if (x < 0 || x > 1) {
                unit_square = false;
                break;
            }
        if (unit_square) {
            const long double s = std::min(std::floor(0.5L / static_cast<long double>(radius)),
                                           std::floor(std::sqrt(static_cast<long double>(n))));
            if (s >= 3)
                side = static_cast<std::size_t>(s);
        }
    }
    if (side >= 3) {
        std::vector<Simplex> bins(side * side);
        auto coordinate = [&](double x) {
            auto b = static_cast<std::size_t>(std::floor(static_cast<long double>(x) * side));
            return b == side ? std::size_t(0) : b; // 1 and 0 coincide on the torus.
        };
        std::vector<std::pair<std::size_t, std::size_t>> positions(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto x = coordinate(pts[2 * i]), y = coordinate(pts[2 * i + 1]);
            positions[i] = {x, y};
            bins[y * side + x].push_back(static_cast<u32>(i));
        }
        auto wrap = [&](std::size_t b, int delta) {
            if (delta < 0)
                return b ? b - 1 : side - 1;
            if (delta > 0)
                return b + 1 == side ? std::size_t(0) : b + 1;
            return b;
        };
        for (std::size_t i = 0; i < n; ++i) {
            const auto xy = positions[i];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto &bin = bins[wrap(xy.second, dy) * side + wrap(xy.first, dx)];
                    for (u32 j : bin)
                        if (j > i)
                            consider_pair(i, j);
                }
        }
    } else {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                consider_pair(i, j);
    }
    return from_graph(n, edges, cap, limit);
}
inline std::pair<KPtr, bool> insert(const KPtr &k, Simplex s, std::size_t limit) {
    canonical(s);
    if (s.empty())
        return {k, false};
    check_simplex_budget(s.size(), limit);
    auto ix = k->index();
    if (ix.count(key(s)))
        return {k, false};
    std::vector<Simplex> cells;
    cells.reserve(k->size());
    for (std::size_t i = 0; i < k->size(); ++i)
        cells.push_back(k->simplex(i));
    Simplex f;
    std::function<void(std::size_t)> visit = [&](std::size_t start) {
        for (std::size_t j = start; j < s.size(); ++j) {
            f.push_back(s[j]);
            if (ix.emplace(key(f), 0).second) {
                check_limit(cells.size() + 1, limit);
                cells.push_back(f);
            }
            visit(j + 1);
            f.pop_back();
        }
    };
    visit(0);
    return {pack(std::move(cells), limit, false), true};
}
struct Subcomplex {
    KPtr parent;
    std::vector<std::uint8_t> mask;
    // Component labels aligned with parent's actual 0-cells (not max vertex ID).
    std::vector<i32> vcomp;
    i32 ncomp = 0;
    std::size_t size() const { return std::count(mask.begin(), mask.end(), std::uint8_t(1)); }
    i32 component(u32 label) const {
        auto p = parent->vertex_index(label);
        return p == no_index ? -1 : vcomp[p];
    }
};
using APtr = std::shared_ptr<Subcomplex>;
inline void find_components(Subcomplex &a) {
    const auto &k = *a.parent;
    auto nv = k.count(0);
    a.vcomp.assign(nv, -1);
    std::vector<std::size_t> p(nv);
    std::iota(p.begin(), p.end(), std::size_t(0));
    auto root = [&](std::size_t x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    };
    auto er = k.range(1);
    for (auto s = er.first; s < er.second; ++s)
        if (a.mask[s]) {
            auto i = k.vertex_index(k.verts[k.off[s]]), j = k.vertex_index(k.verts[k.off[s] + 1]);
            if (i == no_index || j == no_index || !a.mask[i] || !a.mask[j])
                throw std::invalid_argument("subcomplex edge has a missing endpoint");
            i = root(i);
            j = root(j);
            if (i > j)
                std::swap(i, j);
            p[j] = i;
        }
    std::unordered_map<std::size_t, i32> labels;
    a.ncomp = 0;
    for (std::size_t i = 0; i < nv; ++i)
        if (a.mask[i]) {
            auto r = root(i);
            auto it = labels.find(r);
            if (it == labels.end())
                it = labels.emplace(r, a.ncomp++).first;
            a.vcomp[i] = it->second;
        }
}
inline APtr from_mask(KPtr k, std::vector<std::uint8_t> mask, bool validate = true) {
    if (mask.size() != k->size())
        throw std::invalid_argument("mask length must equal num_simplices");
    for (auto x : mask)
        if (x > 1)
            throw std::invalid_argument("mask entries must be 0 or 1");
    if (validate) {
        const auto ix = k->index();
        for (std::size_t s = 0; s < k->size(); ++s)
            if (mask[s] && k->dims[s] > 0) {
                const auto v = k->simplex(s);
                for (std::size_t j = 0; j < v.size(); ++j) {
                    auto f = v;
                    f.erase(f.begin() + j);
                    auto it = ix.find(key(f));
                    if (it == ix.end() || !mask[it->second])
                        throw std::invalid_argument("mask is not a subcomplex: a face is missing");
                }
            }
    }
    auto a = std::make_shared<Subcomplex>();
    a->parent = std::move(k);
    a->mask = std::move(mask);
    find_components(*a);
    return a;
}
inline APtr induced(KPtr k, Simplex selected) {
    canonical(selected);
    for (u32 v : selected)
        if (k->vertex_index(v) == no_index)
            throw std::invalid_argument("unknown selected vertex");
    std::vector<std::uint8_t> mask(k->size(), 1);
    for (std::size_t s = 0; s < k->size(); ++s)
        for (i64 p = k->off[s]; p < k->off[s + 1]; ++p)
            if (!std::binary_search(selected.begin(), selected.end(), k->verts[p])) {
                mask[s] = 0;
                break;
            }
    return from_mask(std::move(k), std::move(mask), false);
}
inline u64 edge_key(u32 a, u32 b) {
    if (a > b)
        std::swap(a, b);
    return (u64(a) << 32) | b;
}
inline APtr flag_subcomplex(KPtr k, const std::vector<u32> &edges, Simplex vertices) {
    if (edges.size() % 2)
        throw std::invalid_argument("edge buffer must contain endpoint pairs");
    const auto ix = k->index();
    std::unordered_set<u64> eset;
    for (std::size_t i = 0; i < edges.size(); i += 2) {
        u32 a = edges[i], b = edges[i + 1];
        if (a == b)
            throw std::invalid_argument("self-loop in subcomplex");
        Simplex s{a, b};
        canonical(s);
        if (!ix.count(key(s)))
            throw std::invalid_argument("subcomplex edge is not in K");
        eset.insert(edge_key(a, b));
        vertices.push_back(a);
        vertices.push_back(b);
    }
    canonical(vertices);
    for (u32 v : vertices)
        if (k->vertex_index(v) == no_index)
            throw std::invalid_argument("unknown subcomplex vertex");
    std::vector<std::uint8_t> mask(k->size(), 1);
    for (std::size_t s = 0; s < k->size(); ++s) {
        const auto v = k->simplex(s);
        for (auto x : v)
            if (!std::binary_search(vertices.begin(), vertices.end(), x)) {
                mask[s] = 0;
                break;
            }
        for (std::size_t i = 0; mask[s] && i < v.size(); ++i)
            for (std::size_t j = i + 1; j < v.size(); ++j)
                if (!eset.count(edge_key(v[i], v[j]))) {
                    mask[s] = 0;
                    break;
                }
    }
    return from_mask(std::move(k), std::move(mask), false);
}
inline bool component_fullness(const Subcomplex &a) {
    const auto &k = *a.parent;
    for (std::size_t s = 0; s < k.size(); ++s)
        if (!a.mask[s]) {
            i32 c = a.component(k.verts[k.off[s]]);
            if (c < 0)
                continue;
            bool same = true;
            for (i64 p = k.off[s] + 1; p < k.off[s + 1]; ++p)
                if (a.component(k.verts[p]) != c) {
                    same = false;
                    break;
                }
            if (same)
                return false;
        }
    return true;
}
inline KPtr cone_model(const Subcomplex &a, std::size_t limit) {
    const auto &k = *a.parent;
    check_limit(k.size() + a.size() + a.ncomp, limit);
    u64 apex0 = k.count(0) ? u64(k.verts[k.count(0) - 1]) + 1 : 0;
    if (a.ncomp && apex0 + static_cast<u64>(a.ncomp) - 1 > std::numeric_limits<u32>::max())
        throw std::overflow_error("no unused uint32 vertex IDs for cone apexes");
    std::vector<Simplex> cells;
    cells.reserve(k.size() + a.size() + a.ncomp);
    for (std::size_t s = 0; s < k.size(); ++s) {
        auto v = k.simplex(s);
        cells.push_back(v);
        if (a.mask[s]) {
            i32 c = a.component(v.front());
            if (c < 0)
                throw std::logic_error("invalid component label");
            v.push_back(static_cast<u32>(apex0 + c));
            cells.push_back(std::move(v));
        }
    }
    for (i32 i = 0; i < a.ncomp; ++i)
        cells.push_back(Simplex{static_cast<u32>(apex0 + i)});
    return pack(std::move(cells), limit, false);
}
struct Boundary {
    std::vector<u32> idx;
    std::vector<i64> off{0};
    i64 rows = 0;
    i64 cols() const { return static_cast<i64>(off.size() - 1); }
    std::size_t payload_bytes() const { return idx.size() * sizeof(u32) + off.size() * sizeof(i64); }
};
inline Boundary boundary(const Complex &k, const Subcomplex *a, i32 d, bool quotient) {
    if (d < 0)
        throw std::invalid_argument("boundary dimension must be nonnegative");
    const auto keep = [&](std::size_t s) {
        return !a || !a->mask[s];
    };
    const i32 nc = quotient && a ? a->ncomp : 0;
    Boundary out;
    if (d == 0) {
        i64 c = nc;
        auto r = k.range(0);
        for (auto s = r.first; s < r.second; ++s)
            if (keep(s))
                ++c;
        out.off.assign(static_cast<std::size_t>(c + 1), 0);
        return out;
    }
    const auto lr = k.range(d - 1);
    // Only the lower-dimensional index is built. No unused all-simplex index.
    std::unordered_map<std::string, u32> low;
    low.reserve(lr.second - lr.first);
    u32 rows = d == 1 ? static_cast<u32>(nc) : 0;
    for (auto s = lr.first; s < lr.second; ++s)
        if (keep(s))
            low.emplace(key(k.verts.data() + k.off[s], k.off[s + 1] - k.off[s]), rows++);
    out.rows = rows;
    auto hi = k.range(d);
    Simplex face, col;
    out.off.reserve(hi.second - hi.first + 1);
    for (auto s = hi.first; s < hi.second; ++s)
        if (keep(s)) {
            auto v = k.simplex(s);
            col.clear();
            for (std::size_t j = 0; j < v.size(); ++j) {
                face = v;
                face.erase(face.begin() + j);
                const auto it = low.find(key(face));
                if (it != low.end())
                    col.push_back(it->second);
                else if (quotient && d == 1) {
                    const i32 c = a ? a->component(face[0]) : -1;
                    if (c < 0)
                        throw std::logic_error("missing boundary vertex is not a collapsed component");
                    col.push_back(static_cast<u32>(c));
                }
            }
            std::sort(col.begin(), col.end());
            for (std::size_t i = 0; i < col.size();) {
                auto j = i + 1;
                while (j < col.size() && col[j] == col[i])
                    ++j;
                if ((j - i) % 2)
                    out.idx.push_back(col[i]);
                i = j;
            }
            out.off.push_back(static_cast<i64>(out.idx.size()));
        }
    return out;
}
inline i64 rank_f2(const Boundary &b) {
    // Plain sparse column reduction (no clearing optimization).
    std::unordered_map<u32, Simplex> pivots;
    pivots.reserve(static_cast<std::size_t>(b.cols()));
    Simplex tmp;
    for (i64 j = 0; j < b.cols(); ++j) {
        Simplex c(b.idx.begin() + b.off[j], b.idx.begin() + b.off[j + 1]);
        while (!c.empty()) {
            auto it = pivots.find(c.back());
            if (it == pivots.end()) {
                const auto low = c.back();
                pivots.emplace(low, std::move(c));
                break;
            }
            tmp.clear();
            std::set_symmetric_difference(c.begin(), c.end(), it->second.begin(), it->second.end(),
                                          std::back_inserter(tmp));
            c.swap(tmp);
        }
    }
    return static_cast<i64>(pivots.size());
}
inline std::vector<i64> cell_counts(const Complex &k, const Subcomplex *a, bool quotient) {
    std::vector<i64> counts(static_cast<std::size_t>(k.dimension + 1), 0);
    for (std::size_t s = 0; s < k.size(); ++s)
        if (!a || !a->mask[s])
            ++counts[k.dims[s]];
    if (quotient && a && a->ncomp) {
        if (counts.empty())
            counts.push_back(0);
        counts[0] += a->ncomp;
    }
    return counts;
}
inline std::vector<i64> betti(const Complex &k, const Subcomplex *a, bool quotient) {
    auto b = cell_counts(k, a, quotient);
    for (i32 d = 1; d <= k.dimension; ++d) {
        i64 r = rank_f2(boundary(k, a, d, quotient));
        b[d - 1] -= r;
        b[d] -= r;
    }
    for (auto x : b)
        if (x < 0)
            throw std::logic_error("negative Betti number");
    return b;
}
struct ChainComplex {
    std::vector<i64> counts;
    std::vector<Boundary> boundaries;
    // Chain-level object only: it has no attaching maps and is not a QF-tree.
    bool quotient = false;
    std::vector<i64> betti() const {
        auto b = counts;
        for (std::size_t d = 1; d < boundaries.size(); ++d) {
            const i64 r = rank_f2(boundaries[d]);
            b[d - 1] -= r;
            b[d] -= r;
        }
        return b;
    }
    std::size_t payload_bytes() const {
        std::size_t n = counts.size() * sizeof(i64);
        for (const auto &b : boundaries)
            n += b.payload_bytes();
        return n;
    }
};
using CPtr = std::shared_ptr<ChainComplex>;
inline CPtr make_chain(const Complex &k, const Subcomplex *a, bool quotient) {
    auto c = std::make_shared<ChainComplex>();
    c->quotient = quotient;
    c->counts = cell_counts(k, a, quotient);
    for (i32 d = 0; d < static_cast<i32>(c->counts.size()); ++d)
        c->boundaries.push_back(boundary(k, a, d, quotient));
    return c; // No reference to the parent: the chain object owns all its data.
}
inline std::pair<i64, i64> low_degree_correction(const Subcomplex &a) {
    auto all = from_mask(a.parent, std::vector<std::uint8_t>(a.parent->size(), 1), false);
    std::unordered_set<i32> touched;
    for (std::size_t i = 0; i < a.vcomp.size(); ++i)
        if (a.vcomp[i] >= 0)
            touched.insert(all->vcomp[i]);
    return {static_cast<i64>(touched.size()), a.ncomp};
}
} // namespace qf

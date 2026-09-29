#pragma once
#include "topology.hpp"
namespace nextqf {
using ops::M;
using ops::V;
inline M transpose(const M &cols, std::size_t rows) {
    M t(rows);
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (auto i : cols[j]) {
            if (i >= rows)
                throw std::invalid_argument("matrix row out of bounds");
            t[i].push_back(j);
        }
    return t;
}
inline M columns(const qf::Boundary &b) {
    M m;
    for (qf::i64 j = 0; j < b.cols(); ++j)
        m.push_back(ops::col(b, j));
    return m;
}
struct Cohomology {
    std::shared_ptr<Editable> space;
    ChainView view;
    std::size_t limit;
    std::vector<ops::Degree> h;
    std::vector<M> coboundaries;
    Cohomology(const Editable &e, std::size_t lim, int maxdim = -1)
        : space(std::make_shared<Editable>(e)), view(e, maxdim), limit(lim) {
        if (!lim)
            throw std::invalid_argument("fill limit must be positive");
        if (!ops::dd_zero(*view.chain, limit))
            throw std::invalid_argument("d squared is nonzero");
        const auto &c = *view.chain;
        for (std::size_t d = 0; d < c.counts.size(); ++d) {
            M delta(c.counts[d]);
            if (d + 1 < c.counts.size())
                delta = transpose(columns(c.boundaries[d + 1]), c.counts[d]);
            coboundaries.push_back(delta);
            h.emplace_back(lim);
            auto &x = h.back();
            if (d) {
                M prev = transpose(columns(c.boundaries[d]), c.counts[d - 1]);
                for (auto &v : prev)
                    x.span.insert(v);
            }
            x.boundary_rank = x.span.pivots.size();
            for (auto z : ops::kernel(delta, lim)) {
                auto r = x.span.reduce(std::move(z)).first;
                if (!r.empty()) {
                    qf::u32 n = x.cycles.size();
                    x.cycles.push_back(r);
                    x.span.store(std::move(r), V{n});
                }
            }
        }
    }
    std::vector<qf::i64> betti() const {
        std::vector<qf::i64> b;
        for (auto &d : h)
            b.push_back(d.cycles.size());
        return b;
    }
    V coord(int d, V c) const {
        auto r = h.at(d).span.reduce(std::move(c));
        if (!r.first.empty())
            throw std::invalid_argument("cochain is not a cocycle in represented degree");
        return r.second;
    }
    bool eval(int degree, const V &cochain, Id cell) const {
        if (space->at(cell).dim != degree)
            return false;
        auto j = view.index.at(cell);
        return std::binary_search(cochain.begin(), cochain.end(), j);
    }
    // Normalize H1 cochains on a deterministic maximal forest. This is a
    // coboundary gauge change; needed to use based polygon-word diagonals.
    // Gauge change: add a coboundary so that the 1-cocycle vanishes on a
    // deterministic maximal forest. Needed so that the based polygon-word
    // diagonal below is a valid cup-product formula.
    V forest_normalized(V a) const {
        if (h.size() < 2)
            return a;
        std::unordered_map<Id, bool> value;
        std::unordered_map<Id, std::vector<std::pair<Id, Id>>> adj;
        for (Id v : space->ids(0))
            adj[v];
        for (Id edge : space->ids(1)) {
            auto [x, y] = space->endpoints(edge);
            adj[x].push_back({y, edge});
            adj[y].push_back({x, edge});
        }
        for (auto &[v, list] : adj)
            std::sort(list.begin(), list.end());
        for (Id root : space->ids(0))
            if (!value.count(root)) {
                value[root] = false;
                Ids todo{root};
                for (std::size_t k = 0; k < todo.size(); ++k)
                    for (auto [v, e] : adj.at(todo[k]))
                        if (!value.count(v)) {
                            value[v] = value.at(todo[k]) ^ eval(1, a, e);
                            todo.push_back(v);
                        }
            }
        V b;
        for (Id e : space->ids(1)) {
            auto [x, y] = space->endpoints(e);
            if (eval(1, a, e) ^ value.at(x) ^ value.at(y))
                b.push_back(view.index.at(e));
        }
        return b;
    }
    // F2 cup product of cocycles. Simplex cells: Alexander-Whitney through the
    // stored iterated face maps, (a u b)(s) = a(s[0..p]) b(s[p..p+q]).
    // Polygon cells with p=q=1 and forest-normalized cocycles:
    //   <a u b, P> = sum_{i<j} A_i B_j + sum_{eps_i=-1} A_i B_i  (mod 2).
    V cochain_product(int p, V a, int q, V b) const {
        if (p < 0 || q < 0 || p >= static_cast<int>(h.size()) || q >= static_cast<int>(h.size()))
            throw std::out_of_range("cochain degree");
        a = ops::parity(std::move(a));
        b = ops::parity(std::move(b));
        for (auto i : a)
            if (i >= view.ids[p].size())
                throw std::out_of_range("cochain coordinate");
        for (auto i : b)
            if (i >= view.ids[q].size())
                throw std::out_of_range("cochain coordinate");
        if (!ops::linear_apply(coboundaries[p], a, limit).empty() ||
            !ops::linear_apply(coboundaries[q], b, limit).empty())
            throw std::invalid_argument("cup_product accepts cocycles, not arbitrary cochains");
        if (p + q >= static_cast<int>(h.size()))
            return {};
        bool polygons = false;
        for (Id id : space->ids(2))
            if (space->at(id).kind == 2) {
                polygons = true;
                break;
            }
        if (polygons && p == 1)
            a = forest_normalized(std::move(a));
        if (polygons && q == 1)
            b = forest_normalized(std::move(b));
        V out;
        int n = p + q;
        for (Id id : view.ids[n]) {
            auto &c = space->at(id);
            bool term = false;
            if (c.kind == 2) {
                if (p == 0)
                    term = eval(0, a, c.base) && eval(2, b, id);
                else if (q == 0)
                    term = eval(2, a, id) && eval(0, b, c.base);
                else if (p == 1 && q == 1) {
                    bool prefix = false;
                    for (Id w : c.word) {
                        Id e = Editable::edge_id(w);
                        bool av = eval(1, a, e), bv = eval(1, b, e);
                        term ^= prefix && bv;
                        if (w < 0)
                            term ^= av && bv;
                        prefix ^= av;
                    }
                } else
                    throw std::logic_error("unsupported polygon diagonal degree");
            } else {
                Id front = id, back = id;
                for (int j = n; j > p; --j)
                    front = space->face(front, j);
                for (int j = 0; j < p; ++j)
                    back = space->face(back, 0);
                term = eval(p, a, front) && eval(q, b, back);
            }
            if (term)
                out.push_back(view.index.at(id));
        }
        if (!ops::linear_apply(coboundaries[n], out, limit).empty())
            throw std::logic_error("cup of cocycles failed cocycle check");
        return out;
    }
    V product(int p, std::size_t i, int q, std::size_t j) const {
        auto c = cochain_product(p, h.at(p).cycles.at(i), q, h.at(q).cycles.at(j));
        return p + q >= static_cast<int>(h.size()) ? V{} : coord(p + q, std::move(c));
    }
    std::vector<M> cocycles() const {
        std::vector<M> out;
        for (auto &d : h)
            out.push_back(d.cycles);
        return out;
    }
};
inline std::vector<M> pullback(const Cohomology &src, const Cohomology &dst, const ops::Images &f) {
    if (f.size() != src.h.size())
        throw std::invalid_argument("cohomology map degree mismatch");
    std::vector<M> out(src.h.size());
    for (std::size_t d = 0; d < f.size(); ++d) {
        if (d >= dst.h.size() || f[d].size() != src.view.ids[d].size())
            throw std::invalid_argument("cohomology map shape mismatch");
        for (auto &c : dst.h[d].cycles) {
            V a;
            for (std::size_t j = 0; j < f[d].size(); ++j) {
                auto k = f[d][j];
                if (k < -1 || k >= static_cast<Id>(dst.view.ids[d].size()))
                    throw std::out_of_range("cohomology map target");
                if (k >= 0 && std::binary_search(c.begin(), c.end(), static_cast<qf::u32>(k)))
                    a.push_back(j);
            }
            out[d].push_back(src.coord(d, std::move(a)));
        }
    }
    return out;
}
// Reference zigzag interval decomposition. Generalized rank = rank(lim->colim).
// Intended as a correctness oracle for SMALL diagrams, never a fast streaming solver.
// Reference zigzag decomposition through generalized ranks
// r(I) = rank(lim V|_I -> colim V|_I) and interval multiplicities by
// two-sided finite differences. Exact but global; small diagrams only.
inline std::vector<std::array<int, 3>> map_barcode(const std::vector<int> &dims, const std::vector<M> &maps,
                                                   const std::vector<int> &dir, std::size_t lim,
                                                   int max_total) {
    int n = dims.size();
    if (n < 1 || static_cast<int>(maps.size()) != n - 1 || dir.size() != maps.size())
        throw std::invalid_argument("zigzag diagram shape");
    int total = 0;
    for (int d : dims) {
        if (d < 0)
            throw std::invalid_argument("negative dimension");
        if (d > max_total - total)
            throw std::length_error("reference zigzag dimension limit");
        total += d;
    }
    for (int k = 0; k < n - 1; ++k) {
        if (dir[k] != 1 && dir[k] != -1)
            throw std::invalid_argument("direction must be +1 or -1");
        int s = dir[k] > 0 ? k : k + 1, t = dir[k] > 0 ? k + 1 : k;
        if (static_cast<int>(maps[k].size()) != dims[s])
            throw std::invalid_argument("zigzag arrow columns");
        for (auto &v : maps[k]) {
            if (ops::parity(v) != v)
                throw std::invalid_argument("zigzag sparse columns must be sorted unique");
            for (auto i : v)
                if (i >= static_cast<unsigned>(dims[t]))
                    throw std::invalid_argument("zigzag arrow rows");
        }
    }
    std::vector<std::vector<int>> r(n, std::vector<int>(n));
    for (int a = 0; a < n; ++a)
        for (int b = a; b < n; ++b) {
            std::vector<int> off(n + 1, 0);
            for (int i = a; i <= b; ++i)
                off[i + 1] = off[i] + dims[i];
            int size = off[b + 1];
            M constraints(size), relations;
            int row = 0;
            for (int k = a; k < b; ++k) {
                int s = dir[k] > 0 ? k : k + 1, t = dir[k] > 0 ? k + 1 : k;
                for (int j = 0; j < dims[s]; ++j) {
                    V rel{static_cast<qf::u32>(off[s] + j)};
                    for (auto i : maps[k][j]) {
                        constraints[off[s] + j].push_back(row + i);
                        rel.push_back(off[t] + i);
                    }
                    relations.push_back(ops::parity(std::move(rel)));
                }
                for (int i = 0; i < dims[t]; ++i)
                    constraints[off[t] + i].push_back(row + i);
                row += dims[t];
            }
            for (auto &v : constraints)
                v = ops::parity(std::move(v));
            M ker = ops::kernel(constraints, lim);
            auto rankrel = ops::rank(relations, lim);
            for (auto &v : ker) {
                V p;
                for (auto j : v)
                    if (j < static_cast<unsigned>(dims[a]))
                        p.push_back(j);
                relations.push_back(std::move(p));
            }
            r[a][b] = ops::rank(relations, lim) - rankrel;
        }
    std::vector<std::array<int, 3>> bars;
    for (int a = 0; a < n; ++a)
        for (int b = a; b < n; ++b) {
            int m = r[a][b] - (a ? r[a - 1][b] : 0) - (b + 1 < n ? r[a][b + 1] : 0) +
                    (a && b + 1 < n ? r[a - 1][b + 1] : 0);
            if (m < 0)
                throw std::logic_error("negative zigzag interval multiplicity");
            if (m)
                bars.push_back({a, b + 1, m});
        }
    return bars;
}
} // namespace nextqf

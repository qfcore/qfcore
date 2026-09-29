#pragma once
#include <gudhi/Simplex_tree.h>
#include "linear.hpp"
namespace ops {
using ST = Gudhi::Simplex_tree<>;
struct TreeData {
    ST st;
    std::vector<ST::Simplex_handle> order;
    std::size_t nv = 0;
    int dim = -1;
    TreeData(const std::vector<u32> &v, const std::vector<i64> &off) {
        if (off.empty() || off.front() != 0 || off.back() != static_cast<i64>(v.size()))
            throw std::invalid_argument("bad offsets");
        for (std::size_t j = 0; j + 1 < off.size(); ++j) {
            if (off[j] < 0 || off[j + 1] <= off[j] || off[j + 1] > static_cast<i64>(v.size()))
                throw std::invalid_argument("bad simplex");
            std::vector<int> s;
            for (i64 p = off[j]; p < off[j + 1]; ++p) {
                if (v[p] >= static_cast<u32>(std::numeric_limits<int>::max()))
                    throw std::overflow_error("vertex label exceeds GUDHI bound");
                s.push_back(v[p]);
            }
            if (!std::is_sorted(s.begin(), s.end()) || std::adjacent_find(s.begin(), s.end()) != s.end() ||
                static_cast<int>(s.size() - 1) < dim)
                throw std::invalid_argument("source must be canonical");
            dim = s.size() - 1;
            if (dim == 0)
                ++nv;
            auto r = st.insert_simplex(s, 0.0);
            if (!r.second)
                throw std::invalid_argument("duplicate simplex");
            st.assign_key(r.first, j);
        }
        order.resize(off.size() - 1);
        for (auto h : st.complex_simplex_range())
            order.at(st.key(h)) = h;
        for (auto h : order)
            if (st.dimension(h) > 0) {
                std::vector<int> s(st.simplex_vertex_range(h).begin(), st.simplex_vertex_range(h).end());
                for (std::size_t j = 0; j < s.size(); ++j) {
                    auto f = s;
                    f.erase(f.begin() + j);
                    if (st.find(f) == st.null_simplex())
                        throw std::invalid_argument("source not closed");
                }
            }
    }
    std::size_t vertex_id(int label) const {
        auto h = st.find(std::vector<int>{label});
        if (h == st.null_simplex())
            throw std::invalid_argument("unknown vertex");
        return st.key(h);
    }
};
struct Snapshot {
    qf::CPtr chain;
    std::vector<i64> remap;
    std::vector<u32> zero_reps, vertex_image;
};
struct TreeState {
    std::shared_ptr<TreeData> data;
    std::vector<std::uint8_t> mask;
    TreeState(std::shared_ptr<TreeData> d, std::vector<std::uint8_t> m)
        : data(std::move(d)), mask(std::move(m)) {
        if (mask.size() != data->order.size())
            throw std::invalid_argument("mask length mismatch");
        for (std::size_t j = 0; j < mask.size(); ++j) {
            if (mask[j] > 1)
                throw std::invalid_argument("nonbinary mask");
            if (mask[j] && data->st.dimension(data->order[j]) > 0)
                for (auto f : data->st.boundary_simplex_range(data->order[j]))
                    if (!mask[data->st.key(f)])
                        throw std::invalid_argument("mask not closed");
        }
    }
    TreeState(const TreeState &) = default;
    std::shared_ptr<TreeState> absorb(const std::vector<u32> &ids) const {
        auto next = std::make_shared<TreeState>(*this);
        std::vector<u32> todo = ids;
        while (!todo.empty()) {
            u32 j = todo.back();
            todo.pop_back();
            if (j >= mask.size())
                throw std::out_of_range("source ID");
            if (next->mask[j])
                continue;
            next->mask[j] = 1;
            auto h = data->order[j];
            if (data->st.dimension(h) > 0)
                for (auto f : data->st.boundary_simplex_range(h))
                    todo.push_back(data->st.key(f));
        }
        return next;
    }
    Snapshot snapshot() const {
        const auto &st = data->st;
        auto &order = data->order;
        std::vector<u32> p(data->nv);
        std::iota(p.begin(), p.end(), 0);
        auto root = [&](u32 x) {
            while (p[x] != x) {
                p[x] = p[p[x]];
                x = p[x];
            }
            return x;
        };
        for (std::size_t j = data->nv; j < order.size() && st.dimension(order[j]) == 1; ++j)
            if (mask[j]) {
                auto vv = st.simplex_vertex_range(order[j]);
                auto it = vv.begin();
                auto a = root(data->vertex_id(*it++)), b = root(data->vertex_id(*it));
                if (a > b)
                    std::swap(a, b);
                p[b] = a;
            }
        std::map<u32, u32> comp;
        for (u32 j = 0; j < data->nv; ++j)
            if (mask[j]) {
                u32 r = root(j);
                if (!comp.count(r))
                    comp[r] = comp.size();
            }
        Snapshot s;
        s.chain = std::make_shared<qf::ChainComplex>();
        auto &c = *s.chain;
        c.quotient = true;
        c.counts.assign(data->dim + 1, 0);
        c.boundaries.resize(data->dim + 1);
        s.remap.assign(mask.size(), -1);
        s.vertex_image.resize(data->nv);
        if (!c.counts.empty())
            c.counts[0] = comp.size();
        for (auto kv : comp)
            s.zero_reps.push_back(kv.first);
        for (std::size_t j = 0; j < order.size(); ++j)
            if (!mask[j]) {
                int d = st.dimension(order[j]);
                s.remap[j] = c.counts[d]++;
                if (d == 0)
                    s.zero_reps.push_back(j);
            }
        for (u32 j = 0; j < data->nv; ++j)
            s.vertex_image[j] = mask[j] ? comp.at(root(j)) : static_cast<u32>(s.remap[j]);
        if (!c.boundaries.empty())
            c.boundaries[0].off.assign(c.counts[0] + 1, 0);
        for (std::size_t d = 1; d < c.counts.size(); ++d)
            c.boundaries[d].rows = c.counts[d - 1];
        for (std::size_t j = data->nv; j < order.size(); ++j)
            if (!mask[j]) {
                auto h = order[j];
                int d = st.dimension(h);
                V col;
                for (auto f : st.boundary_simplex_range(h)) {
                    u32 fid = st.key(f);
                    if (d == 1)
                        col.push_back(s.vertex_image[fid]);
                    else if (!mask[fid])
                        col.push_back(s.remap[fid]);
                }
                col = parity(std::move(col));
                auto &b = c.boundaries[d];
                b.idx.insert(b.idx.end(), col.begin(), col.end());
                b.off.push_back(b.idx.size());
            }
        return s;
    }
};
inline Images snapshot_images(const TreeData &data, const Snapshot &a, const Snapshot &b) {
    Images f(a.chain->counts.size());
    if (!f.empty())
        for (auto v : a.zero_reps)
            f[0].push_back(b.vertex_image.at(v));
    for (std::size_t j = data.nv; j < data.order.size(); ++j)
        if (a.remap[j] >= 0)
            f[data.st.dimension(data.order[j])].push_back(b.remap[j]);
    return f;
}
} // namespace ops

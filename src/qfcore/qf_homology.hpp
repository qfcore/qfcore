// Optional linearization consumer. QFTree construction and I/O never call this.
#pragma once
#include "qftree.hpp"
namespace qf {
inline std::vector<std::pair<u32, i64>> qf_signed_boundary(const QFTree &q, u32 id, bool relative = false) {
    q.require_cell(id);
    std::map<u32, i64> terms;
    const auto d = q.dims[id];
    if (d > 0)
        for (i32 i = 0; i <= d; ++i) {
            auto t = q.facets[q.facet_off[id] + i];
            if (q.dims[t] == d - 1 && !(relative && t < q.ncomp))
                terms[t] += (i % 2 ? -1 : 1);
        }
    std::vector<std::pair<u32, i64>> out;
    for (auto p : terms)
        if (p.second)
            out.push_back(p);
    return out;
}
inline Boundary qf_boundary(const QFTree &q, i32 d, bool relative) {
    if (d < 0)
        throw std::invalid_argument("negative boundary dimension");
    Boundary b;
    auto hi = q.range(d);
    if (d == 0) {
        b.off.assign(q.count(0) - (relative ? q.ncomp : 0) + 1, 0);
        return b;
    }
    auto lo = q.range(d - 1);
    if (relative && d == 1)
        lo.first = std::max<std::size_t>(lo.first, q.ncomp);
    b.rows = static_cast<i64>(lo.second - lo.first);
    for (auto id = hi.first; id < hi.second; ++id) {
        // Preserve local facet occurrences in QF; cancel only in this consumer.
        auto terms = qf_signed_boundary(q, static_cast<u32>(id), relative);
        for (const auto &term : terms)
            if (term.second % 2)
                b.idx.push_back(static_cast<u32>(term.first - lo.first));
        b.off.push_back(static_cast<i64>(b.idx.size()));
    }
    return b;
}
inline CPtr qf_chain(const QFTree &q, bool relative) {
    auto c = std::make_shared<ChainComplex>();
    c->quotient = !relative;
    c->counts.assign(static_cast<std::size_t>(q.source_dimension + 1), 0);
    for (auto d : q.dims)
        ++c->counts[d];
    if (relative && !c->counts.empty())
        c->counts[0] -= q.ncomp;
    for (i32 d = 0; d <= q.source_dimension; ++d)
        c->boundaries.push_back(qf_boundary(q, d, relative));
    return c;
}
inline std::vector<i64> qf_chain_image(const QFTree &q, const QFTree &r, const std::vector<u32> &image, i32 d,
                                       bool relative) {
    if (d < 0)
        throw std::invalid_argument("negative dimension");
    if (!qf_map_valid(q, r, image))
        throw std::invalid_argument("not a compatible pointed face morphism");
    auto src = q.range(d), dst = r.range(d);
    if (d == 0 && relative) {
        src.first = std::max<std::size_t>(src.first, q.ncomp);
        dst.first = std::max<std::size_t>(dst.first, r.ncomp);
    }
    std::vector<i64> out;
    for (auto s = src.first; s < src.second; ++s) {
        auto t = image[s];
        out.push_back(r.dims[t] == d && !(relative && t < r.ncomp)
                          ? static_cast<i64>(t) - static_cast<i64>(dst.first)
                          : -1);
    }
    return out;
}
} // namespace qf

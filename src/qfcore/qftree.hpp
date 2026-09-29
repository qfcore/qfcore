// Source-labelled componentwise-collapse presentations. No homology computation.
// Local facet index i means deletion of source vertex i; it is not an index into the quotient word.
#pragma once
#include "core.hpp"
#include <map>
#include <fstream>
#include <filesystem>
#include <array>
#include <type_traits>

namespace qf {
struct WordNode {
    std::map<u32, u32> children;
    std::vector<u32> cells; // terminal bucket: never merge cells with equal words
};
struct QFTree {
    u64 source_size = 0;
    i32 source_dimension = -1;
    u32 ncomp = 0; // distinguished points occupy IDs [0,ncomp)
    std::vector<i32> dims;
    std::vector<i64> source_ids; // -1 for a component point
    std::vector<u32> supports;
    std::vector<i64> support_off{0};
    std::vector<u32> facets; // one entry per LOCAL occurrence, including constants
    std::vector<i64> facet_off{0};
    std::vector<u32> component_vertices;
    std::vector<i64> component_off{0};
    // Derived indices. Not the topological source of truth.
    std::vector<std::size_t> starts;
    std::vector<u32> vertex_labels, vertex_images;
    std::vector<WordNode> trie;
    bool word_index = true;
    // Optional source archive; its memory is explicitly additional.
    KPtr archive;
    std::vector<std::uint8_t> archive_mask;
    std::size_t size() const { return dims.size(); }
    i32 dimension() const { return dims.empty() ? -1 : dims.back(); }
    void require_cell(std::size_t id) const {
        if (id >= size())
            throw std::out_of_range("QF cell ID out of range");
    }
    bool is_component(std::size_t id) const {
        require_cell(id);
        return id < ncomp;
    }
    std::pair<std::size_t, std::size_t> range(i32 d) const {
        if (d < 0 || d > dimension())
            return {0, 0};
        return {starts[d], starts[d + 1]};
    }
    std::size_t count(i32 d) const {
        auto r = range(d);
        return r.second - r.first;
    }
    Simplex support(std::size_t id) const {
        require_cell(id);
        return Simplex(supports.begin() + support_off[id], supports.begin() + support_off[id + 1]);
    }
    u32 vertex_image(u32 v) const {
        auto it = std::lower_bound(vertex_labels.begin(), vertex_labels.end(), v);
        if (it == vertex_labels.end() || *it != v)
            throw std::invalid_argument("unknown source vertex");
        return vertex_images[it - vertex_labels.begin()];
    }
    Simplex word(std::size_t id) const {
        require_cell(id);
        if (id < ncomp)
            return {static_cast<u32>(id)};
        Simplex w;
        for (i64 p = support_off[id]; p < support_off[id + 1]; ++p)
            w.push_back(vertex_image(supports[p]));
        canonical(w);
        if (w.empty())
            throw std::logic_error("empty genuine support");
        w.resize(static_cast<std::size_t>(dims[id]) + 1, w.back());
        return w;
    }
    std::vector<u32> lookup_word(const Simplex &w) const {
        if (!word_index)
            throw std::logic_error("word index disabled; use with_word_index(True)");
        u32 node = 0;
        for (auto token : w) {
            auto it = trie[node].children.find(token);
            if (it == trie[node].children.end())
                return {};
            node = it->second;
        }
        return trie[node].cells;
    }
    i64 find_source(Simplex s) const {
        canonical(s);
        if (s.empty())
            return -1;
        auto r = range(static_cast<i32>(s.size() - 1));
        // Genuine cells retain canonical source order; skip component points.
        r.first = std::max<std::size_t>(r.first, ncomp);
        auto lo = r.first, hi = r.second;
        while (lo < hi) {
            auto mid = lo + (hi - lo) / 2;
            auto b = supports.begin() + support_off[mid], e = supports.begin() + support_off[mid + 1];
            if (std::lexicographical_compare(b, e, s.begin(), s.end()))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < r.second && support(lo) == s ? static_cast<i64>(lo) : -1;
    }
    void build_indices() {
        const auto d = dimension();
        starts.assign(static_cast<std::size_t>(d + 2), size());
        std::size_t s = 0;
        for (i32 k = 0; k <= d; ++k) {
            starts[k] = s;
            while (s < size() && dims[s] == k)
                ++s;
        }
        std::vector<std::pair<u32, u32>> vm;
        for (u32 c = 0; c < ncomp; ++c)
            for (i64 p = component_off[c]; p < component_off[c + 1]; ++p)
                vm.emplace_back(component_vertices[p], c);
        auto zr = range(0);
        for (auto id = std::max<std::size_t>(ncomp, zr.first); id < zr.second; ++id)
            vm.emplace_back(supports[support_off[id]], static_cast<u32>(id));
        std::sort(vm.begin(), vm.end());
        vertex_labels.clear();
        vertex_images.clear();
        for (const auto &p : vm) {
            if (!vertex_labels.empty() && vertex_labels.back() == p.first)
                throw std::invalid_argument("source vertex belongs to more than one zero-cell");
            vertex_labels.push_back(p.first);
            vertex_images.push_back(p.second);
        }
        trie.clear();
        if (word_index) {
            trie.emplace_back();
            for (std::size_t id = 0; id < size(); ++id) {
                u32 node = 0;
                for (u32 token : word(id)) {
                    auto it = trie[node].children.find(token);
                    if (it == trie[node].children.end()) {
                        if (trie.size() >= std::numeric_limits<u32>::max())
                            throw std::length_error("word trie exceeds uint32 node limit");
                        auto next = static_cast<u32>(trie.size());
                        trie[node].children.emplace(token, next);
                        trie.emplace_back();
                        node = next;
                    } else
                        node = it->second;
                }
                trie[node].cells.push_back(static_cast<u32>(id));
            }
        }
    }
    // Abstract face with component points extended absorbingly in formal degrees.
    u32 formal_face(u32 id, u32 local) const {
        if (id < ncomp)
            return id;
        if (dims[id] <= 0 || local > static_cast<u32>(dims[id]))
            throw std::invalid_argument("invalid local facet index");
        return facets[facet_off[id] + local];
    }
    u32 iterated_face(u32 id, const Simplex &deletions) const {
        require_cell(id);
        i32 formal_dimension = dims[id];
        for (u32 i : deletions) {
            if (formal_dimension <= 0 || i > static_cast<u32>(formal_dimension))
                throw std::invalid_argument("face deletion index exceeds current formal dimension");
            id = formal_face(id, i);
            --formal_dimension;
        }
        return id;
    }
    std::vector<u32> closure(const std::vector<u32> &selected) const {
        std::vector<std::uint8_t> m(size(), 0);
        for (auto id : selected) {
            require_cell(id);
            m[id] = 1;
        }
        for (auto id = size(); id-- > 0;)
            if (m[id])
                for (i64 p = facet_off[id]; p < facet_off[id + 1]; ++p)
                    m[facets[p]] = 1;
        std::vector<u32> result;
        for (std::size_t id = 0; id < size(); ++id)
            if (m[id])
                result.push_back(static_cast<u32>(id));
        return result;
    }
    std::vector<u32> cofaces(u32 target, i32 codimension = -1) const {
        require_cell(target);
        if (codimension < -1)
            throw std::invalid_argument("invalid coface codimension");
        std::vector<std::uint8_t> hit(size(), 0);
        hit[target] = 1;
        std::vector<u32> out;
        for (std::size_t id = target; id < size(); ++id) {
            for (i64 p = facet_off[id]; !hit[id] && p < facet_off[id + 1]; ++p)
                hit[id] = hit[facets[p]];
            if (hit[id] && (codimension < 0 || dims[id] - dims[target] == codimension))
                out.push_back(static_cast<u32>(id));
        }
        return out;
    }
    bool strictly_graded() const {
        std::vector<Simplex> reach(size());
        for (u32 i = 0; i < ncomp; ++i)
            reach[i] = {i};
        for (std::size_t id = ncomp; id < size(); ++id) {
            Simplex inherited, direct;
            for (i64 p = facet_off[id]; p < facet_off[id + 1]; ++p) {
                auto t = facets[p];
                if (t < ncomp)
                    direct.push_back(t);
                else
                    inherited.insert(inherited.end(), reach[t].begin(), reach[t].end());
            }
            canonical(inherited);
            canonical(direct);
            if (dims[id] >= 2)
                for (auto c : direct)
                    if (!std::binary_search(inherited.begin(), inherited.end(), c))
                        return false;
            inherited.insert(inherited.end(), direct.begin(), direct.end());
            canonical(inherited);
            reach[id] = std::move(inherited);
        }
        return true;
    }
    bool regular() const {
        // Checks component fullness for source collapses; this does not recognize balls.
        for (std::size_t id = ncomp; id < size(); ++id) {
            auto v = support(id);
            u32 c = vertex_image(v.front());
            if (c >= ncomp)
                continue;
            bool all = true;
            for (auto x : v)
                if (vertex_image(x) != c) {
                    all = false;
                    break;
                }
            if (all)
                return false;
        }
        return true;
    }
    void validate_records(bool overlaps = true) const {
        if (source_size > std::numeric_limits<u32>::max() || source_dimension < -1 || source_dimension > 62)
            throw std::invalid_argument("invalid source metadata");
        if (size() > source_size || ncomp > size() || source_ids.size() != size())
            throw std::invalid_argument("invalid cell inventory");
        auto offsets = [](const std::vector<i64> &o, std::size_t rows, std::size_t len) {
            if (o.size() != rows + 1 || o.empty() || o.front() != 0 || o.back() != static_cast<i64>(len) ||
                !std::is_sorted(o.begin(), o.end()))
                throw std::invalid_argument("invalid QF offsets");
        };
        offsets(support_off, size(), supports.size());
        offsets(facet_off, size(), facets.size());
        offsets(component_off, ncomp, component_vertices.size());
        std::vector<u32> used_vertices;
        for (u32 c = 0; c < ncomp; ++c) {
            auto b = component_vertices.begin() + component_off[c],
                 e = component_vertices.begin() + component_off[c + 1];
            if (b == e || !std::is_sorted(b, e) || std::adjacent_find(b, e) != e)
                throw std::invalid_argument("invalid component support");
            if (c && component_vertices[component_off[c - 1]] >= *b)
                throw std::invalid_argument("noncanonical component order");
            used_vertices.insert(used_vertices.end(), b, e);
        }
        std::unordered_set<i64> ids;
        for (std::size_t id = 0; id < size(); ++id) {
            i32 d = dims[id];
            if (d < 0 || d > source_dimension || (id && dims[id - 1] > d))
                throw std::invalid_argument("invalid cell dimension order");
            if (id < ncomp) {
                if (d != 0 || source_ids[id] != -1 || support_off[id] != support_off[id + 1] ||
                    facet_off[id] != facet_off[id + 1])
                    throw std::invalid_argument("invalid component-point cell");
                continue;
            }
            auto v = support(id);
            if (v.size() != static_cast<std::size_t>(d) + 1 || !std::is_sorted(v.begin(), v.end()) ||
                std::adjacent_find(v.begin(), v.end()) != v.end())
                throw std::invalid_argument("invalid source support");
            if (id > ncomp && dims[id - 1] == d && !(support(id - 1) < v))
                throw std::invalid_argument("noncanonical or duplicate source support");
            if (source_ids[id] < 0 || static_cast<u64>(source_ids[id]) >= source_size ||
                !ids.insert(source_ids[id]).second)
                throw std::invalid_argument("invalid source simplex ID");
            if (id > ncomp && source_ids[id] <= source_ids[id - 1])
                throw std::invalid_argument("source simplex IDs not increasing");
            if (facet_off[id + 1] - facet_off[id] != (d > 0 ? d + 1 : 0))
                throw std::invalid_argument("wrong number of local facet occurrences");
            if (d == 0)
                used_vertices.push_back(v.front());
            for (i32 i = 0; i <= d && d > 0; ++i) {
                auto t = facets[facet_off[id] + i];
                if (t >= id)
                    throw std::invalid_argument("facet target must precede its source");
                if (t >= ncomp) {
                    auto f = v;
                    f.erase(f.begin() + i);
                    if (dims[t] != d - 1 || support(t) != f)
                        throw std::invalid_argument("facet coordinate/support mismatch");
                } else {
                    for (i32 j = 0; j <= d; ++j)
                        if (j != i) {
                            auto b = component_vertices.begin() + component_off[t],
                                 e = component_vertices.begin() + component_off[t + 1];
                            if (!std::binary_search(b, e, v[j]))
                                throw std::invalid_argument(
                                    "constant facet has vertices outside its component");
                        }
                }
            }
            if (overlaps && d >= 2)
                for (u32 i = 0; i < static_cast<u32>(d); ++i)
                    for (u32 j = i + 1; j <= static_cast<u32>(d); ++j)
                        if (formal_face(formal_face(static_cast<u32>(id), j), i) !=
                            formal_face(formal_face(static_cast<u32>(id), i), j - 1))
                            throw std::invalid_argument("semisimplicial face identity failed");
        }
        std::sort(used_vertices.begin(), used_vertices.end());
        if (std::adjacent_find(used_vertices.begin(), used_vertices.end()) != used_vertices.end())
            throw std::invalid_argument("overlapping zero-cell source supports");
        for (auto v : supports)
            if (!std::binary_search(used_vertices.begin(), used_vertices.end(), v))
                throw std::invalid_argument("missing source vertex image");
        if (archive) {
            if (archive->size() != source_size || archive->dimension != source_dimension)
                throw std::invalid_argument("source archive metadata mismatch");
            auto a = from_mask(archive, archive_mask, true);
            if (used_vertices.size() != archive->count(0) ||
                !std::equal(used_vertices.begin(), used_vertices.end(), archive->verts.begin()))
                throw std::invalid_argument("source archive vertex set mismatch");
            if (static_cast<u32>(a->ncomp) != ncomp || archive->size() - a->size() + ncomp != size())
                throw std::invalid_argument("source archive quotient inventory mismatch");
            for (u32 c = 0; c < ncomp; ++c)
                for (i64 p = component_off[c]; p < component_off[c + 1]; ++p)
                    if (a->component(component_vertices[p]) != static_cast<i32>(c))
                        throw std::invalid_argument("source archive component mismatch");
            for (std::size_t id = ncomp; id < size(); ++id)
                if (a->mask[source_ids[id]] || archive->simplex(source_ids[id]) != support(id))
                    throw std::invalid_argument("source archive survivor mismatch");
        } else if (!archive_mask.empty())
            throw std::invalid_argument("archive mask without source archive");
    }
    std::size_t table_bytes() const {
        return dims.size() * sizeof(i32) + source_ids.size() * sizeof(i64) + supports.size() * sizeof(u32) +
               support_off.size() * sizeof(i64) + facets.size() * sizeof(u32) +
               facet_off.size() * sizeof(i64) + component_vertices.size() * sizeof(u32) +
               component_off.size() * sizeof(i64);
    }
    std::size_t index_payload_bytes() const {
        std::size_t n =
            starts.size() * sizeof(std::size_t) + (vertex_labels.size() + vertex_images.size()) * sizeof(u32);
        for (const auto &node : trie)
            n += node.children.size() * 2 * sizeof(u32) + node.cells.size() * sizeof(u32);
        return n; // Not map/vector object overhead, capacities, or RSS.
    }
    std::size_t archive_bytes() const { return archive ? archive->payload_bytes() + archive_mask.size() : 0; }
};
using TPtr = std::shared_ptr<QFTree>;
inline void append_component(QFTree &q, const Simplex &vertices) {
    q.dims.push_back(0);
    q.source_ids.push_back(-1);
    q.support_off.push_back(static_cast<i64>(q.supports.size()));
    q.facet_off.push_back(static_cast<i64>(q.facets.size()));
    q.component_vertices.insert(q.component_vertices.end(), vertices.begin(), vertices.end());
    q.component_off.push_back(static_cast<i64>(q.component_vertices.size()));
    ++q.ncomp;
}
inline void append_genuine(QFTree &q, i32 d, i64 source_id, const Simplex &support, const Simplex &facets) {
    q.dims.push_back(d);
    q.source_ids.push_back(source_id);
    q.supports.insert(q.supports.end(), support.begin(), support.end());
    q.support_off.push_back(static_cast<i64>(q.supports.size()));
    q.facets.insert(q.facets.end(), facets.begin(), facets.end());
    q.facet_off.push_back(static_cast<i64>(q.facets.size()));
}
inline TPtr make_qftree(const KPtr &k, const APtr &a, bool word_index, bool keep_source) {
    auto q = std::make_shared<QFTree>();
    q->source_size = k->size();
    q->source_dimension = k->dimension;
    q->word_index = word_index;
    if (a && a->parent.get() != k.get())
        throw std::invalid_argument("different source snapshot");
    const auto nc = a ? a->ncomp : 0;
    std::vector<Simplex> comp(nc);
    if (a)
        for (std::size_t v = 0; v < k->count(0); ++v)
            if (a->vcomp[v] >= 0)
                comp[a->vcomp[v]].push_back(k->verts[v]);
    for (const auto &v : comp)
        append_component(*q, v);
    { // Limit the lifetime of the construction-only source lookup and image.
        auto ix = k->index();
        std::vector<u32> image(k->size());
        u32 next = static_cast<u32>(nc);
        for (std::size_t s = 0; s < k->size(); ++s)
            image[s] = (a && a->mask[s]) ? static_cast<u32>(a->component(k->verts[k->off[s]])) : next++;
        for (std::size_t s = 0; s < k->size(); ++s)
            if (!a || !a->mask[s]) {
                auto v = k->simplex(s);
                Simplex targets;
                if (k->dims[s] > 0)
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        auto f = v;
                        f.erase(f.begin() + i);
                        targets.push_back(image.at(ix.at(key(f))));
                    }
                append_genuine(*q, k->dims[s], static_cast<i64>(s), v, targets);
            }
    }
    if (keep_source) {
        q->archive = k;
        q->archive_mask = a ? a->mask : std::vector<std::uint8_t>(k->size(), 0);
    }
    q->validate_records();
    q->build_indices();
    return q;
}
// ---------------------------------------------------------------------------
// Closed-star (local) quotient presentation.
//
// A componentwise collapse K/_SC A changes the attaching data of a simplex only
// when the simplex has a face in A, i.e. a vertex in V(A). Everything else keeps
// its implicit simplicial attachments and can stay in a simplex tree. The
// simplices of K are therefore partitioned into four classes:
//
//   3 = collapsed : sigma in A                                     (removed)
//   2 = star      : sigma not in A, with a vertex in V(A)          (rewritten)
//   1 = frontier  : sigma disjoint from V(A) that is a face of a star simplex
//   0 = untouched : everything else                                (unchanged)
//
// Classes 1,2,3 together form the closed star of A. The QF-tree built on the
// closed-star pair (Cl st A, A) contains one genuine record per star/frontier
// simplex, with global source ids into K; the untouched part is not copied.
// The frontier records are ordinary regular simplices; they exist so that every
// facet target of a star record is a genuine cell of the local tree.
// ---------------------------------------------------------------------------
struct StarPartition {
    std::vector<std::uint8_t> label; // one entry per simplex of K
    std::array<std::size_t, 4> counts{{0, 0, 0, 0}};
};
inline StarPartition star_partition(const Complex &k, const Subcomplex &a) {
    if (a.parent.get() != &k)
        throw std::invalid_argument("subcomplex belongs to a different complex");
    StarPartition part;
    part.label.assign(k.size(), 0);
    // Vertices of A are exactly the zero-cells with a component label.
    const auto ix = k.index();
    for (std::size_t s = 0; s < k.size(); ++s) {
        if (a.mask[s]) {
            part.label[s] = 3;
            continue;
        }
        for (i64 p = k.off[s]; p < k.off[s + 1]; ++p)
            if (a.component(k.verts[p]) >= 0) {
                part.label[s] = 2;
                break;
            }
    }
    // Frontier: faces of star simplices that are neither star nor collapsed.
    // Faces of a frontier simplex are again disjoint from V(A) and faces of the
    // same star simplex, so a single pass over all subfaces suffices.
    Simplex face;
    std::function<void(const Simplex &)> mark_faces = [&](const Simplex &v) {
        if (v.size() <= 1)
            return;
        for (std::size_t i = 0; i < v.size(); ++i) {
            face = v;
            face.erase(face.begin() + i);
            const auto it = ix.find(key(face));
            if (it == ix.end())
                throw std::logic_error("complex is not closed under faces");
            auto &l = part.label[it->second];
            if (l == 0) {
                l = 1;
                mark_faces(face);
            }
        }
    };
    for (std::size_t s = 0; s < k.size(); ++s)
        if (part.label[s] == 2)
            mark_faces(k.simplex(s));
    for (auto l : part.label)
        ++part.counts[l];
    return part;
}
inline TPtr make_star_qftree(const KPtr &k, const APtr &a, const StarPartition &part, bool word_index) {
    if (!a || a->parent.get() != k.get())
        throw std::invalid_argument("star quotient requires a subcomplex of this complex");
    if (part.label.size() != k->size())
        throw std::invalid_argument("partition does not match the complex");
    auto q = std::make_shared<QFTree>();
    q->source_size = k->size();
    q->source_dimension = k->dimension;
    q->word_index = word_index;
    const auto nc = a->ncomp;
    std::vector<Simplex> comp(nc);
    for (std::size_t v = 0; v < k->count(0); ++v)
        if (a->vcomp[v] >= 0)
            comp[a->vcomp[v]].push_back(k->verts[v]);
    for (const auto &v : comp)
        append_component(*q, v);
    {
        auto ix = k->index();
        std::vector<u32> image(k->size(), std::numeric_limits<u32>::max());
        u32 next = static_cast<u32>(nc);
        for (std::size_t s = 0; s < k->size(); ++s) {
            const auto l = part.label[s];
            if (l == 3)
                image[s] = static_cast<u32>(a->component(k->verts[k->off[s]]));
            else if (l == 1 || l == 2)
                image[s] = next++;
        }
        for (std::size_t s = 0; s < k->size(); ++s)
            if (part.label[s] == 1 || part.label[s] == 2) {
                auto v = k->simplex(s);
                Simplex targets;
                if (k->dims[s] > 0)
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        auto f = v;
                        f.erase(f.begin() + i);
                        const auto t = image.at(ix.at(key(f)));
                        if (t == std::numeric_limits<u32>::max())
                            throw std::logic_error("closed star is not face-closed");
                        targets.push_back(t);
                    }
                append_genuine(*q, k->dims[s], static_cast<i64>(s), v, targets);
            }
    }
    q->validate_records();
    q->build_indices();
    return q;
}
// Relative or quotient chain complex of K/_SC A assembled from the hybrid
// representation: untouched simplices use their implicit simplicial faces,
// star and frontier simplices use the records of the local tree. Generators
// are ordered exactly as in make_chain(k,a,quotient), so the two agree
// column for column; this is the consistency check for the local presentation.
inline CPtr star_chain(const Complex &k, const Subcomplex &a, const QFTree &local, const StarPartition &part,
                       bool quotient) {
    if (local.source_size != k.size() || part.label.size() != k.size())
        throw std::invalid_argument("local tree does not belong to this complex");
    const auto keep = [&](std::size_t s) {
        return part.label[s] != 3;
    };
    const i32 nc = quotient ? a.ncomp : 0;
    // Position of every surviving simplex among the survivors of its dimension.
    std::vector<u32> position(k.size(), std::numeric_limits<u32>::max());
    std::vector<i64> counts(static_cast<std::size_t>(k.dimension + 1), 0);
    for (std::size_t s = 0; s < k.size(); ++s)
        if (keep(s))
            position[s] = static_cast<u32>(counts[k.dims[s]]++);
    if (!counts.empty())
        counts[0] += nc;
    // Local cell id -> source simplex, for facet targets of star records.
    auto c = std::make_shared<ChainComplex>();
    c->quotient = quotient;
    c->counts = counts;
    const auto ix = k.index();
    Simplex face, col;
    for (i32 d = 0; d <= k.dimension; ++d) {
        Boundary out;
        if (d == 0) {
            out.off.assign(static_cast<std::size_t>(counts[0] + 1), 0);
            c->boundaries.push_back(std::move(out));
            continue;
        }
        out.rows = counts[d - 1];
        auto hi = k.range(d);
        for (auto s = hi.first; s < hi.second; ++s)
            if (keep(s)) {
                col.clear();
                if (part.label[s] == 0) {
                    auto v = k.simplex(s);
                    for (std::size_t j = 0; j < v.size(); ++j) {
                        face = v;
                        face.erase(face.begin() + j);
                        col.push_back(static_cast<u32>(nc * (d == 1)) + position.at(ix.at(key(face))));
                    }
                } else {
                    const auto id = local.find_source(k.simplex(s));
                    if (id < 0)
                        throw std::logic_error("star simplex missing from the local tree");
                    for (i64 p = local.facet_off[id]; p < local.facet_off[id + 1]; ++p) {
                        const auto t = local.facets[p];
                        if (t < local.ncomp) {
                            if (quotient && d == 1)
                                col.push_back(t);
                        } else
                            col.push_back(static_cast<u32>(nc * (d == 1)) +
                                          position.at(static_cast<std::size_t>(local.source_ids[t])));
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
        c->boundaries.push_back(std::move(out));
    }
    return c;
}

inline bool qf_map_valid(const QFTree &q, const QFTree &r, const std::vector<u32> &image) {
    if (image.size() != q.size())
        return false;
    for (std::size_t id = 0; id < q.size(); ++id) {
        u32 t = image[id];
        if (t >= r.size())
            return false;
        if (id < q.ncomp && t >= r.ncomp)
            return false;
        if (t >= r.ncomp && (r.dims[t] != q.dims[id] || r.support(t) != q.support(id)))
            return false;
        if (q.dims[id] > 0)
            for (u32 i = 0; i <= static_cast<u32>(q.dims[id]); ++i) {
                auto f = image[q.facets[q.facet_off[id] + i]];
                if (t < r.ncomp ? f != t : f != r.facets[r.facet_off[t] + i])
                    return false;
            }
    }
    return true;
}
struct CollapseResult {
    TPtr tree;
    std::vector<u32> image;
};
inline CollapseResult collapse_qftree(const QFTree &q, std::vector<u32> cells, bool close) {
    if (close)
        cells = q.closure(cells);
    std::vector<std::uint8_t> selected(q.size(), 0);
    for (auto id : cells) {
        q.require_cell(id);
        selected[id] = 1;
    }
    for (auto id : cells)
        for (i64 p = q.facet_off[id]; p < q.facet_off[id + 1]; ++p)
            if (!selected[q.facets[p]])
                throw std::invalid_argument(
                    "collapse selection is not face-closed; use close=True explicitly");
    const auto nv = q.count(0);
    std::vector<u32> parent(nv);
    std::iota(parent.begin(), parent.end(), u32(0));
    auto root = [&](u32 x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto er = q.range(1);
    for (auto id = er.first; id < er.second; ++id)
        if (selected[id]) {
            auto a = root(q.facets[q.facet_off[id]]), b = root(q.facets[q.facet_off[id] + 1]);
            if (a > b)
                std::swap(a, b);
            parent[b] = a;
        }
    // Every old component remains distinguished, selected or not.
    std::map<u32, Simplex> groups;
    for (std::size_t j = 0; j < q.vertex_labels.size(); ++j) {
        auto z = q.vertex_images[j];
        if (z < q.ncomp || selected[z])
            groups[root(z)].push_back(q.vertex_labels[j]);
    }
    std::vector<std::pair<u32, Simplex>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) { return a.second.front() < b.second.front(); });
    auto r = std::make_shared<QFTree>();
    r->source_size = q.source_size;
    r->source_dimension = q.source_dimension;
    r->word_index = q.word_index;
    std::unordered_map<u32, u32> to_component;
    for (const auto &g : sorted) {
        to_component[g.first] = r->ncomp;
        append_component(*r, g.second);
    }
    std::vector<u32> image(q.size());
    u32 next = r->ncomp;
    for (std::size_t id = 0; id < q.size(); ++id) {
        if (id < q.ncomp || selected[id]) {
            u32 z = id < q.ncomp ? static_cast<u32>(id) : q.vertex_image(q.supports[q.support_off[id]]);
            auto rt = root(z);
            if (id >= q.ncomp)
                for (auto v : q.support(id))
                    if (root(q.vertex_image(v)) != rt)
                        throw std::logic_error("selected cell has disconnected vertex images");
            image[id] = to_component.at(rt);
        } else
            image[id] = next++;
    }
    for (std::size_t id = q.ncomp; id < q.size(); ++id)
        if (!selected[id]) {
            Simplex f;
            for (i64 p = q.facet_off[id]; p < q.facet_off[id + 1]; ++p)
                f.push_back(image[q.facets[p]]);
            append_genuine(*r, q.dims[id], q.source_ids[id], q.support(id), f);
        }
    if (q.archive) {
        r->archive = q.archive;
        r->archive_mask = q.archive_mask;
        for (std::size_t id = q.ncomp; id < q.size(); ++id)
            if (selected[id])
                r->archive_mask[q.source_ids[id]] = 1;
    }
    r->validate_records();
    r->build_indices();
    if (!qf_map_valid(q, *r, image))
        throw std::logic_error("collapse did not produce a pointed face morphism");
    return {r, std::move(image)};
}

// Versioned binary table. Explicit little-endian integers; no JSON or pickle.
// FNV-1a detects accidental corruption only; it is not a cryptographic digest.
namespace qfio {
constexpr u64 offset_basis = 14695981039346656037ULL, prime = 1099511628211ULL;
constexpr std::array<unsigned char, 8> magic{{'Q', 'F', 'T', 'R', 'E', 'E', '0', '4'}};
struct Writer {
    std::ofstream out;
    u64 hash = offset_basis;
    explicit Writer(const std::filesystem::path &path) : out(path, std::ios::binary | std::ios::trunc) {
        if (!out)
            throw std::runtime_error("cannot open QF output file");
    }
    void byte(unsigned char b, bool checksum = true) {
        out.put(static_cast<char>(b));
        if (checksum) {
            hash ^= b;
            hash *= prime;
        }
    }
    template <class T>
    void scalar(T v, bool checksum = true) {
        using U = std::make_unsigned_t<T>;
        U x = static_cast<U>(v);
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            byte(static_cast<unsigned char>(x & 255), checksum);
            x >>= 8;
        }
    }
    template <class T>
    void vec(const std::vector<T> &v) {
        scalar<u64>(v.size());
        for (T x : v)
            scalar<T>(x);
    }
    void finish() {
        scalar<u64>(hash, false);
        out.flush();
        if (!out)
            throw std::runtime_error("failed writing QF file");
        out.close();
        if (out.fail())
            throw std::runtime_error("failed closing QF file");
    }
};
struct Reader {
    std::ifstream in;
    u64 hash = offset_basis, remaining = 0;
    Reader(const std::filesystem::path &path, u64 max_bytes) : in(path, std::ios::binary | std::ios::ate) {
        if (!in)
            throw std::runtime_error("cannot open QF input file");
        auto n = in.tellg();
        if (n < 0)
            throw std::runtime_error("cannot determine QF file length");
        remaining = static_cast<u64>(n);
        if (remaining > max_bytes)
            throw std::length_error("QF file exceeds max_bytes limit");
        in.seekg(0);
    }
    unsigned char byte(bool checksum = true) {
        if (!remaining)
            throw std::invalid_argument("truncated QF file");
        int b = in.get();
        if (b == EOF)
            throw std::invalid_argument("truncated QF file");
        --remaining;
        if (checksum) {
            hash ^= static_cast<unsigned char>(b);
            hash *= prime;
        }
        return static_cast<unsigned char>(b);
    }
    template <class T>
    T scalar(bool checksum = true) {
        using U = std::make_unsigned_t<T>;
        U x = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i)
            x |= static_cast<U>(byte(checksum)) << (8 * i);
        if constexpr (std::is_signed_v<T>) {
            if (x > static_cast<U>(std::numeric_limits<T>::max()))
                return static_cast<T>(-1 - static_cast<T>(static_cast<U>(~x)));
        }
        return static_cast<T>(x);
    }
    template <class T>
    std::vector<T> vec(u64 max_count = std::numeric_limits<u64>::max()) {
        auto n = scalar<u64>();
        if (n > max_count)
            throw std::length_error("QF vector exceeds configured limit");
        if (n > remaining / sizeof(T) || n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("invalid QF vector length");
        std::vector<T> v(static_cast<std::size_t>(n));
        for (auto &x : v)
            x = scalar<T>();
        return v;
    }
    void finish() {
        auto expected = hash;
        auto actual = scalar<u64>(false);
        if (actual != expected)
            throw std::invalid_argument("QF checksum mismatch");
        if (remaining)
            throw std::invalid_argument("trailing data in QF file");
    }
};
} // namespace qfio
inline void save_qftree(const QFTree &q, const std::filesystem::path &path) {
    qfio::Writer w(path);
    for (auto c : qfio::magic)
        w.byte(c);
    w.scalar<u32>(1);
    w.scalar<u32>((q.word_index ? 1U : 0U) | (q.archive ? 2U : 0U));
    w.scalar<u64>(q.source_size);
    w.scalar<i32>(q.source_dimension);
    w.scalar<u32>(q.ncomp);
    w.vec(q.dims);
    w.vec(q.source_ids);
    w.vec(q.supports);
    w.vec(q.support_off);
    w.vec(q.facets);
    w.vec(q.facet_off);
    w.vec(q.component_vertices);
    w.vec(q.component_off);
    if (q.archive) {
        w.vec(q.archive->verts);
        w.vec(q.archive->off);
        w.vec(q.archive_mask);
    }
    w.finish();
}
inline TPtr load_qftree(const std::filesystem::path &path, std::size_t max_cells, u64 max_bytes,
                        int index_override = -1) {
    qfio::Reader r(path, max_bytes);
    for (auto c : qfio::magic)
        if (r.byte() != c)
            throw std::invalid_argument("not a QFTREE04 binary file");
    if (r.scalar<u32>() != 1)
        throw std::invalid_argument("unsupported QF binary schema version");
    auto flags = r.scalar<u32>();
    if (flags & ~3U)
        throw std::invalid_argument("unsupported QF feature flags");
    auto q = std::make_shared<QFTree>();
    q->word_index = index_override < 0 ? bool(flags & 1U) : bool(index_override);
    q->source_size = r.scalar<u64>();
    q->source_dimension = r.scalar<i32>();
    q->ncomp = r.scalar<u32>();
    if (q->ncomp > max_cells)
        throw std::length_error("too many QF component points");
    q->dims = r.vec<i32>(max_cells);
    q->source_ids = r.vec<i64>(max_cells);
    q->supports = r.vec<u32>();
    q->support_off = r.vec<i64>(static_cast<u64>(max_cells) + 1);
    q->facets = r.vec<u32>();
    q->facet_off = r.vec<i64>(static_cast<u64>(max_cells) + 1);
    q->component_vertices = r.vec<u32>();
    q->component_off = r.vec<i64>(static_cast<u64>(max_cells) + 1);
    if (flags & 2U) {
        auto v = r.vec<u32>();
        auto o = r.vec<i64>(static_cast<u64>(max_cells) + 1);
        q->archive_mask = r.vec<std::uint8_t>(max_cells);
        q->archive = from_arrays(v, o, max_cells);
    }
    r.finish();
    q->validate_records();
    q->build_indices();
    return q;
}
} // namespace qf

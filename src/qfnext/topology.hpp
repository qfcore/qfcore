// Editable, source-free cellular presentations. Stable IDs and inverse incidence.
// Simplicial cells retain ordered facets (including constant point facets).
// Polygonal 2-cells additionally retain signed edge words; never inferred from d2.
#pragma once
#include "qf_homology.hpp"
#include "linear.hpp"
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <queue>
#include <set>
#include <limits>
namespace nextqf {
using Id = std::int64_t;
using Ids = std::vector<Id>;
// One cell record. kind 0: simplex-type cell of dimension dim with an ordered
// list of dim+1 local facets d_0..d_dim (a facet is either a (dim-1)-cell or a
// vertex, the latter meaning a constant attaching map). kind 2: polygonal
// 2-cell attached along the closed signed edge word `word` starting at `base`;
// tokens are +(edge id+1) / -(edge id+1) so that edge 0 is distinguishable.
struct Cell {
    Id id;
    int dim;
    int kind;
    Ids facets;
    Id base = -1;
    Ids word;
}; // word: +(edge+1), -(edge+1)
// Receipt of an edit: removed/changed/created cell ids, the sparse map of
// collapsed cells to their retained vertex (identity elsewhere), and the
// number of records and facet occurrences actually visited (locality counters).
struct Change {
    Ids removed, changed, created;
    std::map<Id, Id> image;
    std::size_t visited = 0, occurrences = 0;
};
struct DSU {
    std::unordered_map<Id, Id> p;
    Id find(Id x) {
        auto i = p.find(x);
        if (i == p.end()) {
            p[x] = x;
            return x;
        }
        if (i->second != x)
            i->second = find(i->second);
        return i->second;
    }
    void join(Id a, Id b) {
        a = find(a);
        b = find(b);
        if (a != b)
            p[std::max(a, b)] = std::min(a, b);
    }
};
// Editable cellular presentation. `cells` is keyed by stable id; `up` is the
// reverse incidence (cell -> cells whose attaching data mention it), kept
// BEFORE coefficient cancellation, so a cell occurring twice in a word with
// opposite signs is still a topological dependency.
struct Editable {
    std::unordered_map<Id, Cell> cells;
    std::unordered_map<Id, std::unordered_set<Id>> up;
    Id next_id = 0;
    std::uint64_t epoch = 0;
    const Cell &at(Id id) const {
        auto i = cells.find(id);
        if (i == cells.end())
            throw std::out_of_range("unknown/deleted cell ID");
        return i->second;
    }
    static Id edge_id(Id w) {
        if (w == 0 || w == std::numeric_limits<Id>::min())
            throw std::invalid_argument("word tokens must be nonzero signed (edge ID+1)");
        return (w > 0 ? w : -w) - 1;
    }
    Ids ids(int d = -1) const {
        Ids v;
        v.reserve(cells.size());
        for (auto &[i, c] : cells)
            if (d < 0 || c.dim == d)
                v.push_back(i);
        std::sort(v.begin(), v.end());
        return v;
    }
    int dimension() const {
        int d = -1;
        for (auto &[i, c] : cells)
            d = std::max(d, c.dim);
        return d;
    }
    // Cells that the attaching data of c reference: facets for a simplex,
    // basepoint and word edges for a polygon.
    static Ids deps(const Cell &c) {
        Ids v;
        if (c.kind == 2) {
            v.push_back(c.base);
            for (Id w : c.word)
                v.push_back(edge_id(w));
        } else
            v = c.facets;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }
    Id face(Id id, int index) const {
        const auto &c = at(id);
        if (c.dim == 0)
            return id;
        if (c.kind == 2)
            throw std::invalid_argument("polygon does not have simplex-local faces");
        if (index < 0 || index > c.dim)
            throw std::out_of_range("local facet index");
        return c.facets.at(index);
    }
    std::pair<Id, Id> endpoints(Id e) const {
        const auto &c = at(e);
        if (c.dim != 1)
            throw std::invalid_argument("word references a nonedge");
        return {c.facets[1], c.facets[0]};
    }
    // Structural validity of one record against the current cells: facet
    // dimensions, the semisimplicial identities d_i d_j = d_{j-1} d_i (i<j)
    // with vertices absorbing further faces, and composability/closedness of
    // polygon words.
    void check(const Cell &c) const {
        if (c.id < 0 || c.id == std::numeric_limits<Id>::max() || c.dim < 0 || c.dim > 64)
            throw std::invalid_argument("cell ID/dimension out of bounds");
        if (c.kind == 2) {
            if (c.dim != 2 || !c.facets.empty() || at(c.base).dim != 0)
                throw std::invalid_argument("polygon requires dim 2, basepoint, and edge word");
            Id cur = c.base;
            for (Id w : c.word) {
                auto [a, b] = endpoints(edge_id(w));
                if (w < 0)
                    std::swap(a, b);
                if (a != cur)
                    throw std::invalid_argument("attaching word is not a composable path");
                cur = b;
            }
            if (cur != c.base)
                throw std::invalid_argument("attaching path is not closed");
            return;
        }
        if (c.kind != 0 || c.base != -1 || !c.word.empty())
            throw std::invalid_argument("invalid simplicial record");
        if (c.dim == 0) {
            if (!c.facets.empty())
                throw std::invalid_argument("vertex has facets");
            return;
        }
        if (c.facets.size() != static_cast<std::size_t>(c.dim + 1))
            throw std::invalid_argument("wrong number of local facets");
        for (Id f : c.facets) {
            auto &x = at(f);
            if (x.dim != c.dim - 1 && x.dim != 0)
                throw std::invalid_argument("facet dimension not codimension one or constant point");
            if (x.kind == 2)
                throw std::invalid_argument("higher simplex cannot use polygon as a local facet");
        }
        if (c.dim >= 2)
            for (int i = 0; i < c.dim; ++i)
                for (int j = i + 1; j <= c.dim; ++j) {
                    Id a = face(c.facets[j], i), b = face(c.facets[i], j - 1);
                    if (a != b)
                        throw std::invalid_argument("codimension-two face identities fail");
                }
    }
    void link(const Cell &c) {
        for (Id f : deps(c))
            up[f].insert(c.id);
        up.try_emplace(c.id);
    }
    void unlink(const Cell &c) {
        for (Id f : deps(c)) {
            auto i = up.find(f);
            if (i != up.end())
                i->second.erase(c.id);
        }
    }
    Id insert(Cell c) {
        if (cells.count(c.id))
            throw std::invalid_argument("cell ID already active");
        check(c);
        Id id = c.id;
        link(c);
        cells.emplace(id, std::move(c));
        next_id = std::max(next_id, id + 1);
        ++epoch;
        return id;
    }
    Id vertex(Id requested = -1) {
        return insert(Cell{requested < 0 ? next_id : requested, 0, 0, {}, -1, {}});
    }
    Id simplex(int dim, Ids facets, Id requested = -1) {
        return insert(Cell{requested < 0 ? next_id : requested, dim, 0, std::move(facets), -1, {}});
    }
    Id disk(Ids word, Id base, Id requested = -1) {
        return insert(Cell{requested < 0 ? next_id : requested, 2, 2, {}, base, std::move(word)});
    }
    Ids closure(const Ids &seeds) const {
        std::unordered_set<Id> s;
        Ids todo;
        for (Id i : seeds) {
            at(i);
            if (s.insert(i).second)
                todo.push_back(i);
        }
        for (std::size_t k = 0; k < todo.size(); ++k)
            for (Id f : deps(at(todo[k])))
                if (s.insert(f).second)
                    todo.push_back(f);
        std::sort(todo.begin(), todo.end());
        return todo;
    }
    Ids cofaces(Id cell) const {
        at(cell);
        std::unordered_set<Id> s{cell};
        Ids todo{cell};
        for (std::size_t k = 0; k < todo.size(); ++k) {
            auto it = up.find(todo[k]);
            if (it != up.end())
                for (Id f : it->second)
                    if (s.insert(f).second)
                        todo.push_back(f);
        }
        std::sort(todo.begin(), todo.end());
        return todo;
    }
    Ids removal_order(Id id, bool cascade) const {
        auto out = cascade ? cofaces(id) : Ids{id};
        if (!cascade && up.at(id).size())
            throw std::invalid_argument("cell is not maximal; use cascade=True");
        std::sort(out.begin(), out.end(),
                  [&](Id a, Id b) { return at(a).dim != at(b).dim ? at(a).dim > at(b).dim : a > b; });
        return out;
    }
    Change erase(Id id, bool cascade) {
        auto order = removal_order(id, cascade);
        Change ch;
        ch.visited = order.size();
        for (Id i : order) {
            const auto &c = at(i);
            ch.occurrences += c.kind == 2 ? c.word.size() + 1 : c.facets.size();
            unlink(c);
            up.erase(i);
            cells.erase(i);
            ch.removed.push_back(i);
        }
        ++epoch;
        return ch;
    }
    // All rewriting is restricted to immediate cofaces of mapped cells. Higher
    // cofaces retain their stable facet IDs and need not be scanned or rewritten.
    // Core of every quotient operation. `image` sends cells to their new
    // representative (a retained vertex). Cells with a != image[a] disappear.
    // Only the IMMEDIATE cofaces of a removed cell need rewriting: their facet
    // slots (or polygon basepoint/word) are redirected through `image`, a
    // collapsed edge is dropped from a word since it becomes a constant path.
    // Higher cofaces keep their stable facet ids and inherit the change.
    Change remap_and_remove(const std::map<Id, Id> &image) {
        Change ch;
        ch.image = image;
        std::unordered_set<Id> gone, affected;
        for (auto [a, b] : image) {
            at(a);
            if (at(b).dim != 0)
                throw std::invalid_argument("collapse target must be a retained vertex");
            if (a != b)
                gone.insert(a);
        }
        for (Id a : gone)
            for (Id co : up.at(a))
                if (!gone.count(co))
                    affected.insert(co);
        // Prepare changed records first. Public callers validate closed selections or
        // vertex pairs before editing; the postcondition below checks the construction.
        std::vector<Cell> replacements;
        replacements.reserve(affected.size());
        for (Id id : affected) {
            Cell c = at(id);
            ch.occurrences += c.kind == 2 ? c.word.size() + 1 : c.facets.size();
            if (c.kind == 2) {
                auto p = image.find(c.base);
                if (p != image.end())
                    c.base = p->second;
                Ids w;
                for (Id t : c.word)
                    if (!gone.count(edge_id(t)))
                        w.push_back(t);
                c.word = std::move(w);
            } else
                for (Id &f : c.facets) {
                    auto p = image.find(f);
                    if (p != image.end())
                        f = p->second;
                }
            replacements.push_back(std::move(c));
        }
        for (auto &c : replacements)
            unlink(at(c.id));
        for (Id id : gone)
            unlink(at(id));
        for (Id id : gone) {
            cells.erase(id);
            up.erase(id);
            ch.removed.push_back(id);
        }
        for (auto &c : replacements) {
            Id id = c.id;
            cells.at(id) = std::move(c);
            link(at(id));
            ch.changed.push_back(id);
        }
        for (Id id : ch.changed)
            check(at(id)); // construction guarantees validity for closed collapses
        ch.visited = image.size() + affected.size();
        std::sort(ch.removed.begin(), ch.removed.end());
        std::sort(ch.changed.begin(), ch.changed.end());
        ++epoch;
        return ch;
    }
    // Componentwise quotient by a closed cell subcomplex: each connected
    // component (in the incidence graph) is sent to its minimum-id vertex.
    // This is the topological operation Q -> Q/B, not an elementary collapse.
    Change collapse(const Ids &selected, bool close) {
        Ids b = close ? closure(selected) : selected;
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        std::unordered_set<Id> s(b.begin(), b.end());
        DSU d;
        for (Id i : b) {
            at(i);
            d.find(i);
            for (Id f : deps(at(i))) {
                if (!s.count(f))
                    throw std::invalid_argument("collapse set is not a subcomplex");
                d.join(i, f);
            }
        }
        std::map<Id, Id> reps;
        for (Id i : b)
            if (at(i).dim == 0) {
                Id r = d.find(i);
                auto it = reps.find(r);
                if (it == reps.end() || i < it->second)
                    reps[r] = i;
            }
        std::map<Id, Id> im;
        for (Id i : b)
            im[i] = reps.at(d.find(i));
        return remap_and_remove(im);
    }
    Change identify_vertices(const std::vector<std::pair<Id, Id>> &pairs) {
        DSU d;
        for (auto [a, b] : pairs) {
            if (at(a).dim || at(b).dim)
                throw std::invalid_argument("point gluing requires vertices");
            d.join(a, b);
        }
        std::map<Id, Id> im;
        for (auto [a, b] : pairs) {
            im[a] = d.find(a);
            im[b] = d.find(b);
        }
        return remap_and_remove(im);
    }
    bool validate() const {
        for (auto &[i, c] : cells) {
            check(c);
            for (Id f : deps(c))
                if (!up.at(f).count(i))
                    throw std::logic_error("inverse incidence missing");
        }
        for (auto &[f, co] : up) {
            at(f);
            for (Id c : co) {
                auto ds = deps(at(c));
                if (!std::binary_search(ds.begin(), ds.end(), f))
                    throw std::logic_error("inverse incidence has stale occurrence");
            }
        }
        return true;
    }
    // Integer cellular boundary: alternating sum of the genuine codimension-one
    // facets, or the exponent sum of the edges of a polygon word. Constant
    // facets of cells of dimension >= 2 contribute nothing.
    std::vector<std::pair<Id, Id>> signed_boundary(Id id) const {
        auto &c = at(id);
        std::map<Id, Id> out;
        if (c.kind == 2) {
            for (Id w : c.word)
                out[edge_id(w)] += w > 0 ? 1 : -1;
        } else if (c.dim > 0)
            for (std::size_t k = 0; k < c.facets.size(); ++k)
                if (at(c.facets[k]).dim == c.dim - 1)
                    out[c.facets[k]] += k % 2 ? -1 : 1;
        std::vector<std::pair<Id, Id>> v;
        for (auto p : out)
            if (p.second)
                v.push_back(p);
        return v;
    }
    Ids boundary(Id id) const {
        Ids v;
        for (auto [i, c] : signed_boundary(id))
            if (c % 2)
                v.push_back(i);
        return v;
    }
    std::size_t incidence_count() const {
        std::size_t n = 0;
        for (auto &[i, c] : cells)
            n += c.kind == 2 ? c.word.size() + 1 : c.facets.size();
        return n;
    }
    static std::shared_ptr<Editable> from_qft(const qf::QFTree &q) {
        auto e = std::make_shared<Editable>();
        for (qf::u32 i = 0; i < q.size(); ++i) {
            Ids fs(q.facets.begin() + q.facet_off[i], q.facets.begin() + q.facet_off[i + 1]);
            e->insert(Cell{static_cast<Id>(i), q.dims[i], 0, std::move(fs), -1, {}});
        }
        e->epoch = 0;
        return e;
    }
};
struct ChainView {
    qf::CPtr chain;
    std::vector<Ids> ids;
    std::unordered_map<Id, qf::u32> index;
    explicit ChainView(const Editable &e, int maxdim = -1) {
        int d = std::max(maxdim, e.dimension());
        chain = std::make_shared<qf::ChainComplex>();
        ids.resize(std::max(0, d + 1));
        for (int k = 0; k <= d; ++k) {
            ids[k] = e.ids(k);
            chain->counts.push_back(ids[k].size());
            for (std::size_t j = 0; j < ids[k].size(); ++j) {
                if (j >= std::numeric_limits<qf::u32>::max())
                    throw std::length_error("chain basis exceeds uint32");
                index[ids[k][j]] = j;
            }
        }
        for (int k = 0; k <= d; ++k) {
            qf::Boundary b;
            b.rows = k ? ids[k - 1].size() : 0;
            for (Id i : ids[k]) {
                for (Id f : e.boundary(i))
                    b.idx.push_back(index.at(f));
                b.off.push_back(b.idx.size());
            }
            chain->boundaries.push_back(std::move(b));
        }
    }
};
inline ops::Images images(const Editable &a, const Editable &b, const std::map<Id, Id> &im, int maxdim) {
    ChainView x(a, maxdim), y(b, maxdim);
    ops::Images out(x.ids.size());
    for (std::size_t d = 0; d < x.ids.size(); ++d)
        for (Id i : x.ids[d]) {
            auto it = im.find(i);
            Id t = it == im.end() ? i : it->second;
            if (t < 0)
                out[d].push_back(-1);
            else if (b.at(t).dim == static_cast<int>(d))
                out[d].push_back(y.index.at(t));
            else
                out[d].push_back(-1);
        }
    return out;
}
// Strict cellular-map check: images must commute with the ordered attaching
// data (slot by slot, word by word), which is stronger than dF = Fd. A torus and
// a wedge S1 v S1 v S2 have identical boundary matrices, but the "identity" cell
// map between them is rejected because their 2-cell attaching words differ.
inline bool cellular_map_valid(const Editable &a, const Editable &b, const std::map<Id, Id> &im) {
    auto target = [&](Id i) {
        auto p = im.find(i);
        Id t = p == im.end() ? i : p->second;
        b.at(t);
        return t;
    };
    for (Id i : a.ids()) {
        const auto &c = a.at(i);
        Id tid = target(i);
        const auto &t = b.at(tid);
        if (t.dim == 0) {
            for (Id f : Editable::deps(c))
                if (target(f) != tid)
                    return false;
            continue;
        }
        if (t.dim != c.dim || t.kind != c.kind)
            return false;
        if (c.kind == 0) {
            for (std::size_t j = 0; j < c.facets.size(); ++j)
                if (target(c.facets[j]) != t.facets[j])
                    return false;
        } else {
            if (target(c.base) != t.base)
                return false;
            Ids word;
            for (Id w : c.word) {
                Id f = target(Editable::edge_id(w));
                if (b.at(f).dim == 0)
                    continue;
                if (b.at(f).dim != 1)
                    return false;
                word.push_back(w > 0 ? f + 1 : -f - 1);
            }
            if (word != t.word)
                return false;
        }
    }
    return true;
}
struct Glued {
    std::shared_ptr<Editable> object;
    std::vector<std::map<Id, Id>> maps;
};
inline Glued glue(const std::vector<std::shared_ptr<Editable>> &parts,
                  const std::vector<std::array<Id, 4>> &pairs) {
    Glued g;
    g.object = std::make_shared<Editable>();
    Id id = 0;
    g.maps.resize(parts.size());
    for (std::size_t k = 0; k < parts.size(); ++k)
        for (Id c : parts[k]->ids())
            g.maps[k][c] = id++;
    for (std::size_t k = 0; k < parts.size(); ++k) {
        auto &m = g.maps[k];
        for (int d = 0; d <= parts[k]->dimension(); ++d)
            for (Id i : parts[k]->ids(d)) {
                Cell c = parts[k]->at(i);
                c.id = m.at(i);
                for (Id &f : c.facets)
                    f = m.at(f);
                if (c.kind == 2) {
                    c.base = m.at(c.base);
                    for (Id &w : c.word) {
                        Id t = m.at(Editable::edge_id(w)) + 1;
                        w = w > 0 ? t : -t;
                    }
                }
                g.object->insert(std::move(c));
            }
    }
    std::vector<std::pair<Id, Id>> vp;
    for (auto p : pairs) {
        if (p[0] < 0 || p[2] < 0 || p[0] >= static_cast<Id>(parts.size()) ||
            p[2] >= static_cast<Id>(parts.size()))
            throw std::out_of_range("module index");
        vp.push_back({g.maps[p[0]].at(p[1]), g.maps[p[2]].at(p[3])});
    }
    auto ch = g.object->identify_vertices(vp);
    for (auto &m : g.maps)
        for (auto &[a, b] : m) {
            auto it = ch.image.find(b);
            if (it != ch.image.end())
                b = it->second;
        }
    return g;
}
inline Ids free_reduce(const Ids &w) {
    Ids out;
    for (Id i : w) {
        if (!out.empty() && out.back() == -i)
            out.pop_back();
        else
            out.push_back(i);
    }
    return out;
}
struct Presentation {
    Id root;
    Ids vertices, generators, relator_cells;
    std::vector<Ids> relators;
    Ids tree_edges;
};
// Edge-path presentation of pi_1 of each component: choose a maximal forest,
// nonforest edges are generators, each 2-cell contributes its boundary word
// (d_2 d_0 d_1^{-1} for an ordered triangle, the stored word for a polygon)
// with forest and constant steps removed.
inline std::vector<Presentation> presentation(const Editable &e) {
    DSU all, tree;
    for (Id v : e.ids(0)) {
        all.find(v);
        tree.find(v);
    }
    std::unordered_set<Id> forest;
    for (Id edge : e.ids(1)) {
        auto [a, b] = e.endpoints(edge);
        all.join(a, b);
        if (tree.find(a) != tree.find(b)) {
            tree.join(a, b);
            forest.insert(edge);
        }
    }
    std::map<Id, Presentation> out;
    for (Id v : e.ids(0)) {
        Id r = all.find(v);
        out[r].root = r;
        out[r].vertices.push_back(v);
    }
    std::unordered_map<Id, Id> gen;
    for (Id edge : e.ids(1)) {
        auto [a, b] = e.endpoints(edge);
        auto &p = out[all.find(a)];
        if (forest.count(edge))
            p.tree_edges.push_back(edge);
        else {
            p.generators.push_back(edge);
            gen[edge] = p.generators.size();
        }
    }
    for (Id f : e.ids(2)) {
        const auto &c = e.at(f);
        Ids w;
        Id root;
        if (c.kind == 2) {
            w = c.word;
            root = all.find(c.base);
        } else {
            root = all.find(e.face(e.face(f, 0), 0));
            const int indices[] = {2, 0, 1};
            for (int j : indices) {
                Id a = c.facets[j];
                if (e.at(a).dim == 1)
                    w.push_back((j == 1 ? -1 : 1) * (a + 1));
            }
        }
        Ids r;
        for (Id v : w) {
            Id edge = Editable::edge_id(v);
            if (!forest.count(edge))
                r.push_back((v > 0 ? 1 : -1) * gen.at(edge));
        }
        out[root].relator_cells.push_back(f);
        out[root].relators.push_back(free_reduce(r));
    }
    std::vector<Presentation> result;
    for (auto &[r, p] : out)
        result.push_back(std::move(p));
    return result;
}
} // namespace nextqf

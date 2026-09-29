#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gudhi/filtered_zigzag_persistence.h>
#include <gudhi/Simplex_tree.h>
#include "algebra.hpp"
namespace py = pybind11;
using namespace nextqf;
template <class T>
std::shared_ptr<T> getcap(py::handle h, const char *n) {
    auto p = static_cast<std::shared_ptr<T> *>(PyCapsule_GetPointer(h.ptr(), n));
    if (!p)
        throw py::error_already_set();
    return *p;
}
py::capsule box(qf::CPtr c) {
    auto p = new qf::CPtr(std::move(c));
    return py::capsule(p, "qfcore.ChainComplex.v3", [](PyObject *o) {
        delete static_cast<qf::CPtr *>(PyCapsule_GetPointer(o, "qfcore.ChainComplex.v3"));
    });
}
struct ZZOptions : Gudhi::zigzag_persistence::Default_filtered_zigzag_options {
    using Cell_key = Id;
};
struct Zigzag {
    using G = Gudhi::zigzag_persistence::Filtered_zigzag_persistence_with_storage<ZZOptions>;
    G zz;
    std::unordered_map<Id, std::pair<int, Ids>> active;
    std::unordered_map<Id, std::unordered_set<Id>> up;
    double time = -1;
    void checktime(double t) {
        if (!std::isfinite(t) || t < time)
            throw std::invalid_argument("zigzag times must be finite and nondecreasing");
    }
    void insert(Id id, Ids boundary, int dim, double t) {
        checktime(t);
        if (active.count(id) || dim < 0)
            throw std::invalid_argument("duplicate cell or invalid dimension");
        std::sort(boundary.begin(), boundary.end());
        if (std::adjacent_find(boundary.begin(), boundary.end()) != boundary.end())
            throw std::invalid_argument("boundary must already be reduced modulo 2");
        Ids dd;
        for (Id f : boundary) {
            if (!active.count(f) || active.at(f).first != dim - 1)
                throw std::invalid_argument("boundary references missing cell/wrong dimension");
            auto &c = active.at(f).second;
            Ids tmp;
            std::set_symmetric_difference(dd.begin(), dd.end(), c.begin(), c.end(), std::back_inserter(tmp));
            dd = std::move(tmp);
        }
        if (!dd.empty())
            throw std::invalid_argument("boundary squared nonzero");
        zz.insert_cell(id, boundary, dim, t);
        active[id] = {dim, boundary};
        for (Id f : boundary)
            up[f].insert(id);
        up.try_emplace(id);
        time = t;
    }
    void remove(Id id, double t) {
        checktime(t);
        auto it = active.find(id);
        if (it == active.end() || !up.at(id).empty())
            throw std::invalid_argument("zigzag removal requires an active algebraically maximal cell");
        zz.remove_cell(id, t);
        for (Id f : it->second.second)
            up[f].erase(id);
        up.erase(id);
        active.erase(id);
        time = t;
    }
    std::vector<std::tuple<int, double, double>> bars() {
        std::vector<std::tuple<int, double, double>> v;
        for (auto &b : zz.get_persistence_diagram())
            v.emplace_back(b.dim, b.birth, b.death);
        std::sort(v.begin(), v.end());
        return v;
    }
};
struct SimplexStore {
    Gudhi::Simplex_tree<> st;
    void insert(const std::vector<int> &s) {
        if (s.empty() || !std::is_sorted(s.begin(), s.end()) ||
            std::adjacent_find(s.begin(), s.end()) != s.end())
            throw std::invalid_argument("sorted nonempty simplex required");
        st.insert_simplex_and_subfaces(s);
    }
    void remove(const std::vector<int> &s) {
        auto h = st.find(s);
        if (h == st.null_simplex())
            throw std::invalid_argument("missing simplex");
        if (st.star_simplex_range(h).size() > 1)
            throw std::invalid_argument("nonmaximal simplex");
        st.remove_maximal_simplex(h);
    }
    std::size_t size() { return st.num_simplices(); }
};
PYBIND11_MODULE(_native, m) {
    m.attr("version") = QF_PACKAGE_VERSION;
    m.attr("gudhi_version") = "3.13.0";
    m.attr("compiler") = __VERSION__;
    py::class_<Cell>(m, "Cell")
        .def(py::init<Id, int, int, Ids, Id, Ids>(), py::arg("id"), py::arg("dimension"), py::arg("kind") = 0,
             py::arg("facets") = Ids{}, py::arg("base") = -1, py::arg("word") = Ids{})
        .def_readonly("id", &Cell::id)
        .def_readonly("dimension", &Cell::dim)
        .def_readonly("kind", &Cell::kind)
        .def_readonly("facets", &Cell::facets)
        .def_readonly("base", &Cell::base)
        .def_readonly("word", &Cell::word);
    py::class_<Change>(m, "Change")
        .def_readonly("removed", &Change::removed)
        .def_readonly("changed", &Change::changed)
        .def_readonly("created", &Change::created)
        .def_readonly("image", &Change::image)
        .def_readonly("visited_cells", &Change::visited)
        .def_readonly("touched_occurrences", &Change::occurrences);
    py::class_<Editable, std::shared_ptr<Editable>>(m, "Editable")
        .def(py::init<>())
        .def_static("from_qft",
                    [](py::handle h) {
                        auto q = getcap<qf::QFTree>(h, "qfcore.QFTree.v4");
                        py::gil_scoped_release r;
                        return Editable::from_qft(*q);
                    })
        .def(
            "rebuilt",
            [](const Editable &e) {
                auto r = std::make_shared<Editable>();
                for (int d = 0; d <= e.dimension(); ++d)
                    for (Id i : e.ids(d))
                        r->insert(e.at(i));
                r->next_id = e.next_id;
                return r;
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "clone", [](const Editable &e) { return std::make_shared<Editable>(e); },
            py::call_guard<py::gil_scoped_release>())
        .def("cell", [](const Editable &e, Id i) { return e.at(i); })
        .def("insert_record", &Editable::insert, py::call_guard<py::gil_scoped_release>())
        .def("ids", &Editable::ids, py::arg("dimension") = -1)
        .def("dimension", &Editable::dimension)
        .def("size", [](const Editable &e) { return e.cells.size(); })
        .def("vertex", &Editable::vertex, py::arg("id") = -1)
        .def("simplex", &Editable::simplex, py::arg("dimension"), py::arg("facets"), py::arg("id") = -1)
        .def("disk", &Editable::disk, py::arg("word"), py::arg("base"), py::arg("id") = -1)
        .def("closure", &Editable::closure, py::call_guard<py::gil_scoped_release>())
        .def("cofaces", &Editable::cofaces, py::call_guard<py::gil_scoped_release>())
        .def("removal_order", &Editable::removal_order)
        .def("erase", &Editable::erase, py::arg("cell"), py::arg("cascade") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("collapse", &Editable::collapse, py::arg("cells"), py::arg("close") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("identify_vertices", &Editable::identify_vertices, py::call_guard<py::gil_scoped_release>())
        .def("boundary", &Editable::boundary)
        .def("signed_boundary", &Editable::signed_boundary)
        .def("validate", &Editable::validate, py::call_guard<py::gil_scoped_release>())
        .def("incidence_count", &Editable::incidence_count)
        .def_readonly("epoch", &Editable::epoch)
        .def_readonly("next_id", &Editable::next_id)
        .def("reserve_ids",
             [](Editable &e, Id next) {
                 if (next < e.next_id)
                     throw std::invalid_argument("next_id is smaller than active IDs");
                 e.next_id = next;
             })
        .def(
            "chain_data",
            [](const Editable &e, int d) {
                std::unique_ptr<ChainView> v;
                {
                    py::gil_scoped_release r;
                    v = std::make_unique<ChainView>(e, d);
                }
                return py::make_tuple(box(v->chain), v->ids);
            },
            py::arg("max_dimension") = -1)
        .def(
            "chain",
            [](const Editable &e, int d) {
                qf::CPtr c;
                {
                    py::gil_scoped_release r;
                    c = ChainView(e, d).chain;
                }
                return box(c);
            },
            py::arg("max_dimension") = -1)
        .def(
            "basis", [](const Editable &e, int d) { return ChainView(e, d).ids; },
            py::arg("max_dimension") = -1);
    py::class_<Glued>(m, "Glued").def_readonly("object", &Glued::object).def_readonly("maps", &Glued::maps);
    m.def("cellular_map_valid", &cellular_map_valid, py::call_guard<py::gil_scoped_release>());
    m.def("glue", &glue, py::call_guard<py::gil_scoped_release>());
    m.def(
        "basis_images",
        [](const std::vector<Ids> &source, const std::vector<Ids> &target, const std::map<Id, Id> &im) {
            if (source.size() != target.size())
                throw std::invalid_argument("degree mismatch");
            ops::Images out(source.size());
            for (std::size_t d = 0; d < source.size(); ++d) {
                std::unordered_map<Id, qf::i64> idx;
                for (std::size_t j = 0; j < target[d].size(); ++j)
                    idx[target[d][j]] = j;
                for (Id i : source[d]) {
                    auto p = im.find(i);
                    Id t = p == im.end() ? i : p->second;
                    auto it = idx.find(t);
                    out[d].push_back(it == idx.end() ? -1 : it->second);
                }
            }
            return out;
        },
        py::call_guard<py::gil_scoped_release>());
    m.def("images", &images, py::arg("source"), py::arg("target"), py::arg("image"), py::arg("max_dimension"),
          py::call_guard<py::gil_scoped_release>());
    py::class_<Presentation>(m, "Presentation")
        .def_readonly("root", &Presentation::root)
        .def_readonly("vertices", &Presentation::vertices)
        .def_readonly("generators", &Presentation::generators)
        .def_readonly("relator_cells", &Presentation::relator_cells)
        .def_readonly("relators", &Presentation::relators)
        .def_readonly("tree_edges", &Presentation::tree_edges);
    m.def("presentation", &presentation, py::call_guard<py::gil_scoped_release>());
    py::class_<Cohomology>(m, "Cohomology")
        .def(py::init<const Editable &, std::size_t, int>(), py::arg("space"),
             py::arg("fill_limit") = 10000000, py::arg("max_dimension") = -1,
             py::call_guard<py::gil_scoped_release>())
        .def("betti", &Cohomology::betti)
        .def("cocycles", &Cohomology::cocycles)
        .def("basis", [](const Cohomology &c) { return c.view.ids; })
        .def("product", &Cohomology::product, py::call_guard<py::gil_scoped_release>())
        .def("cochain_product", &Cohomology::cochain_product, py::call_guard<py::gil_scoped_release>())
        .def("coordinates", &Cohomology::coord);
    m.def("pullback", &pullback, py::call_guard<py::gil_scoped_release>());
    m.def("map_barcode", &map_barcode, py::arg("dimensions"), py::arg("maps"), py::arg("directions"),
          py::arg("fill_limit") = 1000000, py::arg("max_total_dimension") = 512,
          py::call_guard<py::gil_scoped_release>());
    py::class_<Zigzag>(m, "Zigzag")
        .def(py::init<>())
        .def("insert", &Zigzag::insert, py::call_guard<py::gil_scoped_release>())
        .def("remove", &Zigzag::remove, py::call_guard<py::gil_scoped_release>())
        .def("bars", &Zigzag::bars);
    py::class_<SimplexStore>(m, "SimplexStore")
        .def(py::init<>())
        .def("insert", &SimplexStore::insert, py::call_guard<py::gil_scoped_release>())
        .def("remove", &SimplexStore::remove, py::call_guard<py::gil_scoped_release>())
        .def("size", &SimplexStore::size);
}

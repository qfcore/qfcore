#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qf_homology.hpp"
#include "gudhi_state.hpp"
namespace py = pybind11;
using ops::Images;
using ops::M;
using ops::V;
using qf::i64;
using qf::u32;
template <class T>
std::shared_ptr<T> get(py::handle h, const char *name) {
    auto p = static_cast<std::shared_ptr<T> *>(PyCapsule_GetPointer(h.ptr(), name));
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
qf::CPtr gc(py::handle h) {
    return get<qf::ChainComplex>(h, "qfcore.ChainComplex.v3");
}
qf::TPtr gt(py::handle h) {
    return get<qf::QFTree>(h, "qfcore.QFTree.v4");
}
template <class T>
std::vector<T> av(py::array_t<T, py::array::c_style> x) {
    if (x.ndim() != 1)
        throw std::invalid_argument("expected 1D contiguous array");
    return std::vector<T>(x.data(), x.data() + x.size());
}
PYBIND11_MODULE(_ops, m) {
    m.attr("version") = QF_PACKAGE_VERSION;
    m.attr("gudhi_version") = "3.13.0";
    m.attr("compiler") = __VERSION__;
    py::class_<ops::TreeData, std::shared_ptr<ops::TreeData>>(m, "TreeData")
        .def(py::init([](py::array_t<u32, py::array::c_style> v, py::array_t<i64, py::array::c_style> o) {
            auto a = av(v);
            auto b = av(o);
            py::gil_scoped_release r;
            return std::make_shared<ops::TreeData>(a, b);
        }))
        .def("num_simplices", [](const ops::TreeData &d) { return d.order.size(); })
        .def("serialized_bytes", [](const ops::TreeData &d) { return d.st.get_serialization_size(); });
    py::class_<ops::TreeState, std::shared_ptr<ops::TreeState>>(m, "TreeState")
        .def(py::init([](std::shared_ptr<ops::TreeData> d, py::array_t<std::uint8_t, py::array::c_style> x) {
            auto a = av(x);
            py::gil_scoped_release r;
            return std::make_shared<ops::TreeState>(d, std::move(a));
        }))
        .def("absorb", &ops::TreeState::absorb, py::call_guard<py::gil_scoped_release>())
        .def("snapshot", &ops::TreeState::snapshot, py::call_guard<py::gil_scoped_release>())
        .def("mask", [](const ops::TreeState &s) { return s.mask; });
    py::class_<ops::Snapshot>(m, "Snapshot")
        .def("chain", [](const ops::Snapshot &s) { return box(s.chain); })
        .def("payload_bytes", [](const ops::Snapshot &s) {
            return s.chain->payload_bytes() + 8 * s.remap.size() +
                   4 * (s.zero_reps.size() + s.vertex_image.size());
        });
    m.def("snapshot_images", &ops::snapshot_images, py::call_guard<py::gil_scoped_release>());
    m.def("select_sources", [](py::handle h, V ids) {
        auto q = gt(h);
        py::gil_scoped_release r;
        std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::invalid_argument("duplicate selected source ID");
        V out;
        std::size_t next = 0;
        for (u32 j = q->ncomp; j < q->size() && next < ids.size(); ++j) {
            if (q->source_ids[j] == ids[next]) {
                out.push_back(j);
                ++next;
            }
        }
        if (next != ids.size())
            throw std::invalid_argument("selected simplex has already been absorbed or is unknown");
        return out;
    });
    m.def("qf_images", [](py::handle a, py::handle b, V image) {
        auto x = gt(a), y = gt(b);
        py::gil_scoped_release r;
        Images f;
        for (int d = 0; d <= x->source_dimension; ++d)
            f.push_back(qf::qf_chain_image(*x, *y, image, d, false));
        return f;
    });
    m.def("compose_images", &ops::compose_images, py::call_guard<py::gil_scoped_release>());
    m.def("chain_map_valid", [](py::handle a, py::handle b, const Images &f, std::size_t lim) {
        auto x = gc(a), y = gc(b);
        py::gil_scoped_release r;
        return ops::chain_map_valid(*x, *y, f, lim);
    });
    m.def("dd_zero", [](py::handle a, std::size_t lim) {
        auto x = gc(a);
        py::gil_scoped_release r;
        return ops::dd_zero(*x, lim);
    });
    py::class_<ops::Homology, std::shared_ptr<ops::Homology>>(m, "Homology")
        .def(py::init([](py::handle h, std::size_t lim) {
            auto c = gc(h);
            py::gil_scoped_release r;
            return std::make_shared<ops::Homology>(c, lim);
        }))
        .def("betti", &ops::Homology::betti)
        .def("cycles", &ops::Homology::cycles)
        .def("payload_bytes", &ops::Homology::payload);
    m.def("induced", &ops::induced, py::call_guard<py::gil_scoped_release>());
    m.def("compose", &ops::compose, py::call_guard<py::gil_scoped_release>());
    m.def("map_invariants", [](const std::vector<M> &f, const std::vector<i64> &target, std::size_t lim) {
        py::gil_scoped_release release;
        std::vector<std::tuple<i64, M, M>> out;
        if (f.size() != target.size())
            throw std::invalid_argument("target dimensions mismatch");
        for (std::size_t d = 0; d < f.size(); ++d) {
            ops::Span span(lim);
            for (const auto &v : f[d]) {
                for (auto x : v)
                    if (x >= target[d])
                        throw std::out_of_range("induced matrix row");
                span.insert(v);
            }
            i64 r = span.pivots.size();
            M co;
            for (i64 j = 0; j < target[d]; ++j) {
                auto z = span.reduce(V{static_cast<u32>(j)}).first;
                if (!z.empty()) {
                    co.push_back(z);
                    span.store(std::move(z), {});
                }
            }
            out.emplace_back(r, ops::kernel(f[d], lim), std::move(co));
        }
        return out;
    });
    m.def("chain_from_columns", [](std::vector<i64> counts, const std::vector<M> &bs) {
        auto c = std::make_shared<qf::ChainComplex>();
        if (counts.size() != bs.size())
            throw std::invalid_argument("degree mismatch");
        c->counts = counts;
        for (std::size_t d = 0; d < counts.size(); ++d) {
            if (counts[d] < 0 || bs[d].size() != static_cast<std::size_t>(counts[d]))
                throw std::invalid_argument("column mismatch");
            qf::Boundary b;
            b.rows = d ? counts[d - 1] : 0;
            for (auto col : bs[d]) {
                col = ops::parity(std::move(col));
                for (auto i : col)
                    if (i >= b.rows)
                        throw std::invalid_argument("row out of bounds");
                b.idx.insert(b.idx.end(), col.begin(), col.end());
                b.off.push_back(b.idx.size());
            }
            c->boundaries.push_back(std::move(b));
        }
        return box(c);
    });
}

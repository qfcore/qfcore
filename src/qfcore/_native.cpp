#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "core.hpp"
#include "qftree.hpp"
#include "qf_homology.hpp"
#include <exception>
#include <type_traits>

namespace {
constexpr const char *K_NAME = "qfcore.Complex.v3";
constexpr const char *A_NAME = "qfcore.Subcomplex.v3";
constexpr const char *C_NAME = "qfcore.ChainComplex.v3";
constexpr const char *T_NAME = "qfcore.QFTree.v4";
constexpr const char *B_NAME = "qfcore.Boundary.v3";
using BPtr = std::shared_ptr<qf::Boundary>;
struct PythonError {};
struct AllowThreads {
    PyThreadState *state;
    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }
    AllowThreads(const AllowThreads &) = delete;
};
template <class F>
auto native_run(F &&f) {
    AllowThreads unlocked;
    return f();
}

#define QF_TRY try {
#define QF_CATCH                                                                                             \
    }                                                                                                        \
    catch (const PythonError &) {                                                                            \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (const std::bad_alloc &) {                                                                         \
        return PyErr_NoMemory();                                                                             \
    }                                                                                                        \
    catch (const std::length_error &e) {                                                                     \
        PyErr_SetString(PyExc_MemoryError, e.what());                                                        \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (const std::invalid_argument &e) {                                                                 \
        PyErr_SetString(PyExc_ValueError, e.what());                                                         \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (const std::out_of_range &e) {                                                                     \
        PyErr_SetString(PyExc_IndexError, e.what());                                                         \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (const std::overflow_error &e) {                                                                   \
        PyErr_SetString(PyExc_OverflowError, e.what());                                                      \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (const std::exception &e) {                                                                        \
        PyErr_SetString(PyExc_RuntimeError, e.what());                                                       \
        return nullptr;                                                                                      \
    }                                                                                                        \
    catch (...) {                                                                                            \
        PyErr_SetString(PyExc_RuntimeError, "unknown native exception");                                     \
        return nullptr;                                                                                      \
    }

template <class T>
void capsule_delete(PyObject *obj) {
    const char *name = PyCapsule_GetName(obj);
    auto p = static_cast<std::shared_ptr<T> *>(PyCapsule_GetPointer(obj, name));
    if (p)
        delete p;
    else
        PyErr_Clear();
}
template <class T>
PyObject *capsule(std::shared_ptr<T> p, const char *name) {
    auto boxed = std::make_unique<std::shared_ptr<T>>(std::move(p));
    PyObject *o = PyCapsule_New(boxed.get(), name, capsule_delete<T>);
    if (!o)
        throw PythonError{};
    boxed.release();
    return o;
}
template <class T>
std::shared_ptr<T> get(PyObject *o, const char *name) {
    auto p = static_cast<std::shared_ptr<T> *>(PyCapsule_GetPointer(o, name));
    if (!p)
        throw PythonError{};
    return *p;
}
qf::APtr optional_sub(PyObject *a, const qf::KPtr &k) {
    if (a == Py_None)
        return {};
    auto sub = get<qf::Subcomplex>(a, A_NAME);
    if (sub->parent.get() != k.get())
        throw std::invalid_argument("subcomplex belongs to a different complex or an earlier snapshot");
    return sub;
}
struct ReadBuffer {
    Py_buffer b{};
    explicit ReadBuffer(PyObject *obj) {
        if (PyObject_GetBuffer(obj, &b, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0)
            throw PythonError{};
    }
    ~ReadBuffer() { PyBuffer_Release(&b); }
};
template <class T>
std::vector<T> read(PyObject *obj) {
    ReadBuffer buf(obj);
    const auto &b = buf.b;
    const char *f = b.format ? b.format : "";
    if (*f == '@' || *f == '=')
        ++f;
    bool valid = false;
    if constexpr (std::is_same_v<T, qf::u32>)
        valid = std::strcmp(f, "I") == 0 || std::strcmp(f, "L") == 0;
    if constexpr (std::is_same_v<T, qf::i64>)
        valid = std::strcmp(f, "q") == 0 || std::strcmp(f, "l") == 0;
    if constexpr (std::is_same_v<T, std::uint8_t>)
        valid = std::strcmp(f, "B") == 0;
    if constexpr (std::is_same_v<T, double>)
        valid = std::strcmp(f, "d") == 0;
    if (!valid || b.itemsize != sizeof(T) || b.ndim != 1 || b.len % sizeof(T))
        throw std::invalid_argument("expected a contiguous 1-D native-endian buffer with the required dtype");
    std::vector<T> result(static_cast<std::size_t>(b.len / sizeof(T)));
    if (b.len)
        std::memcpy(result.data(), b.buf, static_cast<std::size_t>(b.len));
    return result;
}
// Immutable buffer owners keep the C++ storage alive even after the Python
// complex is deleted, or replaced by a copy-on-write insertion.
struct BufferOwner {
    PyObject_HEAD std::shared_ptr<const void> *owner;
    const void *data;
    Py_ssize_t length;
};
int buffer_get(PyObject *obj, Py_buffer *v, int flags) {
    auto b = reinterpret_cast<BufferOwner *>(obj);
    static const char empty = '\0';
    return PyBuffer_FillInfo(v, obj, const_cast<void *>(b->length ? b->data : &empty), b->length, 1, flags);
}
void buffer_delete(PyObject *obj) {
    auto b = reinterpret_cast<BufferOwner *>(obj);
    delete b->owner;
    PyObject_Del(obj);
}
PyBufferProcs buffer_procs = {buffer_get, nullptr};
PyTypeObject BufferType = {PyVarObject_HEAD_INIT(nullptr, 0)};
template <class T, class Owner>
PyObject *buffer(const std::vector<T> &v, const std::shared_ptr<Owner> &owner) {
    if (v.size() > static_cast<std::size_t>(PY_SSIZE_T_MAX) / sizeof(T))
        throw std::overflow_error("buffer too large");
    auto hold = std::make_unique<std::shared_ptr<const void>>(std::static_pointer_cast<const void>(owner));
    auto o = PyObject_New(BufferOwner, &BufferType);
    if (!o)
        throw PythonError{};
    o->owner = hold.release();
    o->data = v.data();
    o->length = static_cast<Py_ssize_t>(v.size() * sizeof(T));
    return reinterpret_cast<PyObject *>(o);
}
// Construct each tuple item only after the tuple owns all earlier references.
// This also handles allocation failures between two exported buffer objects.
template <class... Constructors>
PyObject *owned_tuple(Constructors &&...construct) {
    PyObject *t = PyTuple_New(sizeof...(construct));
    if (!t)
        throw PythonError{};
    Py_ssize_t position = 0;
    try {
        auto append = [&](auto &&factory) {
            PyObject *item = factory();
            if (!item)
                throw PythonError{};
            PyTuple_SET_ITEM(t, position++, item);
        };
        (append(std::forward<Constructors>(construct)), ...);
    } catch (...) {
        Py_DECREF(t);
        throw;
    }
    return t;
}
PyObject *ints(const std::vector<qf::i64> &values) {
    PyObject *out = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (!out)
        throw PythonError{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto x = PyLong_FromLongLong(values[i]);
        if (!x) {
            Py_DECREF(out);
            throw PythonError{};
        }
        PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), x);
    }
    return out;
}
PyObject *empty(PyObject *, PyObject *) {
    QF_TRY return capsule(std::make_shared<qf::Complex>(), K_NAME);
    QF_CATCH
}
PyObject *from_arrays(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *v, *o;
    Py_ssize_t limit;
    if (!PyArg_ParseTuple(args, "OOn", &v, &o, &limit))
        return nullptr;
    if (limit < 0)
        throw std::invalid_argument("negative max_simplices");
    auto vv = read<qf::u32>(v);
    auto oo = read<qf::i64>(o);
    auto k = native_run([&] { return qf::from_arrays(vv, oo, limit); });
    return capsule(k, K_NAME);
    QF_CATCH
}
PyObject *from_graph(PyObject *, PyObject *args) {
    QF_TRY
    Py_ssize_t n, limit;
    int cap;
    PyObject *e;
    if (!PyArg_ParseTuple(args, "nOin", &n, &e, &cap, &limit))
        return nullptr;
    if (n < 0 || limit < 0)
        throw std::invalid_argument("negative vertex count or max_simplices");
    auto edges = read<qf::u32>(e);
    auto k = native_run([&] { return qf::from_graph(n, edges, cap, limit); });
    return capsule(k, K_NAME);
    QF_CATCH
}
PyObject *from_points(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *p;
    Py_ssize_t n, d, limit;
    double r;
    int cap, torus;
    if (!PyArg_ParseTuple(args, "Onndipn", &p, &n, &d, &r, &cap, &torus, &limit))
        return nullptr;
    if (n < 0 || d < 0 || limit < 0)
        throw std::invalid_argument("negative size");
    auto points = read<double>(p);
    auto k = native_run([&] { return qf::from_points(points, n, d, r, cap, torus, limit); });
    return capsule(k, K_NAME);
    QF_CATCH
}
PyObject *insert(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    Py_ssize_t limit;
    if (!PyArg_ParseTuple(args, "OOn", &h, &s, &limit))
        return nullptr;
    if (limit < 0)
        throw std::invalid_argument("negative max_simplices");
    auto k = get<qf::Complex>(h, K_NAME);
    auto v = read<qf::u32>(s);
    auto p = native_run([&] { return qf::insert(k, std::move(v), limit); });
    return owned_tuple([&] { return capsule(p.first, K_NAME); }, [&] { return PyBool_FromLong(p.second); });
    QF_CATCH
}
PyObject *info(PyObject *, PyObject *h) {
    QF_TRY
    auto k = get<qf::Complex>(h, K_NAME);
    auto nv = k->count(0);
    return Py_BuildValue("KKiK", static_cast<unsigned long long>(k->size()),
                         static_cast<unsigned long long>(nv), k->dimension,
                         nv ? static_cast<unsigned long long>(k->verts[nv - 1]) + 1 : 0ULL);
    QF_CATCH
}
PyObject *arrays(PyObject *, PyObject *h) {
    QF_TRY
    auto k = get<qf::Complex>(h, K_NAME);
    return owned_tuple([&] { return buffer(k->verts, k); }, [&] { return buffer(k->off, k); },
                       [&] { return buffer(k->dims, k); });
    QF_CATCH
}
PyObject *range(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int d;
    if (!PyArg_ParseTuple(args, "Oi", &h, &d))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto r = k->range(d);
    return Py_BuildValue("KK", static_cast<unsigned long long>(r.first),
                         static_cast<unsigned long long>(r.second));
    QF_CATCH
}
PyObject *memory(PyObject *, PyObject *h) {
    QF_TRY
    auto k = get<qf::Complex>(h, K_NAME);
    return Py_BuildValue("KK", static_cast<unsigned long long>(k->payload_bytes()),
                         static_cast<unsigned long long>(k->allocated_bytes()));
    QF_CATCH
}
PyObject *mask(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m;
    if (!PyArg_ParseTuple(args, "OO", &h, &m))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto mm = read<std::uint8_t>(m);
    auto a = native_run([&] { return qf::from_mask(k, std::move(mm), true); });
    return capsule(a, A_NAME);
    QF_CATCH
}
PyObject *induced(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *v;
    if (!PyArg_ParseTuple(args, "OO", &h, &v))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto vv = read<qf::u32>(v);
    auto a = native_run([&] { return qf::induced(k, std::move(vv)); });
    return capsule(a, A_NAME);
    QF_CATCH
}
PyObject *flag(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *e, *v;
    if (!PyArg_ParseTuple(args, "OOO", &h, &e, &v))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto ee = read<qf::u32>(e);
    auto vv = read<qf::u32>(v);
    auto a = native_run([&] { return qf::flag_subcomplex(k, ee, std::move(vv)); });
    return capsule(a, A_NAME);
    QF_CATCH
}
PyObject *sub_data(PyObject *, PyObject *h) {
    QF_TRY
    auto a = get<qf::Subcomplex>(h, A_NAME);
    return owned_tuple([&] { return buffer(a->mask, a); }, [&] { return buffer(a->vcomp, a); },
                       [&] { return PyLong_FromLong(a->ncomp); });
    QF_CATCH
}
PyObject *sub_parent(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *k;
    if (!PyArg_ParseTuple(args, "OO", &h, &k))
        return nullptr;
    auto a = get<qf::Subcomplex>(h, A_NAME);
    auto parent = get<qf::Complex>(k, K_NAME);
    return PyBool_FromLong(a->parent.get() == parent.get());
    QF_CATCH
}
PyObject *regular(PyObject *, PyObject *h) {
    QF_TRY
    auto a = get<qf::Subcomplex>(h, A_NAME);
    return PyBool_FromLong(native_run([&] { return qf::component_fullness(*a); }));
    QF_CATCH
}
PyObject *correction(PyObject *, PyObject *h) {
    QF_TRY
    auto a = get<qf::Subcomplex>(h, A_NAME);
    auto p = native_run([&] { return qf::low_degree_correction(*a); });
    return Py_BuildValue("LL", static_cast<long long>(p.first), static_cast<long long>(p.second));
    QF_CATCH
}
PyObject *cone(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    Py_ssize_t limit;
    if (!PyArg_ParseTuple(args, "On", &h, &limit))
        return nullptr;
    if (limit < 0)
        throw std::invalid_argument("negative max_simplices");
    auto a = get<qf::Subcomplex>(h, A_NAME);
    auto k = native_run([&] { return qf::cone_model(*a, limit); });
    return capsule(k, K_NAME);
    QF_CATCH
}
PyObject *betti(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m;
    int quotient;
    if (!PyArg_ParseTuple(args, "OOp", &h, &m, &quotient))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    auto b = native_run([&] { return qf::betti(*k, a.get(), quotient); });
    return ints(b);
    QF_CATCH
}
PyObject *boundary(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m;
    int d, quotient;
    if (!PyArg_ParseTuple(args, "OOip", &h, &m, &d, &quotient))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    auto b =
        native_run([&] { return std::make_shared<qf::Boundary>(qf::boundary(*k, a.get(), d, quotient)); });
    return capsule(b, B_NAME);
    QF_CATCH
}
PyObject *chain(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m;
    int quotient;
    if (!PyArg_ParseTuple(args, "OOp", &h, &m, &quotient))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    auto c = native_run([&] { return qf::make_chain(*k, a.get(), quotient); });
    return capsule(c, C_NAME);
    QF_CATCH
}
PyObject *chain_info(PyObject *, PyObject *h) {
    QF_TRY
    auto c = get<qf::ChainComplex>(h, C_NAME);
    return owned_tuple([&] { return ints(c->counts); }, [&] { return PyLong_FromSize_t(c->payload_bytes()); },
                       [&] { return PyBool_FromLong(c->quotient); });
    QF_CATCH
}
PyObject *chain_betti(PyObject *, PyObject *h) {
    QF_TRY
    auto c = get<qf::ChainComplex>(h, C_NAME);
    return ints(native_run([&] { return c->betti(); }));
    QF_CATCH
}
PyObject *chain_boundary(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int d;
    if (!PyArg_ParseTuple(args, "Oi", &h, &d))
        return nullptr;
    auto c = get<qf::ChainComplex>(h, C_NAME);
    if (d < 0)
        throw std::invalid_argument("negative dimension");
    if (d >= static_cast<int>(c->boundaries.size())) {
        auto b = std::make_shared<qf::Boundary>();
        if (d > 0 && static_cast<std::size_t>(d - 1) < c->counts.size())
            b->rows = c->counts[d - 1];
        return capsule(b, B_NAME);
    }
    return capsule(BPtr(c, &c->boundaries[d]), B_NAME);
    QF_CATCH
}
PyObject *boundary_data(PyObject *, PyObject *h) {
    QF_TRY
    auto b = get<qf::Boundary>(h, B_NAME);
    return owned_tuple([&] { return buffer(b->idx, b); }, [&] { return buffer(b->off, b); },
                       [&] { return PyLong_FromLongLong(b->rows); },
                       [&] { return PyLong_FromLongLong(b->cols()); });
    QF_CATCH
}
PyObject *boundary_rank(PyObject *, PyObject *h) {
    QF_TRY
    auto b = get<qf::Boundary>(h, B_NAME);
    return PyLong_FromLongLong(native_run([&] { return qf::rank_f2(*b); }));
    QF_CATCH
}

PyObject *uints(const std::vector<qf::u32> &values) {
    std::vector<qf::i64> v(values.begin(), values.end());
    return ints(v);
}
std::filesystem::path file_path(PyObject *obj) {
    PyObject *p = PyOS_FSPath(obj);
    if (!p)
        throw PythonError{};
#ifdef _WIN32
    if (PyBytes_Check(p)) {
        PyObject *unicode = PyUnicode_DecodeFSDefaultAndSize(PyBytes_AS_STRING(p), PyBytes_GET_SIZE(p));
        Py_DECREF(p);
        p = unicode;
        if (!p)
            throw PythonError{};
    }
    Py_ssize_t n = 0;
    wchar_t *w = PyUnicode_AsWideCharString(p, &n);
    Py_DECREF(p);
    if (!w)
        throw PythonError{};
    std::wstring path(w, static_cast<std::size_t>(n));
    PyMem_Free(w);
    if (path.find(L'\0') != std::wstring::npos)
        throw std::invalid_argument("embedded NUL in path");
    return std::filesystem::path(path);
#else
    if (PyUnicode_Check(p)) {
        PyObject *bytes = PyUnicode_EncodeFSDefault(p);
        Py_DECREF(p);
        p = bytes;
        if (!p)
            throw PythonError{};
    }
    std::string path(PyBytes_AS_STRING(p), static_cast<std::size_t>(PyBytes_GET_SIZE(p)));
    Py_DECREF(p);
    if (path.find('\0') != std::string::npos)
        throw std::invalid_argument("embedded NUL in path");
    return std::filesystem::path(path);
#endif
}
PyObject *qft_new(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *a;
    int index, archive;
    if (!PyArg_ParseTuple(args, "OOpp", &h, &a, &index, &archive))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto sub = optional_sub(a, k);
    auto q = native_run([&] { return qf::make_qftree(k, sub, index, archive); });
    return capsule(q, T_NAME);
    QF_CATCH
}
PyObject *qft_info(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    return Py_BuildValue(
        "KiiKiiiKKKK", static_cast<unsigned long long>(q->size()), q->dimension(), static_cast<int>(q->ncomp),
        static_cast<unsigned long long>(q->source_size), q->source_dimension, static_cast<int>(q->word_index),
        static_cast<int>(bool(q->archive)), static_cast<unsigned long long>(q->table_bytes()),
        static_cast<unsigned long long>(q->index_payload_bytes()),
        static_cast<unsigned long long>(q->archive_bytes()), static_cast<unsigned long long>(q->trie.size()));
    QF_CATCH
}
PyObject *qft_data(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    return owned_tuple(
        [&] { return buffer(q->dims, q); }, [&] { return buffer(q->source_ids, q); },
        [&] { return buffer(q->supports, q); }, [&] { return buffer(q->support_off, q); },
        [&] { return buffer(q->facets, q); }, [&] { return buffer(q->facet_off, q); },
        [&] { return buffer(q->component_vertices, q); }, [&] { return buffer(q->component_off, q); },
        [&] { return buffer(q->vertex_labels, q); }, [&] { return buffer(q->vertex_images, q); });
    QF_CATCH
}
PyObject *qft_range(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int d;
    if (!PyArg_ParseTuple(args, "Oi", &h, &d))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto r = q->range(d);
    return Py_BuildValue("KK", static_cast<unsigned long long>(r.first),
                         static_cast<unsigned long long>(r.second));
    QF_CATCH
}
PyObject *qft_validate(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    native_run([&] {
        q->validate_records();
        return 0;
    });
    Py_RETURN_TRUE;
    QF_CATCH
}
PyObject *qft_regular(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    return PyBool_FromLong(native_run([&] { return q->regular(); }));
    QF_CATCH
}
PyObject *qft_graded(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    return PyBool_FromLong(native_run([&] { return q->strictly_graded(); }));
    QF_CATCH
}
PyObject *qft_find(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    if (!PyArg_ParseTuple(args, "OO", &h, &s))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto v = read<qf::u32>(s);
    return PyLong_FromLongLong(native_run([&] { return q->find_source(std::move(v)); }));
    QF_CATCH
}
PyObject *qft_word(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    Py_ssize_t id;
    if (!PyArg_ParseTuple(args, "On", &h, &id))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    if (id < 0)
        throw std::out_of_range("negative cell ID");
    return uints(q->word(id));
    QF_CATCH
}
PyObject *qft_lookup(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    if (!PyArg_ParseTuple(args, "OO", &h, &s))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto v = read<qf::u32>(s);
    return uints(q->lookup_word(v));
    QF_CATCH
}
PyObject *qft_face(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    Py_ssize_t id;
    if (!PyArg_ParseTuple(args, "OnO", &h, &id, &s))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    if (id < 0)
        throw std::out_of_range("negative cell ID");
    q->require_cell(id);
    auto v = read<qf::u32>(s);
    return PyLong_FromUnsignedLong(q->iterated_face(static_cast<qf::u32>(id), v));
    QF_CATCH
}
PyObject *qft_closure(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    if (!PyArg_ParseTuple(args, "OO", &h, &s))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto v = read<qf::u32>(s);
    return uints(native_run([&] { return q->closure(v); }));
    QF_CATCH
}
PyObject *qft_cofaces(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    Py_ssize_t id;
    int codim;
    if (!PyArg_ParseTuple(args, "Oni", &h, &id, &codim))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    if (id < 0)
        throw std::out_of_range("negative cell ID");
    q->require_cell(id);
    return uints(native_run([&] { return q->cofaces(static_cast<qf::u32>(id), codim); }));
    QF_CATCH
}
PyObject *qft_index(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int enabled;
    if (!PyArg_ParseTuple(args, "Op", &h, &enabled))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto r = native_run([&] {
        auto r = std::make_shared<qf::QFTree>(*q);
        r->word_index = enabled;
        r->build_indices();
        return r;
    });
    return capsule(r, T_NAME);
    QF_CATCH
}
PyObject *qft_collapse(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *s;
    int close;
    if (!PyArg_ParseTuple(args, "OOp", &h, &s, &close))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto v = read<qf::u32>(s);
    auto result = native_run(
        [&] { return std::make_shared<qf::CollapseResult>(qf::collapse_qftree(*q, std::move(v), close)); });
    return owned_tuple([&] { return capsule(result->tree, T_NAME); },
                       [&] { return buffer(result->image, result); });
    QF_CATCH
}
PyObject *qft_map_valid(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *k, *s;
    if (!PyArg_ParseTuple(args, "OOO", &h, &k, &s))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto r = get<qf::QFTree>(k, T_NAME);
    auto v = read<qf::u32>(s);
    return PyBool_FromLong(native_run([&] { return qf::qf_map_valid(*q, *r, v); }));
    QF_CATCH
}
PyObject *qft_save(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *p;
    if (!PyArg_ParseTuple(args, "OO", &h, &p))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto path = file_path(p);
    native_run([&] {
        qf::save_qftree(*q, path);
        return 0;
    });
    Py_RETURN_NONE;
    QF_CATCH
}
PyObject *qft_load(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *p;
    Py_ssize_t limit;
    unsigned long long max_bytes;
    int index;
    if (!PyArg_ParseTuple(args, "OnKi", &p, &limit, &max_bytes, &index))
        return nullptr;
    if (limit < 0 || index < -1 || index > 1)
        throw std::invalid_argument("invalid load options");
    auto path = file_path(p);
    auto q = native_run([&] { return qf::load_qftree(path, limit, max_bytes, index); });
    return capsule(q, T_NAME);
    QF_CATCH
}
PyObject *qft_archive(PyObject *, PyObject *h) {
    QF_TRY
    auto q = get<qf::QFTree>(h, T_NAME);
    if (!q->archive)
        throw std::logic_error("source archive not retained; construct with keep_source=True");
    return owned_tuple([&] { return capsule(q->archive, K_NAME); },
                       [&] { return buffer(q->archive_mask, q); });
    QF_CATCH
}
PyObject *qft_chain(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int relative;
    if (!PyArg_ParseTuple(args, "Op", &h, &relative))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto c = native_run([&] { return qf::qf_chain(*q, relative); });
    return capsule(c, C_NAME);
    QF_CATCH
}
PyObject *qft_boundary(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    int d, relative;
    if (!PyArg_ParseTuple(args, "Oip", &h, &d, &relative))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto b = native_run([&] { return std::make_shared<qf::Boundary>(qf::qf_boundary(*q, d, relative)); });
    return capsule(b, B_NAME);
    QF_CATCH
}
PyObject *qft_incidence(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h;
    Py_ssize_t id;
    int relative;
    if (!PyArg_ParseTuple(args, "Onp", &h, &id, &relative))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    if (id < 0)
        throw std::out_of_range("negative cell ID");
    q->require_cell(id);
    auto terms = native_run([&] { return qf::qf_signed_boundary(*q, static_cast<qf::u32>(id), relative); });
    std::vector<qf::i64> ids, coeff;
    for (auto p : terms) {
        ids.push_back(p.first);
        coeff.push_back(p.second);
    }
    return owned_tuple([&] { return ints(ids); }, [&] { return ints(coeff); });
    QF_CATCH
}
PyObject *qft_chain_image(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *k, *s;
    int d, relative;
    if (!PyArg_ParseTuple(args, "OOOip", &h, &k, &s, &d, &relative))
        return nullptr;
    auto q = get<qf::QFTree>(h, T_NAME);
    auto r = get<qf::QFTree>(k, T_NAME);
    auto v = read<qf::u32>(s);
    return ints(native_run([&] { return qf::qf_chain_image(*q, *r, v, d, relative); }));
    QF_CATCH
}

// --- closed-star (local) quotient presentation -------------------------------
PyObject *star_partition(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m;
    if (!PyArg_ParseTuple(args, "OO", &h, &m))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    if (!a)
        throw std::invalid_argument("star partition requires a subcomplex");
    auto part = native_run([&] { return std::make_shared<qf::StarPartition>(qf::star_partition(*k, *a)); });
    std::vector<qf::i64> counts(part->counts.begin(), part->counts.end());
    return owned_tuple([&] { return buffer(part->label, part); }, [&] { return ints(counts); });
    QF_CATCH
}
PyObject *qft_star_new(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m, *l;
    int index;
    if (!PyArg_ParseTuple(args, "OOOp", &h, &m, &l, &index))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    if (!a)
        throw std::invalid_argument("star quotient requires a subcomplex");
    qf::StarPartition part;
    part.label = read<std::uint8_t>(l);
    for (auto x : part.label) {
        if (x > 3)
            throw std::invalid_argument("invalid star label");
        ++part.counts[x];
    }
    auto q = native_run([&] { return qf::make_star_qftree(k, a, part, index); });
    return capsule(q, T_NAME);
    QF_CATCH
}
PyObject *star_chain(PyObject *, PyObject *args) {
    QF_TRY
    PyObject *h, *m, *t, *l;
    int quotient;
    if (!PyArg_ParseTuple(args, "OOOOp", &h, &m, &t, &l, &quotient))
        return nullptr;
    auto k = get<qf::Complex>(h, K_NAME);
    auto a = optional_sub(m, k);
    auto q = get<qf::QFTree>(t, T_NAME);
    if (!a)
        throw std::invalid_argument("star chain requires a subcomplex");
    qf::StarPartition part;
    part.label = read<std::uint8_t>(l);
    for (auto x : part.label) {
        if (x > 3)
            throw std::invalid_argument("invalid star label");
        ++part.counts[x];
    }
    auto c = native_run([&] { return qf::star_chain(*k, *a, *q, part, quotient); });
    return capsule(c, C_NAME);
    QF_CATCH
}

PyObject *build_info(PyObject *, PyObject *) {
#ifdef _MSC_VER
    const char *compiler = "MSVC";
#else
    const char *compiler = __VERSION__;
#endif
    return Py_BuildValue("{s:s,s:s,s:s,s:s}", "language", "C++17", "binding", "CPython C API", "compiler",
                         compiler, "python_headers", PY_VERSION);
}
PyMethodDef methods[] = {{"qft_new", qft_new, METH_VARARGS, nullptr},
                         {"qft_info", qft_info, METH_O, nullptr},
                         {"qft_data", qft_data, METH_O, nullptr},
                         {"qft_range", qft_range, METH_VARARGS, nullptr},
                         {"qft_validate", qft_validate, METH_O, nullptr},
                         {"qft_regular", qft_regular, METH_O, nullptr},
                         {"qft_graded", qft_graded, METH_O, nullptr},
                         {"qft_find", qft_find, METH_VARARGS, nullptr},
                         {"qft_word", qft_word, METH_VARARGS, nullptr},
                         {"qft_lookup", qft_lookup, METH_VARARGS, nullptr},
                         {"qft_face", qft_face, METH_VARARGS, nullptr},
                         {"qft_closure", qft_closure, METH_VARARGS, nullptr},
                         {"qft_cofaces", qft_cofaces, METH_VARARGS, nullptr},
                         {"qft_index", qft_index, METH_VARARGS, nullptr},
                         {"qft_collapse", qft_collapse, METH_VARARGS, nullptr},
                         {"qft_map_valid", qft_map_valid, METH_VARARGS, nullptr},
                         {"qft_save", qft_save, METH_VARARGS, nullptr},
                         {"qft_load", qft_load, METH_VARARGS, nullptr},
                         {"qft_archive", qft_archive, METH_O, nullptr},
                         {"qft_chain", qft_chain, METH_VARARGS, nullptr},
                         {"qft_boundary", qft_boundary, METH_VARARGS, nullptr},
                         {"qft_incidence", qft_incidence, METH_VARARGS, nullptr},
                         {"qft_chain_image", qft_chain_image, METH_VARARGS, nullptr},
                         {"star_partition", star_partition, METH_VARARGS, nullptr},
                         {"qft_star_new", qft_star_new, METH_VARARGS, nullptr},
                         {"star_chain", star_chain, METH_VARARGS, nullptr},

                         {"empty", empty, METH_NOARGS, nullptr},
                         {"from_arrays", from_arrays, METH_VARARGS, nullptr},
                         {"from_graph", from_graph, METH_VARARGS, nullptr},
                         {"from_points", from_points, METH_VARARGS, nullptr},
                         {"insert", insert, METH_VARARGS, nullptr},
                         {"info", info, METH_O, nullptr},
                         {"arrays", arrays, METH_O, nullptr},
                         {"range", range, METH_VARARGS, nullptr},
                         {"memory", memory, METH_O, nullptr},
                         {"mask", mask, METH_VARARGS, nullptr},
                         {"induced", induced, METH_VARARGS, nullptr},
                         {"flag", flag, METH_VARARGS, nullptr},
                         {"sub_data", sub_data, METH_O, nullptr},
                         {"sub_parent", sub_parent, METH_VARARGS, nullptr},
                         {"regular", regular, METH_O, nullptr},
                         {"correction", correction, METH_O, nullptr},
                         {"cone", cone, METH_VARARGS, nullptr},
                         {"betti", betti, METH_VARARGS, nullptr},
                         {"boundary", boundary, METH_VARARGS, nullptr},
                         {"chain", chain, METH_VARARGS, nullptr},
                         {"chain_info", chain_info, METH_O, nullptr},
                         {"chain_betti", chain_betti, METH_O, nullptr},
                         {"chain_boundary", chain_boundary, METH_VARARGS, nullptr},
                         {"boundary_data", boundary_data, METH_O, nullptr},
                         {"boundary_rank", boundary_rank, METH_O, nullptr},
                         {"build_info", build_info, METH_NOARGS, nullptr},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native",
                      "Native source-labelled QF trees, binary I/O, and separate F2 consumers.", -1, methods};
} // anonymous namespace
PyMODINIT_FUNC PyInit__native(void) {
    BufferType.tp_name = "qfcore._native.ReadOnlyBuffer";
    BufferType.tp_basicsize = sizeof(BufferOwner);
    BufferType.tp_dealloc = buffer_delete;
    BufferType.tp_flags = Py_TPFLAGS_DEFAULT;
    BufferType.tp_as_buffer = &buffer_procs;
    if (PyType_Ready(&BufferType) < 0)
        return nullptr;
    return PyModule_Create(&module);
}

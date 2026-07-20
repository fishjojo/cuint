#!/usr/bin/env python
#  Copyright 2026 The CUINT Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ctypes
import os

import numpy as np
import cupy as cp

from pyscf import gto
from pyscf.gto.moleintor import make_loc

from pyscf.gto import NPRIM_OF, NCTR_OF, ANG_OF, PTR_EXP, PTR_COEFF

# caps of the fused kernels; these must match the values the library was
# compiled with (G3C2E_MAX_L_PAIR / G3C2E_MAX_L_AUX / G3C2E_MAX_L_TOTAL in
# the CMake cache, see build/config.h)
MAX_L_PAIR = 3
MAX_L_AUX = 6
MAX_L_TOTAL = 6

# caps of the two-stage path (G3C2E_TS_MAX_L_PAIR / G3C2E_TS_MAX_L_AUX)
TS_MAX_L_PAIR = 4
TS_MAX_L_AUX = 6


def is_fused_class(i_angular, j_angular, k_angular):
    """Whether a class runs the fused single-kernel path (the same rule the
    C dispatch applies with the caps set by get_int3c2e)."""
    return (i_angular <= MAX_L_PAIR and j_angular <= MAX_L_PAIR
            and k_angular <= MAX_L_AUX
            and i_angular + j_angular + k_angular <= MAX_L_TOTAL)

# libcuint links cuBLAS by soname. The process must hold exactly ONE
# matched libcublas.so.13 + libcublasLt.so.13 pair: CuPy's ecosystem may
# already have mapped a pip-wheel cuBLASLt copy, and dlopen-ing the toolkit
# pair by absolute path alongside it puts two different Lt copies in the
# process; cublas then mixes internal tables across versions and crashes
# inside cuBLASLt. So if an Lt copy is already mapped, complete the pair
# from its own directory; otherwise preload the toolkit pair.
_cublas_dirs = []
try:
    with open("/proc/self/maps") as _maps:
        for _line in _maps:
            _path = _line.split()[-1]
            if _path.endswith("libcublasLt.so.13"):
                _cublas_dirs.append(os.path.dirname(_path))
                break
except OSError:
    pass
_cuda_home = (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
              or "/usr/local/cuda")
_cublas_dirs.append(os.path.join(_cuda_home, "lib64"))
for _dir in _cublas_dirs:
    try:
        for _lib in ("libcublasLt.so.13", "libcublas.so.13"):
            ctypes.CDLL(os.path.join(_dir, _lib), mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

libcuint = ctypes.CDLL("../lib/libcuint.so")

# pool of streams used to overlap the independent (pair class, aux group)
# launches; the high angular momentum classes have too few blocks to fill
# the GPU on their own
_N_STREAMS = 8
_stream_pool = None


def _get_stream_pool():
    global _stream_pool
    if _stream_pool is None:
        _stream_pool = [
            cp.cuda.Stream(non_blocking=True) for _ in range(_N_STREAMS)
        ]
    return _stream_pool


class Int3c2eTsPlan(ctypes.Structure):
    """ctypes mirror of Int3c2eTsPlan in cuint.h: plan-time two-stage
    metadata handed to the unified int3c2e entry (NULL for fused classes).
    pair_meta is shared by all aux groups of one pair class; aux_meta by
    all pair classes of one aux group; the aux primitive arrays are the
    ordinary aux_indices / aux_to_function plan arrays."""
    _fields_ = [
        ("pair_meta", ctypes.c_void_p),
        ("aux_meta", ctypes.c_void_p),
        ("run_first", ctypes.c_void_p),
        ("run_count", ctypes.c_void_p),
        ("run_k_ab", ctypes.c_void_p),
        ("run_offset", ctypes.c_void_p),
        ("n_prim_pairs", ctypes.c_int),
        ("n_contracted_pairs", ctypes.c_int),
        ("n_aux_shells", ctypes.c_int),
        ("n_runs", ctypes.c_int),
        ("chunk_shells", ctypes.c_int),
        ("build_h", ctypes.c_int),
    ]


def cast_to_pointer(array):
    if array is None:
        return None  # NULL
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError("Invalid array type")


def _decontract_and_sort(bas, max_l):
    """Expand a shell table into per-primitive shells sorted by angular
    momentum. Returns the primitive shell table, the map from primitive to
    the first function of its contracted shell, the number of functions, and
    the [start, end) primitive range of each angular momentum."""
    bas = np.ascontiguousarray(bas, dtype=np.int32)
    ao_loc = make_loc(bas, "sph")
    n_functions = int(ao_loc[-1])

    ls = bas[:, ANG_OF]
    if ls.max() > max_l:
        raise ValueError(
            f"angular momentum {ls.max()} exceeds the compiled limit {max_l}"
        )

    sort_idx = np.argsort(ls, kind="stable")
    sorted_bas = bas[sort_idx]
    sorted_shl_start = ao_loc[:-1][sort_idx]

    nctr = sorted_bas[:, NCTR_OF]
    nprim = sorted_bas[:, NPRIM_OF]
    decontracted = np.repeat(sorted_bas, nctr, axis=0)
    decontracted[:, NCTR_OF] = 1

    _tmp = np.arange(np.sum(nctr)) - np.repeat(np.cumsum(np.r_[0, nctr[:-1]]), nctr)
    decontracted[:, PTR_COEFF] += _tmp * np.repeat(nprim, nctr)

    shl_start = np.repeat(sorted_shl_start, nctr)
    shl_start += _tmp * np.repeat(2 * sorted_bas[:, ANG_OF] + 1, nctr)

    nprim = np.repeat(nprim, nctr)
    decontracted = np.repeat(decontracted, nprim, axis=0)

    primitive_offset = np.arange(np.sum(nprim)) - np.repeat(
        np.cumsum(np.r_[0, nprim[:-1]]), nprim
    )
    decontracted[:, NPRIM_OF] = 1
    decontracted[:, PTR_COEFF] += primitive_offset
    decontracted[:, PTR_EXP] += primitive_offset

    primitive_to_function = np.repeat(shl_start, nprim)

    angulars = decontracted[:, ANG_OF]
    ranges = np.zeros((angulars.max() + 1, 2), dtype=np.int32)
    for l in range(angulars.max() + 1):
        members = np.flatnonzero(angulars == l)
        if members.size:
            ranges[l] = [members[0], members[-1] + 1]

    return decontracted, primitive_to_function, n_functions, ranges


def _contracted_shells(bas, primitive_to_function):
    """Group the decontracted, angular-momentum-sorted primitives back into
    contracted shells. Returns, per angular momentum, a list of
    (first_function, prim_begin, prim_end); primitives of one contracted
    shell are consecutive and share their first function."""
    angulars = bas[:, ANG_OF]
    shells = [[] for _ in range(angulars.max() + 1)]
    begin = 0
    for p in range(1, len(primitive_to_function) + 1):
        if (p == len(primitive_to_function)
                or primitive_to_function[p] != primitive_to_function[begin]):
            shells[angulars[begin]].append(
                (int(primitive_to_function[begin]), begin, p)
            )
            begin = p
    return shells


# aux chunking in the C driver keeps the per-class workspace under this
# budget (unless even a single aux shell needs more)
TS_WORKSPACE_BUDGET_BYTES = 1 << 30


def _two_stage_workspace_doubles(li, lj, lk, shells_i, shells_j, shells_k):
    """Workspace need of one two-stage class, mirroring the [H][X chunk]
    [V chunk] partition of int3c2e_two_stage_planned: the H matrix is
    persistent, X and V are sized per aux-shell chunk. Diagonal (I == J)
    contracted pairs carry the full primitive square."""
    n_herm = (li + lj + 1) * (li + lj + 2) * (li + lj + 3) // 6 # pair hermite function size
    m_rows = (2 * li + 1) * (2 * lj + 1) # contracted pair size
    k_i = [e - b for _, b, e in shells_i] # per shell privimitve size
    if li == lj:
        total = sum(k_i)
        n_prim_pairs = (total * total + sum(k * k for k in k_i)) // 2
        n_pairs = len(k_i) * (len(k_i) + 1) // 2
    else:
        k_j = [e - b for _, b, e in shells_j]
        n_prim_pairs = sum(k_i) * sum(k_j)
        n_pairs = len(k_i) * len(k_j) # number of shell pairs
    n_aux_shells = len(shells_k)
    ktot_total = n_herm * n_prim_pairs
    h_doubles = m_rows * ktot_total
    per_shell = (ktot_total + n_pairs * m_rows) * (2 * lk + 1)
    ideal = h_doubles + per_shell * n_aux_shells
    budget = TS_WORKSPACE_BUDGET_BYTES // 8
    return min(ideal, max(h_doubles + per_shell, budget))


def _build_ts_pair_meta(li, lj, shells_i, shells_j):
    """Plan-time pair-class metadata for the two-stage path, built once per
    (li, lj) and shared by every aux group: the device int32 blob
    pair_prim_i | pair_prim_j | kprefix | ktot | klocal | pair_i_function
    | pair_j_function, and the host GEMM run descriptors.
    """
    n_herm = (li + lj + 1) * (li + lj + 2) * (li + lj + 3) // 6

    # contracted shell pairs with I <= J (diagonal pairs carry the full
    # primitive square), sorted by contraction degree so the C side sees
    # one contiguous run per degree
    if li == lj:
        shell_pairs = [
            (shells_i[a], shells_j[b])
            for a in range(len(shells_i))
            for b in range(a, len(shells_j))
        ]
    else:
        shell_pairs = [(sa, sb) for sa in shells_i for sb in shells_j]
    shell_pairs.sort(
        key=lambda p: (p[0][2] - p[0][1]) * (p[1][2] - p[1][1])
    )

    pair_i, pair_j, kprefix, ktot, klocal = [], [], [], [], []
    offsets = [0]
    i_func, j_func = [], []
    for (fa, ba, ea), (fb, bb, eb) in shell_pairs:
        k_ab = (ea - ba) * (eb - bb)
        begin = offsets[-1]
        pair_i.append(np.repeat(np.arange(ba, ea), eb - bb))
        pair_j.append(np.tile(np.arange(bb, eb), ea - ba))
        kprefix.append(np.full(k_ab, n_herm * begin))
        ktot.append(np.full(k_ab, n_herm * k_ab))
        klocal.append(np.arange(k_ab))
        offsets.append(begin + k_ab)
        i_func.append(fa)
        j_func.append(fb)
    n_pairs = len(shell_pairs)

    # contiguous equal-K_ab runs of the sorted pair list, one strided
    # batched GEMM each
    run_first, run_count, run_k_ab, run_offset = [], [], [], []
    for pair in range(n_pairs):
        k_ab = offsets[pair + 1] - offsets[pair]
        if run_k_ab and run_k_ab[-1] == k_ab:
            run_count[-1] += 1
        else:
            run_first.append(pair)
            run_count.append(1)
            run_k_ab.append(k_ab)
            run_offset.append(offsets[pair])

    meta_dev = cp.asarray(np.concatenate([
        np.concatenate(pair_i), np.concatenate(pair_j),
        np.concatenate(kprefix), np.concatenate(ktot),
        np.concatenate(klocal),
        np.asarray(i_func), np.asarray(j_func),
    ]).astype(np.int32))
    runs = tuple(
        np.asarray(a, dtype=np.int32)
        for a in (run_first, run_count, run_k_ab, run_offset)
    )
    struct = Int3c2eTsPlan(
        pair_meta=meta_dev.data.ptr,
        aux_meta=0,  # patched per aux group in get_int3c2e
        run_first=runs[0].ctypes.data,
        run_count=runs[1].ctypes.data,
        run_k_ab=runs[2].ctypes.data,
        run_offset=runs[3].ctypes.data,
        n_prim_pairs=offsets[-1],
        n_contracted_pairs=n_pairs,
        n_aux_shells=0,  # patched per aux group
        n_runs=len(run_first),
        chunk_shells=0, # patched per aux group
        build_h=1,
    )
    # the struct holds raw pointers; keep the backing arrays alive with it
    return {"struct": struct, "meta": meta_dev, "runs": runs}


def _build_ts_aux_meta(shells_k, k_range):
    """Plan-time aux-group metadata for the two-stage path, built once per
    k_angular: the device int32 blob aux_prim_offsets | aux_function
    (group-local shell CSR). The aux primitive arrays themselves are the
    plan's aux_indices / aux_to_function, shared with the fused path."""
    r0, r1 = int(k_range[0]), int(k_range[1])
    blob = np.concatenate([
        np.asarray([b - r0 for _, b, _ in shells_k] + [r1 - r0]),
        np.asarray([f for f, _, _ in shells_k]),
    ]).astype(np.int32)
    return {"meta": cp.asarray(blob), "n_aux_shells": len(shells_k)}


def create_int3c2e_plan(mol, auxmol):
    atm, bas, env = gto.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )

    ao_bas, ao_ptf, n_functions, ao_ranges = _decontract_and_sort(
        bas[:mol.nbas], TS_MAX_L_PAIR
    )

    aux_bas, aux_ptf, n_aux, aux_ranges = _decontract_and_sort(
        bas[mol.nbas:], TS_MAX_L_AUX
    )

    n_primitives = ao_bas.shape[0]
    combined_bas = np.vstack([ao_bas, aux_bas])

    # every pair class and aux group; libcuint.int3c2e routes each
    # (i_angular, j_angular, k_angular) class to the fused or two-stage path
    pairs = []
    for i_angular in range(ao_ranges.shape[0]):
        i_range = ao_ranges[i_angular]
        for j_angular in range(i_angular, ao_ranges.shape[0]):
            j_range = ao_ranges[j_angular]
            n_rows = i_range[1] - i_range[0]
            n_cols = j_range[1] - j_range[0]
            if i_angular == j_angular:
                n_pairs = (n_rows + 1) * n_rows // 2
            else:
                n_pairs = n_rows * n_cols
            if n_pairs == 0:
                continue
            pair_indices = cp.array([*i_range, *j_range], dtype=cp.int32)
            pairs.append((i_angular, j_angular, pair_indices, int(n_pairs)))

    aux_groups = []
    for k_angular in range(aux_ranges.shape[0]):
        k_range = aux_ranges[k_angular]
        n_aux_primitives = int(k_range[1] - k_range[0])
        if n_aux_primitives == 0:
            continue
        aux_indices = cp.arange(*k_range, dtype=cp.int32) + n_primitives
        aux_to_function = cp.asarray(
            aux_ptf[k_range[0] : k_range[1]], dtype=cp.int32
        )
        aux_groups.append(
            (k_angular, aux_indices, aux_to_function, n_aux_primitives)
        )

    # workspace for the classes above the fused caps (two-stage path)
    ao_shells = _contracted_shells(ao_bas, ao_ptf)
    aux_shells = _contracted_shells(aux_bas, aux_ptf)
    ts_workspace_doubles = max(
        (
            _two_stage_workspace_doubles(
                i_angular, j_angular, k_angular,
                ao_shells[i_angular], ao_shells[j_angular],
                aux_shells[k_angular],
            )
            for i_angular, j_angular, _, _ in pairs
            for k_angular, _, _, _ in aux_groups
            if not is_fused_class(i_angular, j_angular, k_angular)
        ),
        default=0,
    )

    # plan-time metadata for the two-stage classes (the fused/two-stage
    # routing is frozen here): one pair blob per pair class, one aux blob
    # per aux group, shared across all classes that use them, plus the
    # per-stream workspace slots that let two-stage classes run
    # concurrently
    ts_classes = set()
    ts_pair_meta = {}
    ts_aux_meta = {}
    for i_angular, j_angular, _, _ in pairs:
        for k_angular, _, _, _ in aux_groups:
            if is_fused_class(i_angular, j_angular, k_angular):
                continue
            ts_classes.add((i_angular, j_angular, k_angular))
            if (i_angular, j_angular) not in ts_pair_meta:
                ts_pair_meta[(i_angular, j_angular)] = _build_ts_pair_meta(
                    i_angular, j_angular,
                    ao_shells[i_angular], ao_shells[j_angular],
                )
            if k_angular not in ts_aux_meta:
                ts_aux_meta[k_angular] = _build_ts_aux_meta(
                    aux_shells[k_angular], aux_ranges[k_angular]
                )

    # aux chunk size per class against the workspace slot, matching the
    # [H][X chunk][V chunk] partition the C side uses
    ts_chunk_shells = {}
    for i_angular, j_angular, k_angular in ts_classes:
        pair_struct = ts_pair_meta[(i_angular, j_angular)]["struct"]
        lab = i_angular + j_angular
        n_herm = (lab + 1) * (lab + 2) * (lab + 3) // 6
        m_rows = (2 * i_angular + 1) * (2 * j_angular + 1)
        ktot_total = n_herm * pair_struct.n_prim_pairs
        h_doubles = m_rows * ktot_total
        per_shell = (ktot_total + pair_struct.n_contracted_pairs * m_rows
                     ) * (2 * k_angular + 1)
        chunk = min(ts_aux_meta[k_angular]["n_aux_shells"],
                    (ts_workspace_doubles - h_doubles) // per_shell)
        assert chunk >= 1, "workspace slot below one aux shell"
        ts_chunk_shells[(i_angular, j_angular, k_angular)] = int(chunk)

    ts_workspaces = []
    if ts_classes:
        free_bytes = cp.cuda.Device().mem_info[0]
        result_bytes = n_functions * n_functions * n_aux * 8
        slot_bytes = max(ts_workspace_doubles * 8, 1)
        n_slots = int(max(
            1,
            min(4, (free_bytes - result_bytes - (512 << 20)) // slot_bytes),
        ))
        ts_workspaces = [
            cp.empty(ts_workspace_doubles, dtype=cp.double)
            for _ in range(n_slots)
        ]

    plan = {
        "atms": cp.asarray(atm, dtype=cp.int32),
        "bases": cp.asarray(combined_bas, dtype=cp.int32),
        "envs": cp.asarray(env, dtype=cp.double),
        "shell_to_ao": cp.asarray(ao_ptf, dtype=cp.int32),
        "n_configurations": 1,
        "n_functions": n_functions,
        "n_aux": n_aux,
        "n_primitives": n_primitives,
        "pairs": pairs,
        "aux_groups": aux_groups,
        "ts_workspace_doubles": ts_workspace_doubles,
        "ts_classes": ts_classes,
        "ts_pair_meta": ts_pair_meta,
        "ts_aux_meta": ts_aux_meta,
        "ts_chunk_shells": ts_chunk_shells,
        "ts_workspaces": ts_workspaces,
        "is_screened": 0,
    }

    return plan


def get_int3c2e(plan, out=None):
    """Compute the full (config, i, j, aux) tensor for the plan. Pass the
    previous result as `out` when recomputing (e.g. along a trajectory).
    """
    shape = (plan["n_configurations"], plan["n_functions"],
             plan["n_functions"], plan["n_aux"])
    if out is None:
        result = cp.zeros(shape)
    else:
        assert out.shape == shape and out.dtype == cp.double
        result = out
        result.fill(0.0)

    # the module-level caps (initialized from the compiled ones) drive the
    # C-side routing of the fused entry; the two-stage routing was frozen
    # into plan["ts_classes"] at plan time
    libcuint.int3c2e_set_fused_caps(
        ctypes.c_int(MAX_L_PAIR), ctypes.c_int(MAX_L_AUX),
        ctypes.c_int(MAX_L_TOTAL)
    )

    ts_classes = plan.get("ts_classes", set())
    slots = plan.get("ts_workspaces", [])
    n_slots = len(slots)

    # class launches fan out over the stream pool: each pair class with
    # two-stage work is pinned to one slot stream (its aux groups then run
    # in order, so the H matrix is built once and reused via build_h),
    # while fused classes round-robin over the remaining streams. Classes
    # write disjoint result tiles, so cross-stream concurrency is safe.
    streams = _get_stream_pool()
    fused_streams = streams[n_slots:] or streams
    current = cp.cuda.get_current_stream()
    start = cp.cuda.Event()
    start.record()
    for stream in streams:
        stream.wait_event(start)

    n_fused = 0
    slot_index = 0
    for i_angular, j_angular, pair_indices, n_pairs in plan["pairs"]:
        has_ts = any(
            (i_angular, j_angular, k) in ts_classes
            for k, _, _, _ in plan["aux_groups"]
        )
        if has_ts:
            slot = slot_index % n_slots
            slot_index += 1
            ts_stream = streams[slot]
            ts_struct = plan["ts_pair_meta"][(i_angular, j_angular)]["struct"]
            first_ts = True
        for k_angular, aux_indices, aux_to_function, n_aux_prims in plan["aux_groups"]:
            if (i_angular, j_angular, k_angular) in ts_classes:
                stream = ts_stream
                aux_meta = plan["ts_aux_meta"][k_angular]
                ts_struct.aux_meta = aux_meta["meta"].data.ptr
                ts_struct.n_aux_shells = aux_meta["n_aux_shells"]
                ts_struct.chunk_shells = plan["ts_chunk_shells"][
                    (i_angular, j_angular, k_angular)
                ]
                # the Phase A H matrix depends only on the pair class:
                # build it on the first aux group, reuse afterwards
                ts_struct.build_h = 1 if first_ts else 0
                first_ts = False
                ts_ref = ctypes.byref(ts_struct)
                workspace = slots[slot]
                workspace_ptr = cast_to_pointer(workspace)
                workspace_bytes = workspace.nbytes
            else:
                stream = fused_streams[n_fused % len(fused_streams)]
                n_fused += 1
                ts_ref = None
                workspace_ptr = None
                workspace_bytes = 0
            status = libcuint.int3c2e(
                ctypes.c_void_p(stream.ptr),
                cast_to_pointer(result),
                cast_to_pointer(pair_indices),
                ctypes.c_int(n_pairs),
                ctypes.c_int(plan["n_primitives"]),
                cast_to_pointer(plan["shell_to_ao"]),
                ctypes.c_int(plan["n_functions"]),
                cast_to_pointer(aux_indices),
                ctypes.c_int(n_aux_prims),
                cast_to_pointer(aux_to_function),
                ctypes.c_int(plan["n_aux"]),
                cast_to_pointer(plan["atms"]),
                ctypes.c_int(plan["atms"].size),
                cast_to_pointer(plan["bases"]),
                ctypes.c_int(plan["bases"].size),
                cast_to_pointer(plan["envs"]),
                ctypes.c_int(plan["envs"].size),
                ctypes.c_int(plan["n_configurations"]),
                ctypes.c_int(i_angular),
                ctypes.c_int(j_angular),
                ctypes.c_int(k_angular),
                ctypes.c_int(plan["is_screened"]),
                ts_ref,
                workspace_ptr,
                ctypes.c_size_t(workspace_bytes),
            )
            if status != 0:
                raise RuntimeError(
                    f"int3c2e failed with status {status} for class "
                    f"({i_angular},{j_angular}|{k_angular})"
                )

    for stream in streams:
        current.wait_event(stream.record())

    # in-place (i, j) <-> (j, i) symmetrization; the kernels fill only the
    # li <= lj orientation (with halved diagonal tiles)
    libcuint.int3c2e_symmetrize(
        ctypes.c_void_p(current.ptr),
        cast_to_pointer(result),
        ctypes.c_int(plan["n_configurations"]),
        ctypes.c_int(plan["n_functions"]),
        ctypes.c_int(plan["n_aux"]),
    )
    return result

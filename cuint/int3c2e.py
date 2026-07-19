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
import numpy as np
import cupy as cp

from pyscf import gto
from pyscf.gto.moleintor import make_loc

from pyscf.gto import NPRIM_OF, NCTR_OF, ANG_OF, PTR_EXP, PTR_COEFF

# the fused kernels are instantiated up to (dd|f); see G3C2E_MAX_L_PAIR and
# G3C2E_MAX_L_AUX in CMakeLists.txt
MAX_L_PAIR = 2
MAX_L_AUX = 4

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


def cast_to_pointer(array):
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


def create_int3c2e_plan(mol, auxmol):
    atm, bas, env = gto.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    n_mol_shells = mol._bas.shape[0]

    ao_bas, ao_ptf, n_functions, ao_ranges = _decontract_and_sort(
        bas[:n_mol_shells], MAX_L_PAIR
    )
    aux_bas, aux_ptf, n_aux, aux_ranges = _decontract_and_sort(
        bas[n_mol_shells:], MAX_L_AUX
    )

    n_primitives = ao_bas.shape[0]
    combined_bas = np.vstack([ao_bas, aux_bas])

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
        "is_screened": 0,
    }

    return plan


def get_int3c2e(plan):
    result = cp.zeros(
        (plan["n_configurations"], plan["n_functions"], plan["n_functions"],
         plan["n_aux"])
    )

    # classes only accumulate into result with atomicAdd, so their kernels
    # can run concurrently; events keep the pool ordered after the zeros
    # above and before the transpose below without blocking the host
    #streams = _get_stream_pool()
    #current_stream = cp.cuda.get_current_stream()
    #result_ready = current_stream.record()
    #for stream in streams:
    #    stream.wait_event(result_ready)

    #launch_index = 0
    for i_angular, j_angular, pair_indices, n_pairs in plan["pairs"]:
        for k_angular, aux_indices, aux_to_function, n_aux_prims in plan[
            "aux_groups"
        ]:
            #stream = streams[launch_index % len(streams)]
            #launch_index += 1
            libcuint.int3c2e(
                #ctypes.c_void_p(stream.ptr),
                0,
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
            )

    #for stream in streams:
    #    current_stream.wait_event(stream.record())

    return result + result.transpose(0, 2, 1, 3)

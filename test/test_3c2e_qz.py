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

"""Realistic high angular momentum test: 3 waters, cc-pVQZ AO (g shells)
with cc-pVQZ-jkfit aux, mixing fused and two-stage classes."""

import time

import numpy as np
import cupy as cp
import pyscf
from pyscf import df

from cuint.int3c2e import create_int3c2e_plan, get_int3c2e, is_fused_class


def main():
    mol = pyscf.M(
        #atom="""
        # O  0.0000  0.0000  0.1173
        # H  0.0000  0.7572 -0.4692
        # H  0.0000 -0.7572 -0.4692
        # O  3.0000  0.0000  0.1173
        # H  3.0000  0.7572 -0.4692
        # H  3.0000 -0.7572 -0.4692
        # O  0.0000  3.0000  0.1173
        # H  0.0000  3.7572 -0.4692
        # H  0.0000  2.2428 -0.4692
        #""",
        atom="4185_water5CYC.xyz",
        #atom="4216_water10PP2.xyz",
        basis="ccpvqz",
        verbose=0,
    )
    auxmol = df.addons.make_auxmol(mol, "ccpvqzjkfit")
    print(f"nao: {mol.nao}  naux: {auxmol.nao}  "
          f"AO l max: {mol._bas[:, 1].max()}  "
          f"aux l max: {auxmol._bas[:, 1].max()}")

    plan = create_int3c2e_plan(mol, auxmol)
    n_two_stage = sum(
        1
        for pi, pj, _, _ in plan["pairs"]
        for k, _, _, _ in plan["aux_groups"]
        if not is_fused_class(pi, pj, k)
    )
    n_classes = len(plan["pairs"]) * len(plan["aux_groups"])
    print(f"classes: {n_classes}  two-stage: {n_two_stage}")
    print(f"result: {mol.nao**2 * auxmol.nao * 8 / 1e9:.2f} GB  workspace: "
          f"{plan['ts_workspace_doubles'] * 8 / 1e9:.2f} GB")

    result = get_int3c2e(plan)
    cp.cuda.Device().synchronize()
    # release the warm-up tensor so the timed call reuses its pooled block
    # instead of paying a fresh multi-GB cudaMalloc inside the timing
    result = None

    start = time.perf_counter()
    result = get_int3c2e(plan)
    cp.cuda.Device().synchronize()
    print(f"GPU time (warm): {time.perf_counter() - start:.2f} s")

    start = time.perf_counter()
    reference = df.incore.aux_e2(mol, auxmol, intor="int3c2e_sph")
    print(f"PySCF time: {time.perf_counter() - start:.2f} s")

    #error = np.abs(result.get()[0] - reference).max()
    #scale = np.abs(reference).max()
    #print(f"max abs error: {error:.3e} (ref scale {scale:.3e})")
    #assert error < 1e-10
    #print("cc-pVQZ 3-water test passed")


if __name__ == "__main__":
    main()

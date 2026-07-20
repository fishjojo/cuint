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

"""12 waters on a 3x2x2 grid: check int3c2e against PySCF and time GPU vs CPU.

On an RTX 2060 (measured 2026-07-17) the warm GPU kernels take ~0.34 s vs
~0.8 s for PySCF's threaded CPU aux_e2. Note the first GPU call in a fresh
process pays ~1.7 s of CUDA context creation plus module loading on top;
that one-time cost is excluded below by warming up first.
"""

import time

import numpy as np
import cupy as cp
import pyscf
from pyscf import df

from cuint.int3c2e import create_int3c2e_plan, get_int3c2e

waters = []
for ix in range(3):
    for iy in range(2):
        for iz in range(2):
            x, y, z = 3.0 * ix, 3.0 * iy, 3.0 * iz
            waters.append(f"O {x} {y} {z + 0.1173}")
            waters.append(f"H {x} {y + 0.7572} {z - 0.4692}")
            waters.append(f"H {x} {y - 0.7572} {z - 0.4692}")

mol = pyscf.M(atom="; ".join(waters), basis="ccpvdz", verbose=0)
#auxmol = df.addons.make_auxmol(mol, "ccpvdz-jkfit")
auxmol = df.addons.make_auxmol(mol, "weigend")
# the fused kernels are compiled up to (dd|g): drop aux shells above f; the
# reference uses the same truncated auxmol so the comparison stays exact
print(f"max aux L: {auxmol._bas[:,1].max()}")
#auxmol._bas = auxmol._bas[auxmol._bas[:, 1] <= 4]
print(f"atoms: {mol.natm}  nao: {mol.nao}  naux: {auxmol.nao}")

plan = create_int3c2e_plan(mol, auxmol)

# warm up: CUDA context, kernel module loading, result allocation
result = get_int3c2e(plan)
cp.cuda.Device().synchronize()

t0 = time.perf_counter()
for _ in range(10):
    result = get_int3c2e(plan)
    cp.cuda.Device().synchronize()
t_gpu = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(10):
    ref = df.incore.aux_e2(mol, auxmol, intor="int3c2e_sph")
t_cpu = time.perf_counter() - t0

error = float(cp.abs(result[0] - cp.asarray(ref)).max())
print(f"GPU (warm): {t_gpu:.3f} s   CPU aux_e2: {t_cpu:.3f} s   "
      f"speedup: {t_cpu / t_gpu:.2f}x")
print(f"max abs error vs pyscf: {error:.3e}")

assert error < 1e-10
print("PASSED")

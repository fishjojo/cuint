#!/usr/bin/env python
# Copyright 2026 The CUINT Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import cupy as cp
import pyscf
from cuint.overlap import *

atoms = """
 O  0.74030004  3.22499019 -2.63607016
 H -0.70186004  3.73419022 -1.10377007
 H  0.52820003  2.24309013  0.63029004
"""

# cc-pVQZ covers angular momenta up to g functions
mol = pyscf.M(atom=atoms, basis="ccpvqz", verbose=0)
print(mol.nao)

plan = create_ovlp_plan_new(
    np.array([mol._atm]),
    np.array([mol._bas]),
    np.array([mol._env]),
    screening=False,
)

# consistency with the dedicated kernels
exp = get_gen_ovlp(plan, 0, 0)[0, 0].get()
ref = get_ovlp(plan)[0].get()
assert np.linalg.norm(exp - ref) < 1e-12

exp = get_gen_ovlp(plan, 1, 0)[0].get()
ref = get_ovlp_gradient(plan)[0].get()
assert np.linalg.norm(exp - ref) < 1e-12

# references from libcint
exp = get_gen_ovlp(plan, 0, 0)[0, 0].get()
ref = mol.intor("int1e_ovlp")
assert np.linalg.norm(exp - ref) < 1e-9

# (nabla i | j)
exp = get_gen_ovlp(plan, 1, 0)[0].get()
ref = mol.intor("int1e_ipovlp")
assert np.linalg.norm(exp - ref) < 1e-9

# (i | nabla j) = -(nabla i | j)
exp = get_gen_ovlp(plan, 0, 1)[0].get()
ref = -mol.intor("int1e_ipovlp")
assert np.linalg.norm(exp - ref) < 1e-9

# (nabla i | nabla j), components xixj, xiyj, ..., zizj
exp = get_gen_ovlp(plan, 1, 1)[0].get()
ref = mol.intor("int1e_ipovlpip")
assert np.linalg.norm(exp - ref) < 1e-9

# (nabla nabla i | j), components xixi, xiyi, ..., zizi
exp = get_gen_ovlp(plan, 2, 0)[0].get()
ref = mol.intor("int1e_ipipovlp")
assert np.linalg.norm(exp - ref) < 1e-9

# (i | nabla nabla j)
exp = get_gen_ovlp(plan, 0, 2)[0].get()
ref = mol.intor("int1e_ipipovlp").transpose(0, 2, 1)
assert np.linalg.norm(exp - ref) < 1e-9

print("all gen_ovlp tests passed")

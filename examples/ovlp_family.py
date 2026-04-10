#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf.lib import logger
from cuint.overlap import *

atoms = """
 H -5.8768512   5.12198377 -1.26494537 
 O  0.74030004  3.22499019 -2.63607016 
 O -0.70186004  3.73419022 -1.10377007 
 O  0.52820003  2.24309013  0.63029004 
 O -0.65872004 -1.03644006  0.63759004 
 O  5.1026003   0.25111001  1.07359006 
 O  4.89220029 -0.60234004 -0.96180006 
 H -5.21283031  3.49429021 -1.6806301  
 H -6.65250039  3.67819022 -0.69658004 
 H -4.60719027  2.84869017  0.37581002 
 H -5.33926031  4.10679024  1.28509008 
 H -3.52792021  5.47159032  0.84199005 
 H -3.13841019  4.55309027 -0.52554003 
 H -1.88338011  3.29109019  0.36715002 
 H -3.90790023  4.32899026  3.05729018 
 H -2.51366015  4.47649026  4.02269024 
 H -0.60170004  3.71609022  3.53579021 
 H -0.23990001  2.99799018  1.98529012 
 H  2.99270018  3.06959018  0.63479004 
 H  3.07640018  3.66869022 -1.07466006 
 H  0.32460002  0.77903005  1.98499012 
 H  2.83170017  0.95740006  1.85019011 
 H  2.57340015 -0.81599005 -2.01725012 
 H  0.25210001 -0.87582005 -1.99946012 
 H -0.80800005  0.62354004 -0.34406002 
 H -1.18747007 -0.66697004  1.35629008 
 C -5.69024034  4.14939024 -0.85985005 
 C -4.87813029  3.91519023  0.36217002 
 C -3.50766021  4.50929027  0.42271002 
 C -2.09375012  3.78579022  2.23039013 
 C  2.55770015  3.08179018 -0.30109002 
 C  1.26620007  2.80479017 -0.38768002 
 C  0.35050002  3.29749019 -1.42015008 
 C  0.79280005  0.87059005  0.99709006 
 C  2.22200013  0.54087003  1.04769006 
 C  2.87230017 -0.02237     0.02981    
 C  2.08770012 -0.47549003 -1.14679007 
 C  0.75890004 -0.52751003 -1.14053007 
 C  0.          0.          0.         
 C  4.41950026 -0.16452001  0.08243    
 N -2.57711015  3.58139021  1.03049006 
 N -2.88298017  4.42689026  3.10039018 
 N -0.89229005  3.4320902   2.58479015 
 H -2.8452184   4.08725233 -7.96569106 
 C -1.93058011  3.63679021 -7.65799045 
 H -1.6601401   4.11089024 -6.7617304  
 H -1.16235007  3.87599023 -8.34833049 
 C -2.04990012  2.20949013 -7.33058043 
 H -1.08692006  1.78573011 -7.05850042 
 H -2.43780014  1.6741801  -8.23554049 
 C -3.11184018  1.99921012 -6.23388037 
 H -3.11532018  1.01148006 -5.79542034 
 H -4.10829024  2.10179012 -6.59544039 
 N -3.11811018  3.05029018 -5.23416031 
 H -3.95787023  3.56869021 -5.25329031 
 C -2.30034014  3.17609019 -4.23414025 
 N -1.33959008  2.28369013 -4.08815024 
 H -0.52310003  2.52539015 -3.4467602  
 H -1.21521007  1.53340009 -4.74475028 
 N -2.30937014  4.15919025 -3.3954102  
 H -1.81017011  3.92719023 -2.51286015 
 H -2.73591016  5.0376903  -3.54513021 
 H -5.62148223  0.0749434   4.51897458 
 C -5.1050003   0.94497006  4.27319025 
 H -5.29814031  1.7734701   4.92579029 
 H -5.42952032  1.24591007  3.25369019 
 C -3.60350021  0.90213005  4.25479025 
 H -3.26256019  0.57044003  5.24969031 
 H -3.23193019  1.85316011  3.87329023 
 C -2.80272017 -0.10838001  3.3094902  
 O -1.99608012  0.47585003  2.53339015 
 O -2.79795016 -1.34740008  3.48879021
"""
debug_atoms = """
 H -5.8768512   5.12198377 -1.26494537 
 O  0.74030004  3.22499019 -2.63607016 
 """

mol = pyscf.M(
    spin=1,
    atom=atoms,  # water molecule
    basis="ccpvtz",  # basis set
    verbose=0,  # control the level of print info
)
print(mol.nao)

log = logger.new_logger(mol, 5)

log.init_timer = lambda **k: (logger.process_clock(), logger.perf_counter())

n_config = 1

t0 = log.init_timer()
plan = create_ovlp_plan_new(
    np.array([mol._atm for _ in range(n_config)]),
    np.array([mol._bas for _ in range(n_config)]),
    np.array([mol._env for _ in range(n_config)]),
    screening=False,
)
t0 = log.timer("overhead", *t0)

# check numerical + warm up
exp = get_ovlp(plan).get()
ref = mol.intor("int1e_ovlp")
assert np.linalg.norm(exp - ref) < 1e-9

origin = np.array([0.2, 0.3, 0.4])
exp = get_dipole(plan, origin).get()
with mol.with_common_origin(origin):
    ref = mol.intor("int1e_r")
assert np.linalg.norm(exp - ref) < 1e-9

exp = get_quadrupole(plan, origin).get()
with mol.with_common_origin(origin):
    ref = mol.intor("int1e_rr")
assert np.linalg.norm(exp - ref) < 1e-9

exp = get_ovlp_gradient(plan).get()
ref = mol.intor("int1e_ipovlp")
assert np.linalg.norm(exp - ref) < 1e-9

exp = get_dipole_gradient(plan, origin).get()
with mol.with_common_origin(origin):
    ref = mol.intor("int1e_irp")
assert np.linalg.norm(exp - ref) < 1e-9

exp = get_quadrupole_gradient(plan, origin).get()
with mol.with_common_origin(origin):
    ref = mol.intor("int1e_irrp")
assert np.linalg.norm(exp - ref) < 1e-9

# measure timing
n = 10
t0 = log.init_timer()
for _ in range(n):
    result = get_ovlp(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("ovlp", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_dipole(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("dipole", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_quadrupole(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("quadrupole", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_ovlp_gradient(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("gradient", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_dipole_gradient(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("dipole gradient", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_quadrupole_gradient(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("quadrupole gradient", *t0)

t0 = log.init_timer()
for _ in range(n):
    result = get_ovlp(plan)
    result = get_dipole(plan)
    result = get_quadrupole(plan)
    result = get_ovlp_gradient(plan)
    result = get_dipole_gradient(plan)
    result = get_quadrupole_gradient(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("family", *t0)

for _ in range(n):
    plan = create_ovlp_plan_new(
        np.array([mol._atm for _ in range(n_config)]),
        np.array([mol._bas for _ in range(n_config)]),
        np.array([mol._env for _ in range(n_config)]),
    )
    result = get_ovlp(plan)
    result = get_dipole(plan)
    result = get_quadrupole(plan)
    result = get_ovlp_gradient(plan)
    result = get_dipole_gradient(plan)
    result = get_quadrupole_gradient(plan)
cp.cuda.get_current_stream().synchronize()
t0 = log.timer("family with plan", *t0)

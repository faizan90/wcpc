# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64


cdef void gen_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL n_cpus,
    ) nogil

cdef void mod_cp_rules(
    DT_UL_NP_t[:, :] cp_rules,
    DT_UL_NP_t[:, :] cp_rules_idx_ctr,
    DT_UL_NP_t[:, :] loc_mod_ctr,
    const DT_UL max_idxs_ct,
    const DT_UL n_cps,
    const DT_UL n_pts,
    const DT_UL n_fuzz_nos,
    const DT_UL run_type,
    DT_UL *rand_k,
    DT_UL *rand_i,
    DT_UL *rand_v,
    DT_UL *old_v_i_k,
    ) nogil


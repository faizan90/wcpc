# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

### obj_ftns:False;True;False;False;False;False;False;False

### op_mp_memb_flag:True
### op_mp_obj_ftn_flag:True

import numpy as np
cimport numpy as np

from .gen_mod_cp_rules cimport (gen_cp_rules, mod_cp_rules)
from .memb_ftns cimport (calc_membs_dof_cps, update_membs_dof_cps)
from .cp_obj_ftns cimport (obj_ftn_refresh, obj_ftn_update)

ctypedef double DT_D
ctypedef unsigned long DT_UL
ctypedef long long DT_LL
ctypedef unsigned long long DT_ULL
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64
DT_UL_NP = np.uint64


cdef extern from "math.h" nogil:
    cdef:
        DT_D exp(DT_D x)
        bint isnan(DT_D x)

cdef extern from "./rand_gen.h" nogil:
    cdef: 
        DT_D rand_c()
        void warm_up()  # call this at least once
        void re_seed(DT_ULL x)  # calls warm_up as well

warm_up()


cpdef classify_cps(dict args_dict):
    cdef:
        # ulongs
        Py_ssize_t i, j, k, l
        int no_steep_anom_flag, gen_mod_cp_err_flag = 0, thresh_steep = 1
        DT_UL n_cps, n_pts, n_time_steps, n_fuzz_nos, msgs, n_max = 0
        DT_UL n_cpus, n_cpus_obj, n_cpus_mem
        DT_UL curr_n_iter, curr_m_iter, max_m_iters, max_n_iters
        DT_UL best_accept_iters, accept_iters, rand_acc_iters, reject_iters
        DT_UL rand_k, rand_i, rand_v, old_v_i_k, run_type, no_cp_val
        DT_UL curr_fuzz_idx, last_best_accept_n_iter, max_idxs_ct
        DT_UL rollback_iters_ct, new_iters_ct, update_iters_ct
        DT_UL max_temp_adj_atmps, curr_temp_adj_iter = 0
        DT_UL max_iters_wo_chng, curr_iters_wo_chng = 0, temp_adjed = 0
        DT_UL temp_adj_iters, min_acc_rate, max_acc_rate
        DT_UL n_anom_rows, n_anoms_cols, curr_anom_row, curr_anom_col

        # doubles
        DT_D anneal_temp_ini, temp_red_alpha, curr_anneal_temp, p_l
        DT_D best_obj_val, curr_obj_val, pre_obj_val, rand_p, boltz_p
        DT_D acc_rate, temp_inc, lo_freq_pen_wt, min_freq

        # other variables
        list curr_n_iters_list = []
        list curr_obj_vals_list = []
        list best_obj_vals_list = []
        list acc_rate_list = []
        list cp_pcntge_list = []
        list ants = [[], []]

        # 1D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] best_sel_cps
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] chnge_steps
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] sel_cps, old_sel_cps

        # 2D ulong arrays
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] cp_rules, best_cps
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] cp_rules_idx_ctr
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] best_cp_rules_idx_ctr
        np.ndarray[DT_UL_NP_t, ndim=2, mode='c'] loc_mod_ctr
        np.ndarray[np.uint32_t, ndim=2, mode='c'] rands_rec_arr
        np.ndarray[np.uint8_t, ndim=2, mode='c'] anom_crnr_flags_arr

        # 2D double arrays
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] slp_anom, fuzz_nos_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] dofs_arr, best_dofs_arr

        # 3D double arrays
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] mu_i_k_arr
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] cp_dof_arr

        # arrays for all obj. ftns.
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] obj_ftn_wts_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] ppt_cp_n_vals_arr

        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] in_cats_ppt_arr

        # ulongs for obj. ftns.
        Py_ssize_t q
        DT_UL n_cats

        # doubles obj. ftn. 2
        Py_ssize_t r
        DT_UL n_o_2_threshs

        # arrays for obj. ftn. 2
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] o_2_ppt_thresh_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] cats_ppt_mean_pis_arr
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] cats_obj_2_vals_arr
        np.ndarray[DT_D_NP_t, ndim=3, mode='c'] cats_ppt_cp_mean_pis_arr

    # read everythings from the given dict. Must do explicitly.
    in_cats_ppt_arr = args_dict['in_cats_ppt_arr_calib']
    n_cats = in_cats_ppt_arr.shape[1]
    n_max = max(n_max, n_cats)
    o_2_ppt_thresh_arr = args_dict['o_2_ppt_thresh_arr']
    n_o_2_threshs = o_2_ppt_thresh_arr.shape[0]

    obj_ftn_wts_arr = args_dict['obj_ftn_wts_arr']
    n_cps = args_dict['n_cps']
    no_cp_val = args_dict['no_cp_val']
    p_l = args_dict['p_l']
    fuzz_nos_arr = args_dict['fuzz_nos_arr']
    slp_anom = args_dict['slp_anom_calib']
    anneal_temp_ini = args_dict['anneal_temp_ini']
    temp_red_alpha = args_dict['temp_red_alpha']
    max_m_iters = args_dict['max_m_iters']
    max_n_iters = args_dict['max_n_iters']
    n_cpus_obj = args_dict['n_cpus_obj']
    n_cpus_mem = args_dict['n_cpus_mem']
    max_idxs_ct = args_dict['max_idxs_ct']
    max_iters_wo_chng = args_dict['max_iters_wo_chng']
    temp_adj_iters = args_dict['temp_adj_iters']
    min_acc_rate = args_dict['min_acc_rate']
    max_acc_rate = args_dict['max_acc_rate']
    max_temp_adj_atmps = args_dict['max_temp_adj_atmps']
    lo_freq_pen_wt = args_dict['lo_freq_pen_wt']
    min_freq = args_dict['min_freq']
    no_steep_anom_flag = <int> args_dict['no_steep_anom_flag']
    n_anom_rows = <DT_UL> args_dict['n_anom_rows']
    n_anom_cols = <DT_UL> args_dict['n_anom_cols']

    n_cpus = <DT_UL> max(n_cpus_obj, n_cpus_mem)

    if 'msgs' in args_dict:
        msgs = <DT_UL> args_dict[ 'msgs']

    else:
        msgs = 0

    assert n_cps >= 2, 'n_cps cannot be less than 2!'

    if msgs:
        print('\n')
        print('Calibrating CPs...')
        print('n_cats:', n_cats)
        print('o_2_ppt_thresh_arr:', o_2_ppt_thresh_arr)
        print('n_o_2_threshs:', n_o_2_threshs)
        print('n_cps:', n_cps)
        print('n_cpus:', n_cpus)
        print('n_cpus_obj:', n_cpus_obj)
        print('n_cpus_mem:', n_cpus_mem)
        print('no_cp_val:', no_cp_val)
        print('p_l:', p_l)
        print('fuzz_nos_arr:\n', fuzz_nos_arr)
        print('anneal_temp_ini:', anneal_temp_ini)
        print('temp_red_alpha:', temp_red_alpha)
        print('max_m_iters:', max_m_iters)
        print('max_n_iters:', max_n_iters)
        print('max_idxs_ct:', max_idxs_ct)
        print('obj_ftn_wts_arr:', obj_ftn_wts_arr)
        print('anom shape: (%d, %d)' % (slp_anom.shape[0], slp_anom.shape[1]))
        print('max_iters_wo_chng:', max_iters_wo_chng)
        print('temp_adj_iters:', temp_adj_iters)
        print('min_acc_rate:', min_acc_rate)
        print('max_acc_rate:', max_acc_rate)
        print('max_temp_adj_atmps:', max_temp_adj_atmps)
        print('lo_freq_pen_wt:', lo_freq_pen_wt)
        print('min_freq:', min_freq)
        print('n_max:', n_max)
        print('no_steep_anom_flag:', no_steep_anom_flag)
        print('n_anom_rows:', n_anom_rows)
        print('n_anom_cols:', n_anom_cols)
        print('in_cats_ppt_arr shape: (%d, %d)' % (in_cats_ppt_arr.shape[0], in_cats_ppt_arr.shape[1]))

    # initialize the required variables
    n_pts = slp_anom.shape[1]
    n_fuzz_nos = fuzz_nos_arr.shape[0]
    n_time_steps = slp_anom.shape[0]

    if max_idxs_ct > <DT_UL>(n_pts / n_fuzz_nos):
        max_idxs_ct = <DT_UL> max(1, (n_pts / n_fuzz_nos))
        print(("\n######### max_idxs_ct reset to %d!#########\n" % max_idxs_ct))

    curr_n_iter = 0
    curr_m_iter = 0
    curr_iters_wo_chng = 0
    best_obj_val = -np.inf
    pre_obj_val = best_obj_val
    curr_anneal_temp = anneal_temp_ini  # to change temp on the fly

    best_accept_iters = 0
    accept_iters = 0
    rand_acc_iters = 0
    reject_iters = 0

    new_iters_ct = 0
    update_iters_ct = 0
    rollback_iters_ct = 0

    rand_k = n_cps - 1
    rand_i = n_pts - 1
    rand_v = n_fuzz_nos
    old_v_i_k = n_fuzz_nos

    # run_type == 1 means fresh start i.e. everything is reset except
    # the cp_rules. This is done when curr_m_iter >= max_m_iter as well.
    # runtype == 2 means an update cycle i.e. values at the given CP and point
    # are changed. Also, only those days that have a changed CP are evaluated
    # further.
    # run_type = 3 means a rollback cycle i.e. everything is set to the last
    # value that it had on successful run.
    run_type = 1

    # initialize the required arrays
    cp_rules = np.random.randint(0, n_fuzz_nos + 1, size=(n_cps, n_pts), dtype=DT_UL_NP)

    cp_rules_idx_ctr = np.zeros(shape=(n_cps, n_fuzz_nos), dtype=DT_UL_NP)
    best_cp_rules_idx_ctr = cp_rules_idx_ctr.copy()
    loc_mod_ctr = np.zeros((n_cps, n_pts), dtype=DT_UL_NP)

    anom_crnr_flags_arr = np.ones((n_pts, 8), dtype=np.uint8)
    if no_steep_anom_flag:
        for k in range(n_pts):
            curr_anom_row = <DT_UL> (k / n_anom_cols)
            curr_anom_col = <DT_UL> (k % n_anom_cols)

            if curr_anom_row == 0:
                anom_crnr_flags_arr[k, 0] = 0
                anom_crnr_flags_arr[k, 1] = 0
                anom_crnr_flags_arr[k, 2] = 0

            if curr_anom_col == 0:
                anom_crnr_flags_arr[k, 0] = 0
                anom_crnr_flags_arr[k, 3] = 0
                anom_crnr_flags_arr[k, 5] = 0

            if curr_anom_row == (n_anom_rows - 1):
                anom_crnr_flags_arr[k, 5] = 0
                anom_crnr_flags_arr[k, 6] = 0
                anom_crnr_flags_arr[k, 7] = 0

            if curr_anom_col == (n_anom_cols - 1):
                anom_crnr_flags_arr[k, 2] = 0
                anom_crnr_flags_arr[k, 4] = 0
                anom_crnr_flags_arr[k, 7] = 0

    gen_cp_rules(
        cp_rules,
        cp_rules_idx_ctr,
        anom_crnr_flags_arr,
        no_steep_anom_flag,
        max_idxs_ct,
        n_cps,
        n_pts,
        n_fuzz_nos,
        n_cpus,
        n_anom_cols,
        thresh_steep,
        &gen_mod_cp_err_flag)

    if gen_mod_cp_err_flag:
        raise RuntimeError('gen_cp_rules failed. Choose a lower value for max_idxs_ct!')

    best_cps = cp_rules.copy()
    best_sel_cps = np.zeros(n_time_steps, dtype=DT_UL_NP)

    uni_cps, cps_freqs = np.unique(best_sel_cps, return_counts=True)
    cp_rel_freqs = 100 * cps_freqs / float(n_time_steps)
    cp_rel_freqs = np.round(cp_rel_freqs, 2)

    if msgs:
        print('\n%-10s:%s' % ('Unique CPs', 'Relative Frequencies (%)'))
        for x, y in zip(uni_cps, cp_rel_freqs):
            print('%10d:%-20.2f' % (x, y))

        print('\ncp_rules_idx_ctr:\n', cp_rules_idx_ctr.T)
        print('\nbest_cp_rules_idx_ctr:\n', best_cp_rules_idx_ctr.T)
        print(50 * '#', '\n\n')

    mu_i_k_arr = np.zeros(shape=(n_time_steps, n_cps, n_pts), dtype=DT_D_NP)
    cp_dof_arr = np.zeros(shape=(n_time_steps, n_cps, n_fuzz_nos), dtype=DT_D_NP)
    sel_cps = np.full(n_time_steps, no_cp_val, dtype=DT_UL_NP)
    old_sel_cps = sel_cps.copy()

    chnge_steps = np.zeros(n_time_steps, dtype=DT_UL_NP)
    dofs_arr = np.full((n_time_steps, n_cps), 0.0, dtype=DT_D_NP)
    best_dofs_arr = dofs_arr.copy()

    # an array to save all the randomly generated integers
    rands_rec_arr = np.full((max_n_iters + 5, 3), 9999, dtype=np.uint32)

    # initialize the obj. ftn. variables
    ppt_cp_n_vals_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)

    # initialize obj. ftn. 2 variables
    cats_ppt_mean_pis_arr = np.full((n_cats, n_o_2_threshs), 0.0, dtype=DT_D_NP)
    cats_ppt_cp_mean_pis_arr = np.full((n_cats, n_cps, n_o_2_threshs), 0.0, dtype=DT_D_NP)
    cats_obj_2_vals_arr = np.full((n_cats, n_o_2_threshs), 0.0, dtype=DT_D_NP)

    # fill some arrays used for obj. 2 and 5 ftns.
    for q in range(n_cats):
        for r in range(n_o_2_threshs):
            cats_ppt_mean_pis_arr[q, r] = np.mean(in_cats_ppt_arr[:, q] > o_2_ppt_thresh_arr[r])
            assert (not isnan(cats_ppt_mean_pis_arr[q, r]) and (cats_ppt_mean_pis_arr[q, r] > 0))

    # start simulated annealing
    while ((curr_n_iter < max_n_iters) and (curr_iters_wo_chng < max_iters_wo_chng)) or (not temp_adjed) or (run_type == 2):
        if (curr_m_iter >= max_m_iters) and (run_type == 2) and (temp_adjed):
            curr_m_iter = 0
            curr_anneal_temp *= temp_red_alpha
            run_type = 1

        mod_cp_rules(
            cp_rules,
            cp_rules_idx_ctr,
            loc_mod_ctr,
            anom_crnr_flags_arr,
            no_steep_anom_flag,
            max_idxs_ct,
            n_cps,
            n_pts,
            n_fuzz_nos,
            run_type,
            &rand_k,
            &rand_i,
            &rand_v,
            &old_v_i_k,
            n_anom_cols,
            thresh_steep,
            &gen_mod_cp_err_flag)

        if gen_mod_cp_err_flag:
            raise RuntimeError('mod_cp_rules failed. Choose a lower value for max_idxs_ct!')

        if run_type == 1:
            new_iters_ct += 1

        elif run_type == 2:
            update_iters_ct += 1

        elif run_type == 3:
            rollback_iters_ct += 1

        # fill/update the membership, DOF and selected CPs arrays
        if run_type == 1:
            calc_membs_dof_cps(
                cp_rules,
                mu_i_k_arr,
                cp_dof_arr,
                slp_anom,
                fuzz_nos_arr,
                dofs_arr,
                sel_cps,
                old_sel_cps,
                chnge_steps,
                no_cp_val,
                p_l,
                n_cpus_mem,
                n_time_steps,
                n_pts,
                n_cps,
                n_fuzz_nos)

        elif run_type == 2:
            update_membs_dof_cps(
                old_v_i_k,
                rand_v,
                rand_k,
                rand_i,
                cp_rules,
                mu_i_k_arr,
                cp_dof_arr,
                cp_rules_idx_ctr,
                slp_anom,
                fuzz_nos_arr,
                dofs_arr,
                sel_cps,
                old_sel_cps,
                chnge_steps,
                no_cp_val,
                p_l,
                n_cpus_mem,
                n_time_steps,
                n_cps,
                n_fuzz_nos)

        elif run_type == 3:
            update_membs_dof_cps(
                rand_v,
                old_v_i_k,
                rand_k,
                rand_i,
                cp_rules,
                mu_i_k_arr,
                cp_dof_arr,
                cp_rules_idx_ctr,
                slp_anom,
                fuzz_nos_arr,
                dofs_arr,
                sel_cps,
                old_sel_cps,
                chnge_steps,
                no_cp_val,
                p_l,
                n_cpus_mem,
                n_time_steps,
                n_cps,
                n_fuzz_nos)

        # calculate the objective function values
        if run_type == 1:
            # start from the begining
            curr_obj_val = obj_ftn_refresh(
                in_cats_ppt_arr,
                n_cats,
                cats_ppt_cp_mean_pis_arr,
                cats_ppt_mean_pis_arr,
                o_2_ppt_thresh_arr,
                cats_obj_2_vals_arr,
                n_o_2_threshs,
                ppt_cp_n_vals_arr,
                obj_ftn_wts_arr,
                sel_cps,
                lo_freq_pen_wt,
                min_freq,
                n_cpus_obj,
                n_cps,
                n_max,
                n_time_steps)

            run_type = 2

        else:
            # only update at steps where the CP has changed
            curr_obj_val = obj_ftn_update(
                in_cats_ppt_arr,
                n_cats,
                cats_ppt_cp_mean_pis_arr,
                cats_ppt_mean_pis_arr,
                o_2_ppt_thresh_arr,
                cats_obj_2_vals_arr,
                n_o_2_threshs,
                ppt_cp_n_vals_arr,
                obj_ftn_wts_arr,
                sel_cps,
                old_sel_cps,
                chnge_steps,
                lo_freq_pen_wt,
                min_freq,
                n_cpus_obj,
                n_cps,
                n_max,
                n_time_steps)

        #print(curr_m_iter, curr_n_iter, run_type, round(curr_obj_val, 2), round(pre_obj_val, 2))
        if run_type == 3:
            run_type = 2
            for i in range(n_time_steps):
                old_sel_cps[i] = sel_cps[i]

            continue

        assert not isnan(curr_obj_val), 'curr_obj_val is NaN!(%s)' % curr_n_iter

        #print(curr_m_iter, curr_n_iter, run_type, round(curr_obj_val, 2), round(pre_obj_val, 2))

        # a maximizing function
        if (curr_obj_val > best_obj_val) and (run_type == 2):
            best_obj_val = curr_obj_val
            last_best_accept_n_iter = curr_n_iter
            for i in range(n_time_steps):
                best_sel_cps[i] = sel_cps[i]
                for j in range(n_cps):
                    best_dofs_arr[i, j] = dofs_arr[i, j]

            for j in range(n_cps):
                for k in range(n_pts):
                    best_cps[j, k] = cp_rules[j, k]

                for l in range(n_fuzz_nos):
                    best_cp_rules_idx_ctr[j, l] = cp_rules_idx_ctr[j, l]

            best_accept_iters += 1

        if curr_obj_val > pre_obj_val:
            pre_obj_val = curr_obj_val
            accept_iters += 1

        else:
            rand_p = rand_c()
            boltz_p = exp((curr_obj_val - pre_obj_val) / curr_anneal_temp)
            if rand_p < boltz_p:
                pre_obj_val = curr_obj_val
                rand_acc_iters += 1

            else:
                run_type = 3
                #cp_rules[rand_k, rand_i] = old_v_i_k
                reject_iters += 1

        if run_type == 3:
            curr_iters_wo_chng += 1

        else:
            curr_iters_wo_chng = 0

        acc_rate = round(100.0 * (accept_iters + rand_acc_iters) / (accept_iters + rand_acc_iters + reject_iters), 6)

        if (not curr_m_iter) and temp_adjed:
            if msgs:
                print('\ncurr_m_iter:', curr_m_iter)
                print('curr_n_iter:', curr_n_iter)

                print('curr_obj_val:', curr_obj_val)
                print('pre_obj_val:', pre_obj_val)
                print('best_obj_val:', best_obj_val)

                print('best_accept_iters:', best_accept_iters)
                print('last_best_accept_n_iter:', last_best_accept_n_iter)
                print('accept_iters:', accept_iters)
                print('rand_acc_iters:', rand_acc_iters)
                print('reject_iters:', reject_iters)
                print('curr_anneal_temp:', curr_anneal_temp)

                print('new_iters_ct:', new_iters_ct)
                print('update_iters_ct:', update_iters_ct)
                print('rollback_iters_ct:', rollback_iters_ct)

                print('acceptance rate (%age):', acc_rate)
                print('curr_iters_wo_chng:', curr_iters_wo_chng)

                #print('cp_dof_arr min, max:', cp_dof_arr.min(), cp_dof_arr.max())

                print('rand_p, boltz_p:', rand_p, boltz_p)
                uni_cps, cps_freqs = np.unique(best_sel_cps, return_counts=True)
                cp_rel_freqs = 100 * cps_freqs / float(n_time_steps)
                cp_rel_freqs = np.round(cp_rel_freqs, 2)

                print('%-25s' % 'Unique CPs:', ['%5d' % int(_) for _ in uni_cps])
                print('%-25s' % 'Relative Frequencies (%):', ['%5.2f' % float(_) for _ in cp_rel_freqs])

                print('\nbest_cp_rules_idx_ctr:\n', best_cp_rules_idx_ctr.T)

        if (curr_n_iter >= temp_adj_iters) and (not temp_adjed) and (run_type == 2):
            print("\n\n#########Checking for acceptance rate#########")
            print('anneal_temp_ini:', anneal_temp_ini)
            if min_acc_rate <= acc_rate <= max_acc_rate:
                print('acc_rate (%f%%) is acceptable!' % acc_rate)
                temp_adjed = 1

            else:
                if ants[0] and ants[1]:
                    #print(ants)
                    if acc_rate < min_acc_rate:
                        print('accp_rate (%0.2f%%) is too low!' % acc_rate)
                        ants[0] = [acc_rate, anneal_temp_ini]

                    elif acc_rate > max_acc_rate:
                        print('accp_rate (%0.2f%%) is too high!' % acc_rate)
                        ants[1] = [acc_rate, anneal_temp_ini]

                    #print(anneal_temp_ini)
                    anneal_temp_ini = 0.5 * ((ants[1][1] + ants[0][1]))
                    curr_anneal_temp = anneal_temp_ini
                    #print(anneal_temp_ini)
                    #print(ants)

                else:
                    if acc_rate < min_acc_rate:
                        ants[0] = [acc_rate, anneal_temp_ini]
                        print('accp_rate (%0.2f%%) is too low!' % acc_rate)
                        temp_inc = (1 + ((min_acc_rate) * 0.01))
                        print('Increasing anneal_temp_ini by %0.2f%%...' % (100 * (temp_inc - 1)))
                        anneal_temp_ini = anneal_temp_ini * temp_inc
                        curr_anneal_temp = anneal_temp_ini

                    elif acc_rate > max_acc_rate:
                        ants[1] = [acc_rate, anneal_temp_ini]
                        print('accp_rate (%0.2f%%) is too high!' % acc_rate)
                        temp_inc = max(1e-6, (1 - ((acc_rate) * 0.01)))
                        print('Reducing anneal_temp_ini to %0.2f%%...' %  (100 * (1 - temp_inc)))
                        anneal_temp_ini = anneal_temp_ini * temp_inc
                        curr_anneal_temp = anneal_temp_ini

                if curr_temp_adj_iter < max_temp_adj_atmps:
                    run_type = 1
                    curr_n_iter = 0
                    curr_m_iter = 0
                    best_obj_val = -np.inf
                    continue

                else:
                    print('#######Could not converge to an acceptable annealing temperature in %d tries!#########')
                    print('Terminating optimization....')
                    raise RuntimeError

        curr_obj_vals_list.append(curr_obj_val)
        best_obj_vals_list.append(best_obj_val)
        acc_rate_list.append(acc_rate)
        cp_pcntge_list.append(ppt_cp_n_vals_arr.copy() / n_time_steps)
        curr_n_iters_list.append(curr_n_iter)
        rands_rec_arr[curr_n_iter, 0] = rand_k
        rands_rec_arr[curr_n_iter, 1] = rand_i
        rands_rec_arr[curr_n_iter, 2] = rand_v
        curr_m_iter += 1
        curr_n_iter += 1

    out_dict = {}
    for key in args_dict:
        out_dict[key] = args_dict[key]

    out_dict['n_pts_calib'] = n_pts
    out_dict['n_fuzz_nos'] = n_fuzz_nos
    out_dict['n_max'] = n_max
    out_dict['n_cats_calib'] = n_cats
    out_dict['n_time_steps_calib'] = n_time_steps
    out_dict['last_n_iter'] = curr_n_iter
    out_dict['last_m_iter'] = curr_m_iter
    out_dict['new_iters_ct'] = new_iters_ct
    out_dict['update_iters_ct'] = update_iters_ct
    out_dict['rollback_iters_ct'] = rollback_iters_ct
    out_dict['best_accept_iters'] = best_accept_iters
    out_dict['accept_iters'] = accept_iters
    out_dict['rand_acc_iters'] = rand_acc_iters
    out_dict['reject_iters'] = reject_iters
    out_dict['last_best_accept_n_iter'] = last_best_accept_n_iter
    out_dict['last_obj_val'] = curr_obj_val
    out_dict['best_obj_val'] = best_obj_val
    out_dict['pre_obj_val'] = pre_obj_val
    out_dict['last_anneal_temp'] = curr_anneal_temp
    out_dict['mu_i_k_arr_calib'] = mu_i_k_arr
    out_dict['dofs_arr_calib'] = dofs_arr
    out_dict['best_dofs_arr'] = best_dofs_arr
    out_dict['last_cp_rules'] = cp_rules
    out_dict['best_cp_rules'] = best_cps
    out_dict['best_sel_cps'] = best_sel_cps
    out_dict['last_sel_cps'] = sel_cps
    out_dict['old_sel_cps'] = old_sel_cps
    out_dict['last_cp_rules_idx_ctr'] = cp_rules_idx_ctr
    out_dict['best_cp_rules_idx_ctr'] = best_cp_rules_idx_ctr
    out_dict['curr_obj_vals_arr'] = np.array(curr_obj_vals_list)
    out_dict['best_obj_vals_arr'] = np.array(best_obj_vals_list)
    out_dict['cp_pcntge_arr'] = np.array(cp_pcntge_list)
    out_dict['curr_n_iters_arr'] = np.array(curr_n_iters_list, dtype=np.uint64)
    out_dict['acc_rate_arr'] = np.array(acc_rate_list)
    out_dict['loc_mod_ctr'] = loc_mod_ctr
    out_dict['rands_rec_arr'] = rands_rec_arr
    return out_dict


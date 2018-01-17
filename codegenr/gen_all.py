'''
Created on Dec 7, 2017

@author: Faizan-Uni
'''

# import os
# import timeit
# import time
from pathlib import Path
from fnmatch import fnmatch

from .write_cp_classi_main import write_cp_classi_main_lines
from .write_gen_mod_cp_rules import write_gen_mod_cp_rules_lines
from .write_memb_ftns import write_memb_ftns_lines
from .write_obj_ftns import write_obj_ftns_lines


def create_classi_cython_files(obj_1_flag=False,
                               obj_2_flag=False,
                               obj_3_flag=False,
                               obj_4_flag=False,
                               obj_5_flag=False,
                               obj_6_flag=False,
                               nonecheck=True,
                               boundscheck=True,
                               wraparound=True,
                               cdivision=True,
                               infer_types=None,
                               language_level=3,
                               force_compile=False,
                               out_dir=''):

    assert any([obj_1_flag,
                obj_2_flag,
                obj_3_flag,
                obj_4_flag,
                obj_5_flag,
                obj_6_flag])
    assert out_dir

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    params_dict['obj_1_flag'] = obj_1_flag
    params_dict['obj_2_flag'] = obj_2_flag
    params_dict['obj_3_flag'] = obj_3_flag
    params_dict['obj_4_flag'] = obj_4_flag
    params_dict['obj_5_flag'] = obj_5_flag
    params_dict['obj_6_flag'] = obj_6_flag
    
    out_dir = Path(out_dir)
    
    path_to_main_pyx = out_dir / 'cp_classi_main.pyx'
    path_to_cp_rules_pyx = out_dir / 'gen_mod_cp_rules.pyx'
    path_to_memb_ftns_pyx = out_dir / 'memb_ftns.pyx'

    compile_classi_main = False
    compile_gen_mod_cp_rules = False
    compile_memb_ftns = False
        
    if path_to_main_pyx.exists() and (not force_compile):
        mtch_str = ''
        compile_classi_main = True
        new_flags_list = [obj_1_flag,
                          obj_2_flag,
                          obj_3_flag,
                          obj_4_flag,
                          obj_5_flag,
                          obj_6_flag]
        
        with open(path_to_main_pyx, 'r') as pyx_hdl:
            for line in pyx_hdl:
                if fnmatch(line, '### obj_ftns:*'):
                    mtch_str = line
                    break

        if mtch_str:
            _ = (mtch_str.split(':')[1]).strip().split(';')
            old_flags_list = [True if x == 'True' else False for x in _]
            
            assert len(new_flags_list) == len(old_flags_list)

            if new_flags_list == old_flags_list:
                compile_classi_main = False
    else:
        compile_classi_main = True

    if (not path_to_cp_rules_pyx.exists()) or force_compile:
        compile_gen_mod_cp_rules = True

    if (not path_to_memb_ftns_pyx.exists()) or force_compile:
        compile_memb_ftns = True

    if compile_classi_main:
        write_cp_classi_main_lines(params_dict)
        write_obj_ftns_lines(params_dict)

    if compile_gen_mod_cp_rules:
        write_gen_mod_cp_rules_lines(params_dict)

    if compile_memb_ftns:
        write_memb_ftns_lines(params_dict)

    return


def create_justi_cython_files(nonecheck=True,
                              boundscheck=True,
                              wraparound=True,
                              cdivision=True,
                              infer_types=None,
                              language_level=3,
                              force_compile=False,
                              out_dir=''):

    assert out_dir

    out_dir = Path(out_dir)
    path_to_memb_ftns_pyx = out_dir / 'memb_ftns.pyx'

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    if force_compile:
        compile_memb_ftns = True
    else:
        compile_memb_ftns = False

    if not path_to_memb_ftns_pyx.exists():
        compile_memb_ftns = True

    if compile_memb_ftns:
        write_memb_ftns_lines(params_dict)

    return

# if __name__ == '__main__':
#     print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
#     START = timeit.default_timer()  # to get the runtime of the program
#
#     main_dir = Path(os.getcwd())
#
#     tab = '    '
#     nonecheck = False
#     boundscheck = False
#     wraparound = False
#     cdivision = True
#     language_level = 3
#     infer_types = None
#     out_dir = os.getcwd()
#
#     obj_1_flag = True
#     obj_2_flag = True
#     obj_3_flag = True
#     obj_4_flag = True
#     obj_5_flag = True
#
# #     nonecheck = True
# #     boundscheck = True
# #     wraparound = True
# #     cdivision = False
#
# #     obj_1_flag = False
# #     obj_2_flag = False
# #     obj_3_flag = False
# #     obj_4_flag = False
# #     obj_5_flag = False
#
#     os.chdir(main_dir)
#
#     assert any([obj_1_flag, obj_2_flag, obj_3_flag, obj_4_flag, obj_5_flag])
#
#     params_dict = {}
#     params_dict['tab'] = tab
#     params_dict['nonecheck'] = nonecheck
#     params_dict['boundscheck'] = boundscheck
#     params_dict['wraparound'] = wraparound
#     params_dict['cdivision'] = cdivision
#     params_dict['language_level'] = language_level
#     params_dict['infer_types'] = infer_types
#     params_dict['out_dir'] = out_dir
#
#     params_dict['obj_1_flag'] = obj_1_flag
#     params_dict['obj_2_flag'] = obj_2_flag
#     params_dict['obj_3_flag'] = obj_3_flag
#     params_dict['obj_4_flag'] = obj_4_flag
#     params_dict['obj_5_flag'] = obj_5_flag
#
#     write_cp_classi_main_lines(params_dict)
#     write_gen_mod_cp_rules_lines(params_dict)
#     write_memb_ftns_lines(params_dict)
#     write_obj_ftns_lines(params_dict)
#
#     obj_1_flag = True
#     obj_2_flag = True
#     obj_3_flag = True
#     obj_4_flag = True
#     obj_5_flag = True
#
#     params_dict['obj_1_flag'] = obj_1_flag
#     params_dict['obj_2_flag'] = obj_2_flag
#     params_dict['obj_3_flag'] = obj_3_flag
#     params_dict['obj_4_flag'] = obj_4_flag
#     params_dict['obj_5_flag'] = obj_5_flag
#     write_validate_cps_lines(params_dict)
#     write_validate_cps_qual_lines(params_dict)
#
#     STOP = timeit.default_timer()  # Ending time
#     print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
#            ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))

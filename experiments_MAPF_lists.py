from algs.alg_sipps import run_sipps
from algs.alg_temporal_a_star import run_temporal_a_star
from algs.alg_mapf_PrP import run_prp_sipps, run_prp_a_star, run_k_prp
from algs.alg_mapf_LNS2 import run_lns2, run_k_lns2
from algs.alg_mapf_pibt import run_pibt
from algs.alg_mapf_lacam import run_lacam
from algs.alg_mapf_lacam_star import run_lacam_star
from algs.alg_mapf_cga import run_cga_mapf
from algs.alg_mapf_cga_pure import run_cga_pure

# ------------------------------------------------------------------------------------------------------------ #
# General
# ------------------------------------------------------------------------------------------------------------ #

alg_list_general = [
    # ------------------------------------------------ #
    # PrP Family
    # ------------------------------------------------ #
    (run_prp_a_star, {
        'alg_name': f'PrP-A*',
        'constr_type': 'hard',
        'pf_alg_name': 'a_star',
        'pf_alg': run_temporal_a_star,
        'to_render': False,
    }),
    (run_prp_sipps, {
        'alg_name': f'PrP-SIPPS',
        'constr_type': 'hard',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'to_render': False,
    }),
    (run_k_prp, {
        'alg_name': f'15-PrP-A*',
        'constr_type': 'hard',
        'k_limit': 15,
        'pf_alg_name': 'a_star',
        'pf_alg': run_temporal_a_star,
        'to_render': False,
    }),
    (run_k_prp, {
        'alg_name': f'15-PrP-SIPPS',
        'constr_type': 'hard',
        'k_limit': 15,
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'to_render': False,
    }),
    # ------------------------------------------------ #

    # ------------------------------------------------ #
    # LNS2 Family
    # ------------------------------------------------ #
    (run_prp_sipps, {
        'alg_name': f'PrP',
        'constr_type': 'hard',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'to_render': False,
    }),
    # (run_lns2, {
    #     'alg_name': f'LNS2',
    #     'constr_type': 'soft',
    #     'n_neighbourhood': 5,
    #     'to_render': False,
    # }),
    (run_k_lns2, {
        'k_limit': (k_limit := 5),
        'alg_name': f'LNS2',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'n_neighbourhood': k_limit,
        'to_render': False,
    }),
    (run_pibt, {
        'alg_name': f'PIBT',
        'to_render': False,
    }),
    (run_lacam, {
        'alg_name': f'LaCAM',
        'to_render': False,
    }),
    (run_lacam_star, {
        'alg_name': f'LaCAM*',
        'flag_star': False,
        'to_render': False,
    }),
]

# ------------------------------------------------------------------------------------------------------------ #
# CGA Experiments
# ------------------------------------------------------------------------------------------------------------ #


alg_list_cga = [
    (run_prp_sipps, {
        'alg_name': f'PrP',
        'constr_type': 'hard',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'to_render': False,
    }),
    # (run_lns2, {
    #     'alg_name': f'LNS2',
    #     'constr_type': 'soft',
    #     'n_neighbourhood': 5,
    #     'to_render': False,
    # }),
    (run_k_lns2, {
        'k_limit': (k_limit := 15),
        'alg_name': f'LNS2',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
        'n_neighbourhood': k_limit,
        'to_render': False,
    }),
    (run_pibt, {
        'alg_name': f'PIBT',
        'to_render': False,
    }),
    (run_lacam, {
        'alg_name': f'LaCAM',
        'to_render': False,
    }),
    (run_lacam_star, {
        'alg_name': f'LaCAM*',
        'flag_star': False,
        # 'flag_star': True,
        'to_render': False,
    }),
    (run_cga_pure, {
        'alg_name': f'CGA-MAPF',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
    # (run_cga_mapf, {
    #     'alg_name': f'CGA+PIBT',
    #     'alt_goal_flag': 'first',
    #     'alt_goal_num': 1,
    #     'to_render': False,
    # }),

]


alg_list_top = [
    (run_lacam, {
        'alg_name': f'LaCAM',
        'to_render': False,
    }),
    (run_lacam_star, {
        'alg_name': f'LaCAM*',
        'flag_star': False,
        # 'flag_star': True,
        'to_render': False,
    }),
    (run_cga_pure, {
        'alg_name': f'CGA-MAPF',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
    # (run_cga_mapf, {
    #     'alg_name': f'CGA+PIBT',
    #     'alt_goal_flag': 'first',
    #     'alt_goal_num': 1,
    #     'to_render': False,
    # }),

]


alg_list_cga_only = [
    (run_cga_pure, {
        'alg_name': f'CGA-MAPF',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
    (run_cga_mapf, {
        'alg_name': f'CGA+PIBT',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),

]


alg_list_cga_lacam = [
    (run_cga_pure, {
        'alg_name': f'MACGA',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
    (run_cga_mapf, {
        'alg_name': f'MACGA+PIBT',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
    (run_lacam_star, {
        'alg_name': f'LaCAM*',
        'flag_star': False,
        'to_render': False,
    }),

]

# ------------------------------------------------------------------------------------------------------------ #
# MACGA Experiments
# ------------------------------------------------------------------------------------------------------------ #

alg_list_MACGA_paper_experiments = [
    # ------------------------------------------------ #
    # PrP Family
    # ------------------------------------------------ #
    # (run_k_prp, {
    #     'alg_name': f'PrP',
    #     'constr_type': 'hard',
    #     'k_limit': 15,
    #     'pf_alg_name': 'sipps',
    #     'pf_alg': run_sipps,
    #     'to_render': False,
    # }),
    # ------------------------------------------------ #
    # LNS2 Family
    # ------------------------------------------------ #
    # (run_k_lns2, {
    #     'k_limit': (k_limit := 15),
    #     'alg_name': f'LNS2',
    #     'pf_alg_name': 'sipps',
    #     'pf_alg': run_sipps,
    #     'n_neighbourhood': k_limit,
    #     'to_render': False,
    # }),
    # ------------------------------------------------ #
    # PIBT Family
    # ------------------------------------------------ #
    # (run_pibt, {
    #     'alg_name': f'PIBT',
    #     'to_render': False,
    # }),
    # (run_lacam, {
    #     'alg_name': f'LaCAM',
    #     'to_render': False,
    # }),
    (run_lacam_star, {
        'alg_name': f'LaCAM*',
        'flag_star': False,
        'to_render': False,
    }),
    # ------------------------------------------------ #
    # MACGA Family
    # ------------------------------------------------ #
    # (run_cga_pure, {
    #     'alg_name': f'MACGA',
    #     'alt_goal_flag': 'first',
    #     'alt_goal_num': 1,
    #     'to_render': False,
    # }),
    (run_cga_mapf, {
        'alg_name': f'MACGA+PIBT',
        'alt_goal_flag': 'first',
        'alt_goal_num': 1,
        'to_render': False,
    }),
]

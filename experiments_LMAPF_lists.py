from globals import *
from functions_general import *
from functions_plotting import *

from algs.alg_sipps import run_sipps
from algs.alg_temporal_a_star import run_temporal_a_star
from algs.alg_lifelong_PrP import run_lifelong_prp
from algs.alg_lifelong_LNS2 import run_lifelong_LNS2
from algs.alg_lifelong_PIBT import run_lifelong_pibt
from algs.alg_lifelong_cga import run_lifelong_cga
from algs.alg_lifelong_cga_pure import run_lifelong_cga_pure


alg_list_general = [
        # (run_lifelong_prp, {
        #     'alg_name': f'PrP',
        #     'constr_type': 'hard',
        #     'pf_alg': run_sipps,
        #     'pf_alg_name': 'sipps',
        #     'k_limit': 5,
        #     'to_render': False,
        # }),
        # (run_lifelong_prp, {
        #     'alg_name': f'PrP-A*',
        #     'constr_type': 'hard',
        #     'pf_alg': run_temporal_a_star,
        #     'pf_alg_name': 'a_star',
        #     'k_limit': 5,
        #     'to_render': False,
        # }),
        # (run_lifelong_LNS2, {
        #     'alg_name': f'LNS2-SIPPS',
        #     'constr_type': 'soft',
        #     'k_limit': 5,
        #     'n_neighbourhood': 5,
        #     'pf_alg_name': 'sipps',
        #     'pf_alg': run_sipps,
        #     'to_render': False,
        # }),
        # (run_lifelong_LNS2, {
        #     'alg_name': f'LNS2',
        #     'constr_type': 'hard',
        #     'k_limit': 5,
        #     'n_neighbourhood': 5,
        #     'pf_alg_name': 'a_star',
        #     'pf_alg': run_temporal_a_star,
        #     'to_render': False,
        # }),
        # (run_lifelong_cga, {
        #     'alg_name': f'CGA(L)+PIBT',
        #     # 'alt_goal_flag': 'all',
        #     'alt_goal_flag': 'first',
        #     # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        #     'to_render': False,
        # }),
        # (run_lifelong_pibt, {
        #     'alg_name': f'PIBT',
        #     'to_render': False,
        # }),
        (run_lifelong_cga_pure, {
            'alg_name': f'CGA(L)',
            'alt_goal_flag': 'first',
            'alt_goal_num': 1,
            'to_render': False,
        }),
    ]


alg_list = [
        (run_lifelong_prp, {
            'alg_name': f'L-PrP-SIPPS',
            'constr_type': 'hard',
            'pf_alg': run_sipps,
            'pf_alg_name': 'sipps',
            'k_limit': 5,
            'final_render': False,
        }),
        (run_lifelong_prp, {
            'alg_name': f'L-PrP-A*',
            'constr_type': 'hard',
            'pf_alg': run_temporal_a_star,
            'pf_alg_name': 'a_star',
            'k_limit': 5,
            'final_render': False,
        }),
        (run_lifelong_LNS2, {
            'alg_name': f'L-LNS2-SIPPS',
            'constr_type': 'soft',
            'k_limit': 5,
            'n_neighbourhood': 5,
            'pf_alg_name': 'sipps',
            'pf_alg': run_sipps,
            'final_render': False,
        }),
        (run_lifelong_LNS2, {
            'alg_name': f'L-LNS2-A*',
            'constr_type': 'hard',
            'k_limit': 5,
            'n_neighbourhood': 5,
            'pf_alg_name': 'a_star',
            'pf_alg': run_temporal_a_star,
            'final_render': False,
        }),
        (run_lifelong_pibt, {
            'alg_name': f'L-PIBT',
            'final_render': False,
        }),
    ]

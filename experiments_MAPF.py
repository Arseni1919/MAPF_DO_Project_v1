from globals import *
from functions_general import *
from functions_plotting import *

from algs.alg_sipps import run_sipps
from algs.alg_temporal_a_star import run_temporal_a_star
from algs.alg_mapf_PrP import run_prp_sipps, run_prp_a_star, run_k_prp
from algs.alg_mapf_LNS2 import run_lns2, run_k_lns2
from algs.alg_mapf_pibt import run_pibt
from algs.alg_mapf_lacam import run_lacam
from algs.alg_mapf_lacam_star import run_lacam_star
from experiments_MAPF_lists import *


@use_profiler(save_dir='stats/experiments_mapf.pstat')
def run_mapf_experiments():
    # ------------------------------------------------------------------------------------------------------------ #
    # General params
    # ------------------------------------------------------------------------------------------------------------ #
    set_seed(random_seed_bool=False, seed=119)
    # set_seed(random_seed_bool=False, seed=9256)  # 500 - room
    # set_seed(random_seed_bool=False, seed=1112)
    # set_seed(random_seed_bool=True)

    # ------------------------------------------------------------------------------------------------------------ #
    # MAPF
    # ------------------------------------------------------------------------------------------------------------ #
    # img_dir = '10_10_my_corridor.map'
    # img_dir = '10_10_my_rand.map'
    # img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    # img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'

    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    img_dir = 'maze-32-32-4.map'

    # ------------------------------------------------- #

    # n_agents_list = [400]
    # n_agents_list = [600]
    # n_agents_list = [300, 350, 400, 450]
    # n_agents_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # n_agents_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20]
    # n_agents_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # n_agents_list = [50, 100, 150, 200, 250, 300, 350]
    # n_agents_list = [150, 200, 250, 300, 350]
    # n_agents_list = [10, 50, 100, 150, 200, 250, 300, 350, 400]
    # n_agents_list = [50, 100, 150, 200, 250, 300, 350, 400]
    # n_agents_list = [350, 400, 450, 500, 550]
    # n_agents_list = [200, 300, 400, 500, 600, 700]  # empty, rand10
    # n_agents_list = [100, 200, 300, 400, 500, 600]  # rand20
    # n_agents_list = [600]
    # n_agents_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550] # maze4
    # n_agents_list = [100, 150, 200, 250, 300, 350, 400, 450] # maze2, room
    # n_agents_list = [100, 150, 200, 250, 300, 350, 400, 450]
    # n_agents_list = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    # n_agents_list = [200, 300, 400, 500, 600]
    n_agents_list = [250, 300]

    # ------------------------------------------------- #

    # i_problems = 2
    # i_problems = 3
    # i_problems = 5
    # i_problems = 10
    # i_problems = 15
    # i_problems = 20
    i_problems = 30

    # ------------------------------------------------- #

    # limits
    # max_time = 1e7  # seconds
    max_time = 60  # seconds
    # max_time = 30  # seconds
    # max_time = 10  # seconds
    # max_time = 5  # seconds
    # debug
    # to_assert = True
    to_assert = False
    # rendering
    to_render = False
    # to_render = True
    # final_render = False
    # saving
    # to_save = False
    to_save = True

    # ------------------------------------------------- #

    # ------------------------------------------------- #

    # alg_list = alg_list_cga
    # alg_list = alg_list_top
    # alg_list = alg_list_cga_only
    # alg_list = alg_list_cga_lacam
    alg_list = alg_list_MACGA_paper_experiments

    # ------------------------------------------------- #

    # ------------------------------------------------- #

    logs_dict: Dict[str, Any] = {
        params['alg_name']: {
            f'{n_agents}': {
                'soc': [],
                'makespan': [],
                'sr': [],
                'time': [],
            } for n_agents in n_agents_list
        } for alg, params in alg_list
    }
    logs_dict['alg_names'] = [params['alg_name'] for alg, params in alg_list]
    logs_dict['n_agents_list'] = n_agents_list
    logs_dict['i_problems'] = i_problems
    logs_dict['img_dir'] = img_dir
    logs_dict['max_time'] = max_time
    logs_dict['expr_type'] = 'MAPF'

    # ------------------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------------------ #
    if to_render:
        fig, ax = plt.subplots(2, 2, figsize=(17, 10))
    to_continue_dict = {params['alg_name']: True for alg, params in alg_list}

    path_to_maps: str = 'maps'
    path_to_heuristics: str = 'logs_for_heuristics'
    path_to_sv_maps: str = 'logs_for_freedom_maps'

    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    h_dict = exctract_h_dict(img_dir, path_to_heuristics)
    blocked_sv_map: np.ndarray = get_blocked_sv_map(img_dir, folder_dir=path_to_sv_maps)


    for n_agents in n_agents_list:

        for i_problem in range(i_problems):

            start_nodes: List[Node] = random.sample(nodes, n_agents)
            goal_nodes: List[Node] = random.sample(nodes, n_agents)

            for alg, params in alg_list:

                # preps
                params['img_np'] = img_np
                params['blocked_sv_map'] = blocked_sv_map
                params['max_time'] = max_time
                alg_name = params['alg_name']

                solved, paths_dict, alg_info = False, None, {}
                if to_continue_dict[alg_name]:
                    # the run
                    paths_dict, alg_info = alg(
                        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim, params
                    )
                    solved = paths_dict is not None

                if not solved:
                    break
                if solved:
                    logs_dict[alg_name][f'{n_agents}']['sr'].append(1)
                    logs_dict[alg_name][f'{n_agents}']['time'].append(alg_info['time'])
                    logs_dict[alg_name][f'{n_agents}']['makespan'].append(get_makespan_metric(paths_dict))
                    logs_dict[alg_name][f'{n_agents}']['soc'].append(get_soc_metric(paths_dict))
                else:
                    logs_dict[alg_name][f'{n_agents}']['sr'].append(0)
                print(f'\n{n_agents=}, {i_problem=}, {alg_name=}')

            # plot
            if to_render:
                plot_sr(ax[0, 0], info=logs_dict)
                plot_time_metric(ax[0, 1], info=logs_dict)
                # plot_time_metric_cactus(ax[0, 1], info=logs_dict)
                plot_makespan(ax[1, 1], info=logs_dict)
                plot_soc(ax[1, 0], info=logs_dict)
                # plot_soc_cactus(ax[1, 0], info=logs_dict)
                # plot_makespan_cactus(ax[1, 1], info=logs_dict)
                plt.tight_layout()
                plt.pause(0.01)

        # check if solved all the prev problems
        for alg, params in alg_list:
            if sum(logs_dict[params['alg_name']][f'{n_agents}']['sr']) == 0:
                to_continue_dict[params['alg_name']] = False

    if to_save:
        save_results(logs_dict)

    print('\n[INFO]: finished BIG MAPF experiments')
    if to_render:
        plt.show()


# if final_render:
#     fig, ax = plt.subplots(1, 2, figsize=(14, 7))


if __name__ == '__main__':
    run_mapf_experiments()

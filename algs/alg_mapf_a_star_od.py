from algs.alg_functions_cga import *
from run_single_MAPF_func import run_mapf_alg


def run_a_star_od(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict
) -> Tuple[None, Dict] | Tuple[Dict[str, List[Node]], Dict]:
    max_time: int | float = params['max_time']
    alg_name: str = params['alg_name']
    to_render: bool = params['to_render']
    img_np: np.ndarray = params['img_np']
    blocked_sv_map: np.ndarray = params['blocked_sv_map']

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    start_time = time.time()

    # create agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    n_agents = len(agents_dict)

    # algorithm
    # TODO

    # plot

    # return None, {}
    return {a.name: a.path for a in agents}, {'agents': agents, 'time': None, 'makespan': None}


@use_profiler(save_dir='../stats/alg_a_star_od.pstat')
def main():

    to_render = True
    # to_render = False

    params = {
        'max_time': 1000,
        'alg_name': 'A*-OD',
        'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        # 'alt_goal_flag': 'all',
        'to_render': to_render,
    }
    # run_mapf_alg(alg=run_cga_pure, params=params, final_render=False)
    run_mapf_alg(alg=run_a_star_od, params=params, final_render=True)


if __name__ == '__main__':
    main()
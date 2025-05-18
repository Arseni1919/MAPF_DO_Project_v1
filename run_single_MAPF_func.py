from globals import *
from functions_general import *
from functions_plotting import *


def run_mapf_alg(alg, params, final_render: bool = True, random_seed: bool = False):
    if random_seed:
        set_seed(random_seed_bool=True)
    else:
        set_seed(random_seed_bool=False, seed=5)
        # set_seed(random_seed_bool=False, seed=123)
        # set_seed(random_seed_bool=False, seed=2953)
    img_dir = params['img_dir']
    n_agents = params['num_agents']
    n_goal_nodes = params['num_goals']
    cost_func = params['cost_func']
    path_to_maps: str = '../maps'
    path_to_heuristics: str = '../logs_for_heuristics'
    path_to_sv_maps: str = '../logs_for_freedom_maps'
    print("CWD:", os.getcwd())
    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)

    h_dict: Dict[str, np.ndarray] = exctract_h_dict(img_dir, path_to_heuristics)

    try:
        blocked_sv_map: np.ndarray = get_blocked_sv_map(img_dir, folder_dir=path_to_sv_maps)
    except:
        RuntimeWarning("blocked_sv_map is None")
        blocked_sv_map = None
    # sv_map: np.ndarray = get_sv_map(img_dir, folder_dir=path_to_sv_maps)

    start_nodes: List[Node] = random.sample(nodes, n_agents)
    # goal_nodes: List[Node] = random.sample(nodes, n_agents)
    goal_nodes: List[Node] = random.sample(nodes, n_goal_nodes)
    # start_nodes: List[Node] = [nodes_dict['4_8'], nodes_dict['4_4'], nodes_dict['8_8']]
    # goal_nodes: List[Node] = [nodes_dict['4_2'], nodes_dict['4_4'], nodes_dict['8_8']]
    # start_nodes: List[Node] = [nodes_dict['4_8'], nodes_dict['4_4']]
    # goal_nodes: List[Node] = [nodes_dict['4_2'], nodes_dict['4_4']]
    # goal_nodes: List[Node] = [nodes_dict['4_1'], nodes_dict['4_0']]
    # start_nodes: List[Node] = [nodes_dict['4_0'], nodes_dict['4_1'], nodes_dict['4_2']]
    # goal_nodes: List[Node] = [nodes_dict['4_1'], nodes_dict['4_0'], nodes_dict['4_2']]

    # start_nodes: List[Node] = [nodes_dict['4_0'], nodes_dict['4_1'], nodes_dict['4_2'], nodes_dict['4_3']]
    # goal_nodes: List[Node] = [nodes_dict['4_1'], nodes_dict['4_0'], nodes_dict['4_2'], nodes_dict['4_3']]

    params['img_np'] = img_np
    # params['sv_map'] = sv_map
    params['blocked_sv_map'] = blocked_sv_map
    paths_dict, info = alg(
        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim, params
    )
    print('\n')
    print(paths_dict)
    validate_no_collisions_by_timestamp(paths_dict)
    # plot
    if final_render and paths_dict is not None:
        agents: List = info['agents']
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.001
        # plot_rate = 0.5
        # plot_rate = 1
        max_path_len = max([len(a.path) for a in agents])
        soc = sum([len(a.path) for a in agents])

        print(f'\n{max_path_len=}')
        print(f'{soc=}')
        for i in range(max_path_len):
            # update curr nodes
            for a in agents:
                a.update_curr_node(i)
            # plot the iteration
            i_agent = agents[0]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
                'iteration': i,
            }
            plot_step_in_env(ax[0], plot_info)
            plt.pause(plot_rate)
        plt.show()


def validate_no_collisions_by_timestamp(paths: dict[str, list[str]]) -> bool:
    max_len = max(len(path) for path in paths.values())

    for t in range(max_len):
        positions_at_t = set()
        for agent, path in paths.items():
            # Clamp to the last position if agent's path is shorter
            pos = path[t] if t < len(path) else path[-1]
            if pos in positions_at_t:
                print(f"FAIL! -> Collision at time {t} on node '{pos}'")
                return False
            positions_at_t.add(pos)

    print("No collisions detected.")
    return True






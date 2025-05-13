import copy

from algs.alg_mapf_lacam import *
from algs.alg_functions_a_star_od import *
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
    agents, agents_dict = create_agents_od(start_nodes, goal_nodes, h_dict)

    # create config for start
    config_start: Dict[str, Node] = {a.name: a.start_node for a in agents}
    config_start_node = ConfigNode(config_start, agents, g=0, h=0)

    # create config for goal
    config_goal: Dict[str, Node] = {a.name: a.goal_node for a in agents}
    config_goal_agents: Dict[AgentOD, Node] = {a: a.start_node for a in agents}
    config_goal_name: str = get_config_name(config_goal)

    # open is a heap of configs
    open_list = []
    heapq.heappush(open_list, config_start_node)

    # closed_list is a set on configNodes
    closed_list = set()
    print(f'The start is at {get_config_name(config_start)}')
    print(f'The goal is at {config_goal_name}')
    iteration = 0
    completed_steps = 0

    while len(open_list) > 0 and time_is_good(start_time, max_time):  # line 4 in pseudocode

        curr_config: ConfigNode = heapq.heappop(open_list)


        # if curr_config.name in closed_list:
        #     continue

        if curr_config.is_intermediate:  # intermediate nodes

            for idx, agent in enumerate(curr_config.agents):
                # or agent.curr_node.xy_name == agent.goal_node_name
                if agent.assigned_move:
                    continue
                else:
                    neighbours = agent.pre_move_position.neighbours_nodes
                    for n in neighbours:
                        if neighbour_taken(idx, agent, n, curr_config):
                            continue
                        agent_h = 0
                        if agent.goal_node:
                            agent_h = h_dict[agent.goal_node_name][n.x][n.y]
                        agent.heuristic = agent_h
                        new_config_node = get_next_config(agent, curr_config, n, h_dict, True, idx)
                        heapq.heappush(open_list, new_config_node)

                    break
            if curr_config.agents[-1].assigned_move:
                completed_steps += 1

        else:  # standard nodes

            # goal test
            if goal_test(curr_config):
                curr_config.chosen_agent = None
                paths_dict = backtrack_od(curr_config)
                for a_name, path in paths_dict.items():
                    agents_dict[a_name].path = path
                runtime = time.time() - start_time
                makespan: int = max([len(a.path) for a in agents])
                return paths_dict, {'agents': agents, 'time': runtime, 'makespan': makespan}

            # check for duplicate standard nodes
            if curr_config.name in closed_list:
                continue

            agent_0 = curr_config.agents[0]
            neighbours = agent_0.pre_move_position.neighbours_nodes
            for n in neighbours:
                # if (agent_0.curr_node.xy_name == agent_0.goal_node_name
                #         and n != agent_0.curr_node):
                #     continue

                # add to open
                agent_h = h_dict[curr_config.agents[0].goal_node_name][n.x][n.y]
                agent_0.heuristic = agent_h
                new_config_node = get_next_config(agent_0, curr_config, n, h_dict, False)

                heapq.heappush(open_list, new_config_node)
                closed_list.add(curr_config.name)





        runtime = time.time() - start_time
        print(
            f'\r{'*' * 10} | '
            f'[{alg_name}] {iteration=: <3} | '
            #f'finished: {N_new.finished}/{n_agents: <3} | '
            f'runtime: {runtime: .2f} seconds | '
            f'{len(open_list)=} | '
            f'{len(closed_list)=} | '
            f'{'*' * 10}',
            end='')
        iteration += 1

    # return None, {}
    if len(open_list) == 0:
        print("Open was empty, no solution found.")
    else:
        print("Time out.")

    return {a.name: a.path for a in agents}, {'agents': agents, 'time': None, 'makespan': None}


@use_profiler(save_dir='../stats/alg_a_star_od.pstat')
def main():

    # to_render = True
    to_render = True

    params = {
        'max_time': 1000,
        'alg_name': 'A*-OD',
        'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        # 'alt_goal_flag': 'all',
        'to_render': to_render,
    }

    run_mapf_alg(alg=run_a_star_od, params=params, final_render=True)



if __name__ == '__main__':
    main()
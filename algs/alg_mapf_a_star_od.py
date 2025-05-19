import heapq
import time
from matplotlib import pyplot as plt
from algs.alg_functions_a_star_od import *
from functions_general import use_profiler, time_is_good
from run_single_MAPF_func import run_mapf_alg
from typing import Tuple, Dict, List, Union


def run_a_star_od(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Union[Tuple[None, Dict], Tuple[Dict[str, List[Node]], Dict]]:
    max_time: int = params['max_time']
    alg_name: str = params['alg_name']
    to_render: bool = params['to_render']
    img_np: np.ndarray = params['img_np']
    cost_func: str = params['cost_func']
    blocked_sv_map: np.ndarray = params['blocked_sv_map']
    weights = params['weights']

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
    completed_steps = -1
    expansions = 0

    while len(open_list) > 0 and time_is_good(start_time, max_time):  # line 4 in pseudocode

        curr_config: ConfigNode = heapq.heappop(open_list)
        expansions += 1
        if curr_config.is_intermediate:  # intermediate nodes
            agents_counter = 0
            for idx, agent in enumerate(curr_config.agents):
                agents_counter += 1
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
                        new_config_node = get_next_config(agent, curr_config, n, h_dict, True, idx, cost_func, weights)
                        heapq.heappush(open_list, new_config_node)

                    break

        else:  # standard nodes
            # if all(agent.curr_node.xy_name == agent.goal_node_name for agent in curr_config.agents):
            #     continue

            # goal test
            if goal_test(curr_config):
                # curr_config.chosen_agent = None

                paths_dict = backtrack_od(curr_config)
                for a_name, path in paths_dict.items():
                    agents_dict[a_name].path = path
                runtime = time.time() - start_time
                makespan: int = max([len(a.path) for a in agents])
                fuel: int = get_fuel(agents)
                moved_agents: int = len(curr_config.moved_yet)
                print(
                    f" runtime: {runtime},"
                    f" expansions: {expansions},"
                    f" completed steps: {completed_steps},"
                    f" makespan: {makespan}",
                    f" fuel: {fuel}",
                    f" moved agents: {moved_agents}",)
                return paths_dict, {'agents': agents, 'time': runtime, 'makespan': makespan, 'moved agents': moved_agents}

            completed_steps += 1

            # check for duplicate standard nodes
            if curr_config.name in closed_list:
                continue

            agent_0 = curr_config.agents[0]
            neighbours = agent_0.pre_move_position.neighbours_nodes
            for n in neighbours:
                agent_h = h_dict[curr_config.agents[0].goal_node_name][n.x][n.y]
                agent_0.heuristic = agent_h
                new_config_node = get_next_config(agent_0, curr_config, n, h_dict, False, 0, cost_func, weights)

                heapq.heappush(open_list, new_config_node)
            closed_list.add(curr_config.name)

        # runtime = time.time() - start_time
        # print(
        #     f'\r{'*' * 10} | '
        #     f'[{alg_name}] {iteration=: <3} | '
        #     f'finished: {N_new.finished}/{n_agents: <3} | '
        #     f'runtime: {runtime: .2f} seconds | '
        #     f'{len(open_list)=} | '
        #     f'{len(closed_list)=} | '
        #     f'{'*' * 10}',
        #     end='')
        iteration += 1

    # return None, {}
    if len(open_list) == 0:
        print("Open was empty, no solution found.")
    else:
        print("Time out.")

    return {a.name: a.path for a in agents}, {'agents': agents, 'time': None, 'makespan': None}


@use_profiler(save_dir='../stats/alg_a_star_od.pstat')
def main():
    to_render = True
    max_time = 3600
    num_agents = 8
    # cost_func = 'makespan'
    # cost_func = 'fuel'
    # cost_func = 'moved_agents'
    cost_func = 'average'
    weights = {'moved_agents': 0, 'fuel': 1, 'makespan': 0}
    for num_goals in range(1, 6):
        img_dir = "4_4_random.map"
        params = {
            'max_time': max_time,
            'alg_name': 'A*-OD',
            'alt_goal_flag': 'first',
            'to_render': to_render,
            'num_agents': num_agents,
            'num_goals': num_goals,
            'img_dir': img_dir,
            'cost_func': cost_func,
            'weights': weights
        }

        run_mapf_alg(alg=run_a_star_od, params=params, final_render=True)




import platform

def beep():
    if platform.system() == "Windows":
        # Windows beep
        import winsound
        winsound.Beep(1000, 500)  # frequency (Hz), duration (ms)
    else:
        # Unix beep (requires `beep` command installed or fallback)
        print('\a')  # Try terminal bell
        # os.system('beep')  # Uncomment if you have the `beep` utility installed



if __name__ == '__main__':
    main()

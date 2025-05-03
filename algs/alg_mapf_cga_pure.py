import heapq

from algs.alg_functions_cga import *
from algs.alg_functions_pibt import run_procedure_pibt
from run_single_MAPF_func import run_mapf_alg


def run_cga_pure(
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
    agents.sort(key=lambda a: a.priority, reverse=True)

    iteration = 0
    finished = False
    while not finished:

        # PREPARATIONS
        (config_from, occupied_from, config_to, occupied_to,
         cga_step_agents_names, cga_curr_step_lists, blocked_nodes_names) = get_preparations(agents, iteration)
        last_visit_dict: Dict[str, int] = {n.xy_name: 0 for n in nodes}

        # calc the step
        for agent in agents:
            # if planned, continue
            if agent.name in config_to:
                continue
            # if at its alt goal, switch to the original goal
            if agent.alt_goal_node is not None and config_from[agent.name] == agent.alt_goal_node:
                agent.alt_goal_node = None
            goal_node = agent.get_goal_node()
            non_sv_nodes_np = blocked_sv_map[goal_node.x, goal_node.y]
            moved_agents = calc_cga_step(
                agent, iteration,
                config_from, occupied_from, config_to, occupied_to,
                agents, agents_dict, nodes, nodes_dict, last_visit_dict, h_dict, non_sv_nodes_np,
                blocked_nodes_names, params, start_time, max_time)
            for m_a in moved_agents:
                heapq.heappush(cga_step_agents_names, m_a.name)
            cga_curr_step_lists.append(moved_agents)
            update_blocked_nodes_names_after_cga(blocked_nodes_names, moved_agents, iteration)
            continue

        # execute the step + check the termination condition
        finished = True
        agents_finished = []
        for agent in agents:
            if agent.name in config_to:
                next_node = config_to[agent.name]
            else:
                next_node = config_from[agent.name]
                config_to[agent.name] = next_node
                occupied_to[next_node.xy_name] = agent
                agent.message += f'| [{iteration}] stay |'
            if agent.name not in cga_step_agents_names:
                # from PIBT
                agent.path.append(next_node)
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            if agent.curr_node != agent.goal_node:
                finished = False
                agent.priority += 1
            else:
                agent.priority = agent.init_priority
                agents_finished.append(agent)

        # unfinished first
        agents.sort(key=lambda a: a.priority, reverse=True)

        # print + render
        runtime = time.time() - start_time
        print(f'\r{'*' * 10} | [{alg_name}] {iteration=: <3} | finished: {len(agents_finished)}/{n_agents: <3} | runtime: {runtime: .2f} seconds | {'*' * 10}', end='')
        if to_render and iteration >= 0:
            # update curr nodes
            for a in agents:
                a.curr_node = config_to[a.name]
            # plot the iteration
            # i_agent = agents_dict['agent_0']
            # i_agent = agents_dict['agent_174']
            i_agent = agents[0]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
            }
            plot_step_in_env(ax[0], plot_info)
            plt.pause(0.001)
            # plt.pause(0.5)
            # plt.pause(1)
        # if iteration >= 0:
        #     check_vc_ec_neic_iter(agents, iteration + 1)
        iteration += 1
        if runtime > max_time:
            return None, {}

    # checks
    # for i in range(len(agents[0].path)):
    #     check_vc_ec_neic_iter(agents, i)
    runtime = time.time() - start_time
    return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': iteration}


@use_profiler(save_dir='../stats/alg_cga_mapf_pure.pstat')
def main():

    to_render = True
    # to_render = False

    params = {
        'max_time': 1000,
        'alg_name': 'CGA-PURE',
        'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        # 'alt_goal_flag': 'all',
        'to_render': to_render,
    }
    # run_mapf_alg(alg=run_cga_pure, params=params, final_render=False)
    run_mapf_alg(alg=run_cga_pure, params=params, final_render=True)


if __name__ == '__main__':
    main()




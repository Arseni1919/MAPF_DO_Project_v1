# from algs.alg_functions_pibt import *
from algs.alg_functions_cga import *
from algs.alg_functions_pibt import run_procedure_pibt
from run_single_MAPF_func import run_mapf_alg


def run_lifelong_cga(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict
) -> Tuple[None, Dict] | Tuple[Dict[str, List[Node]], Dict]:
    """
    -> LMAPF:
    - stop condition: the end of n iterations where every iteration has a time limit
    - behaviour, when agent is at its goal: agent receives a new goal
    - output: throughput
    """
    max_iter_time: int | float = params['max_iter_time']
    n_steps: int = params['n_steps']
    alg_name: bool = params['alg_name']
    to_render: bool = params['to_render']
    img_np: np.ndarray = params['img_np']
    sv_map: np.ndarray = params['sv_map']
    k_limit: bool = params['k_limit']

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    start_time = time.time()
    throughput: int = 0

    # create agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    n_agents = len(agents_dict)
    agents.sort(key=lambda a: a.priority, reverse=True)

    for step_iter in range(n_steps):

        # throughput += update_goal_nodes(agents, nodes)
        if step_iter > 0 and (step_iter - 1) % k_limit == 0:
            # update goal and throughput
            throughput += update_goal_nodes(agents, nodes)

        # PREPARATIONS
        (config_from, occupied_from, config_to, occupied_to,
         cga_step_agents_names, cga_curr_step_lists, blocked_nodes_names) = get_preparations(agents, step_iter)
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
            curr_node: Node = config_from[agent.name]
            next_node: Node = get_min_h_nei_node(curr_node, goal_node, h_dict)

            # if non_sv_nodes_np[next_node.x, next_node.y]:
            if sv_map[next_node.x, next_node.y] and not next_is_blocked(next_node, agent, config_from):
                run_procedure_pibt(
                    agent, config_from, occupied_from, config_to, occupied_to,
                    agents_dict, nodes_dict, h_dict, blocked_nodes_names,
                    step_iter, f'{agent}', with_swap=False)
                continue
            else:
                moved_agents = calc_cga_step(
                    agent, step_iter,
                    config_from, occupied_from, config_to, occupied_to,
                    agents, agents_dict, nodes, nodes_dict, last_visit_dict, h_dict, sv_map,
                    blocked_nodes_names, params, start_time, max_iter_time,
                    to_block_edges_of_goal=False,
                    # to_block_edges_of_goal=True,
                )
                for m_a in moved_agents:
                    heapq.heappush(cga_step_agents_names, m_a.name)
                cga_curr_step_lists.append(moved_agents)
                update_blocked_nodes_names_after_cga(blocked_nodes_names, moved_agents, step_iter)
                continue

        # execute the step + check the termination condition
        agents_finished, agents_unfinished = [], []
        for agent in agents:
            if agent.name in config_to:
                next_node = config_to[agent.name]
            else:
                next_node = config_from[agent.name]
                config_to[agent.name] = next_node
                occupied_to[next_node.xy_name] = agent
                agent.message += f'| [{step_iter}] stay |'
            if agent.name not in cga_step_agents_names:
                # Unplanned (e.g. blocked) or from PIBT
                agent.path.append(next_node)
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            if agent.goal_node is None:
                agent.priority = agent.init_priority
                agents_finished.append(agent)
            elif agent.curr_node != agent.goal_node:
                agent.priority += 1
                agents_unfinished.append(agent)
            else:
                agent.priority = agent.init_priority
                agents_finished.append(agent)

        # unfinished first
        # agents.sort(key=lambda a: a.priority, reverse=True)
        agents_unfinished.sort(key=lambda a: a.priority, reverse=True)
        agents = [*agents_finished, *agents_unfinished]

        # throughput += update_goal_nodes(agents, nodes)

        # print + render
        runtime = time.time() - start_time
        print(f'\r[{alg_name}] {step_iter=: <3} | runtime: {runtime: .2f} s. | {throughput=}', end='')
        # ------------------------------ #
        # ------------------------------ #
        # ------------------------------ #
        if to_render:
            # plot the iteration
            i_agent = agents[0]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
                'i': step_iter,
            }
            plot_step_in_env(ax[0], plot_info)
            plt.pause(0.001)
            # plt.pause(1)

    # checks
    # for i in range(len(agents[0].path)):
    #     check_vc_ec_neic_iter(agents, i, to_count=False)
    return {a.name: a.path for a in agents}, {'agents': agents, 'throughput': throughput}


@use_profiler(save_dir='../stats/alg_lifelong_cga.pstat')
def main():

    # to_render = True
    to_render = False

    params = {
        'max_iter_time': 5,  # seconds
        'n_steps': 200,
        'k_limit': 5,
        'alg_name': f'Lifelong_CGA_PIBT',
        # 'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        'alt_goal_flag': 'all',
        'to_render': to_render,
    }
    run_mapf_alg(alg=run_lifelong_cga, params=params, final_render=False)
    # run_mapf_alg(alg=run_lifelong_cga, params=params, final_render=True)


if __name__ == '__main__':
    main()


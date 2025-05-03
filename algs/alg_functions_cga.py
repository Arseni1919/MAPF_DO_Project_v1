import heapq
import random

from globals import *
from functions_general import *
from functions_plotting import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def next_is_blocked(next_node: Node, agent: AgentAlg, config_from: Dict[str, Node]) -> bool:
    """
    Blocked here means the corner location surrounded by walls from three sides,
    where there is no escape room for the agent in the corner.
    """
    curr_node: Node = config_from[agent.name]
    nei_nodes_names = next_node.neighbours[:]
    nei_nodes_names.remove(next_node.xy_name)  # self
    if curr_node != next_node:
        nei_nodes_names.remove(curr_node.xy_name)  # blocked
    return len(nei_nodes_names) == 0


def sort_agents(agents: List[AgentAlg]):
    agents.sort(key=lambda a: a.priority, reverse=True)


def get_blocked_nodes_names(
        agents: List[AgentAlg],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
        iteration: int
) -> List[str]:
    blocked_nodes_names: List[str] = []
    for agent in agents:
        if agent.name in config_to:
            heapq.heappush(blocked_nodes_names, config_from[agent.name].xy_name)
            heapq.heappush(blocked_nodes_names, config_to[agent.name].xy_name)
        if len(agent.path) - 1 >= iteration + 2:
            for n in agent.path[iteration + 2:]:
                if n.xy_name not in blocked_nodes_names:
                    heapq.heappush(blocked_nodes_names, n.xy_name)
    return blocked_nodes_names


def update_blocked_nodes_names_after_pibt(
        blocked_nodes_names: List[str],
        agents: List[AgentAlg],
        config_from: Dict[str, Node],
        config_to: Dict[str, Node],
) -> List[str]:
    for agent in agents:
        if agent.name in config_to and config_to[agent.name].xy_name not in blocked_nodes_names:
            heapq.heappush(blocked_nodes_names, config_from[agent.name].xy_name)
            heapq.heappush(blocked_nodes_names, config_to[agent.name].xy_name)
    return blocked_nodes_names


def update_blocked_nodes_names_after_cga(
        blocked_nodes_names: List[str],
        agents: List[AgentAlg],
        iteration: int
) -> List[str]:
    for agent in agents:
        if len(agent.path) - 1 >= iteration + 2:
            for n in agent.path[iteration + 2:]:
                if n.xy_name not in blocked_nodes_names:
                    heapq.heappush(blocked_nodes_names, n.xy_name)
    return blocked_nodes_names


def stay(agent: AgentAlg, at_node: Node, config_to: Dict[str, Node], occupied_to: Dict[str, AgentAlg]) -> None:
    config_to[agent.name] = at_node
    occupied_to[at_node.xy_name] = agent


def get_min_h_nei_node(curr_node: Node, goal_node: Node, h_dict: Dict[str, np.ndarray]) -> Node:
    nei_nodes = curr_node.neighbours_nodes[:]
    goal_h_np: np.ndarray = h_dict[goal_node.xy_name]
    min_h_nei_node = min(nei_nodes, key=lambda n: int(goal_h_np[n.x, n.y]))
    return min_h_nei_node


def build_corridor_from_nodes(
        curr_node: Node,
        goal_node: Node,
        h_dict: Dict[str, np.ndarray],
        non_sv_nodes_np: np.ndarray,
        # occupied_to: Dict[str, AgentAlg],
        # blocked_nodes_names: List[str],
) -> List[Node]:
    main_next_node = get_min_h_nei_node(curr_node, goal_node, h_dict)
    # not_in_occupied_to = main_next_node.xy_name not in occupied_to
    # not_in_blocked = main_next_node.xy_name not in blocked_nodes_names
    # if not not_in_occupied_to or not not_in_blocked:
    #     return []
    corridor: List[Node] = [curr_node, main_next_node]
    is_non_sv = non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0
    is_not_goal = main_next_node != goal_node
    # while is_non_sv and is_not_goal and not_in_occupied_to and not_in_blocked:
    while non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0 and main_next_node != goal_node:
        main_next_node = get_min_h_nei_node(main_next_node, goal_node, h_dict)
        # not_in_occupied_to = main_next_node.xy_name not in occupied_to
        # not_in_blocked = main_next_node.xy_name not in blocked_nodes_names
        # if not not_in_occupied_to or not not_in_blocked:
        #     return corridor
        corridor.append(main_next_node)
        # is_non_sv = non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0
        # is_not_goal = main_next_node != goal_node
    return corridor


def unfold_path(next_node: Node, son_to_father_dict: Dict[str, Node | None]) -> List[Node]:
    path = [next_node]
    father = son_to_father_dict[next_node.xy_name]
    while father is not None:
        path.append(father)
        father = son_to_father_dict[father.xy_name]
    path.reverse()
    return path


def find_ev_path(
        node: Node,
        corridor: List[Node],
        blocked_nodes_names: List[str],
        main_from_node: Node,
        main_goal_node: Node,
        edge_blocked_nodes_names: List[str],
        captured_free_nodes_names: List[str],
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
) -> Tuple[List[Node], Node, bool] | Tuple[None, None, bool]:
    open_list: Deque[Node] = deque([node])
    open_names: List[str] = [node.xy_name]
    closed_names: List[str] = []

    son_to_father_dict: Dict[str, Node | None] = {node.xy_name: None}
    blocked_is_involved: bool = False
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        open_names.remove(next_node.xy_name)
        if next_node not in corridor and next_node.xy_name not in captured_free_nodes_names and next_node.xy_name not in occupied_from:
            ev_path = unfold_path(next_node, son_to_father_dict)
            return ev_path, next_node, blocked_is_involved

        for nei_node in next_node.neighbours_nodes:
            nei_xy_name = nei_node.xy_name
            # self ref
            if nei_xy_name == next_node.xy_name:
                continue
            if nei_xy_name in closed_names:
                continue
            if nei_xy_name in open_names:
                continue
            if next_node.xy_name == main_goal_node.xy_name and nei_xy_name in edge_blocked_nodes_names:
                continue
            if nei_xy_name == main_from_node.xy_name:
                continue
            if nei_node.xy_name in blocked_nodes_names:
                blocked_is_involved = True
                continue
            if nei_node.xy_name in occupied_to:
                blocked_is_involved = True
                continue
            open_list.append(nei_node)
            heapq.heappush(open_names, nei_node.xy_name)
            son_to_father_dict[nei_node.xy_name] = next_node
        heapq.heappush(closed_names, next_node.xy_name)

    return None, None, blocked_is_involved


def get_alt_goal_node(
        node: Node,
        occupied_from: Dict[str, AgentAlg],
        non_sv_nodes_np: np.ndarray,
        agents: List[AgentAlg],
) -> Node:
    """
    Switch to alt goal (non-SV node that is also not a goal of anyone)
    """
    other_goals: List[Node] = [agent.get_goal_node() for agent in agents]
    open_list: Deque[Node] = deque([node])
    closed_names_list_heap = []
    possible_nodes: List[Node] = []
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        is_non_sv = non_sv_nodes_np[next_node.x, next_node.y] == 1
        is_not_in_other_goals = next_node not in other_goals
        is_not_occupied = next_node.xy_name not in occupied_from
        is_rand_positive = random.random() > 0.8
        if is_non_sv and is_not_in_other_goals and is_not_occupied:
            # return next_node
            possible_nodes.append(next_node)
            if is_rand_positive:
                return random.choice(possible_nodes)
        for nei_node in next_node.neighbours_nodes:
            # self ref
            if nei_node == next_node:
                continue
            if nei_node.xy_name in closed_names_list_heap:
                continue
            open_list.append(nei_node)
        heapq.heappush(closed_names_list_heap, next_node.xy_name)
    if len(possible_nodes) > 0:
        return random.choice(possible_nodes)
    raise RuntimeError('nope')


def update_last_visit_dict(last_visit_dict: Dict[str, int], given_agents: List[AgentAlg]) -> None:
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path):
            last_visit_dict[n.xy_name] = max(i_n, last_visit_dict[n.xy_name])


def push_ev_agents(
        ev_path: List[Node],
        ev_config_from: Dict[str, Node],
        ev_occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
        iteration: int,
        main_agent: AgentAlg,
        last_visit_dict,
) -> Tuple[int, List[AgentAlg]]:
    ev_chain_dict: Dict[str, Node | None] = {}
    max_i = len(ev_path)
    for i, n in enumerate(ev_path):
        if i + 1 == max_i:
            ev_chain_dict[n.xy_name] = None
            break
        ev_chain_dict[n.xy_name] = ev_path[i + 1]
    assert len(ev_chain_dict) == max_i

    agents_to_assign: List[AgentAlg] = []
    locations_to_assign: List[Node] = []
    for n in ev_path:
        if n.xy_name in ev_occupied_from:
            i_agent: AgentAlg = ev_occupied_from[n.xy_name]
            assert i_agent != main_agent
            agents_to_assign.append(i_agent)
            # assert i_agent.name not in config_to
            locations_to_assign.append(n)
    locations_to_assign = locations_to_assign[1:]
    locations_to_assign.append(ev_path[-1])
    agents_to_assign.reverse()
    locations_to_assign.reverse()

    for a, final_n in zip(agents_to_assign, locations_to_assign):
        new_path: List[Node] = []
        curr_node: Node = ev_config_from[a.name]
        while curr_node != final_n:
            next_node: Node = ev_chain_dict[curr_node.xy_name]
            next_n_last_visit = last_visit_dict[next_node.xy_name]
            while len(a.path) + len(new_path) <= next_n_last_visit:
                new_path.append(curr_node)
            new_path.append(next_node)
            curr_node = next_node
        # assert a.path[-1].xy_name in new_path[0].neighbours

        # path change
        a.path.extend(new_path)

        update_last_visit_dict(last_visit_dict, [a])
    max_len = max([len(a.path) for a in agents_to_assign])
    return max_len, agents_to_assign


def get_last_visit_dict(given_list: List[Node], given_agents: List[AgentAlg], iteration: int) -> Dict[str, int]:
    last_visit_dict = {n.xy_name: 0 for n in given_list}
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path[iteration:]):
            if n in given_list:
                last_visit_dict[n.xy_name] = max(i_n + iteration, last_visit_dict[n.xy_name])
    return last_visit_dict


def push_main_agent(
        main_agent: AgentAlg,
        corridor: List[Node],
        moved_agents: List[AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
        iteration: int
) -> None:
    assert main_agent not in moved_agents
    assert len(main_agent.path) - 1 == iteration
    last_visit_dict = get_last_visit_dict(corridor, moved_agents, iteration)
    assert corridor[0] == main_agent.path[-1]
    prev_n = corridor[0]
    for c_n in corridor[1:]:
        c_n_last_visit = last_visit_dict[c_n.xy_name]
        while len(main_agent.path) <= c_n_last_visit:
            main_agent.path.append(prev_n)
        main_agent.path.append(c_n)
        prev_n = c_n

    next_step: Node = main_agent.path[iteration + 1]
    config_to[main_agent.name] = next_step
    occupied_to[next_step.xy_name] = main_agent
    main_agent.message += f'| [{iteration}] main |'


def get_preparations(agents: List[AgentAlg], iteration: int):
    config_from: Dict[str, Node] = {}
    occupied_from: Dict[str, AgentAlg] = {}
    config_to: Dict[str, Node] = {}
    occupied_to: Dict[str, AgentAlg] = {}
    cga_step_agents_names: List[str] = []
    cga_curr_step_lists: List[List[AgentAlg]] = []
    blocked_nodes_names: List[str] = []
    for agent in agents:
        config_from[agent.name] = agent.path[iteration]
        occupied_from[agent.path[iteration].xy_name] = agent
        # Update config_to with agents that have planned future steps
        if len(agent.path) - 1 >= iteration + 1:
            next_node: Node = agent.path[iteration + 1]
            config_to[agent.name] = next_node
            occupied_to[next_node.xy_name] = agent
            heapq.heappush(cga_step_agents_names, agent.name)
            heapq.heappush(blocked_nodes_names, config_from[agent.name].xy_name)
            heapq.heappush(blocked_nodes_names, config_to[agent.name].xy_name)
            for n in agent.path[iteration + 2:]:
                if n.xy_name not in blocked_nodes_names:
                    heapq.heappush(blocked_nodes_names, n.xy_name)
    return config_from, occupied_from, config_to, occupied_to, cga_step_agents_names, cga_curr_step_lists, blocked_nodes_names


def calc_cga_step(
        main_agent: AgentAlg,
        iteration: int,
        config_from: Dict[str, Node],
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
        agents: List[AgentAlg],
        agents_dict: Dict[str, AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        last_visit_dict: Dict[str, int],
        h_dict: Dict[str, np.ndarray],
        non_sv_nodes_np: np.ndarray,
        blocked_nodes_names: List[str],
        params: dict,
        start_time: float, max_time: int | float,
        to_block_edges_of_goal: bool = True
) -> List[AgentAlg]:
    """
    v - Build corridor
    v - Find EV paths (with the goal node blocked around except the corridor edge)
    v - If there are no EV:
    v         - Because it is impossible:
    v             - Switch to alt goal (non-SV node that is also not a goal of anyone)
    v     - Return
    v - Evacuate ev-agents
    v - Build the steps in the corridor to the main agent
    v - Return
    """
    main_from_node: Node = config_from[main_agent.name]
    if main_from_node == main_agent.get_goal_node():
        return []

    # Build corridor
    corridor: List[Node] = build_corridor_from_nodes(
        config_from[main_agent.name], main_agent.get_goal_node(), h_dict, non_sv_nodes_np,
        # occupied_to, blocked_nodes_names
    )
    if not to_block_edges_of_goal and len(corridor) >= 1:
        new_corridor: List[Node] = []
        for n in corridor:
            if n.xy_name in occupied_to or n.xy_name in blocked_nodes_names:
                break
            new_corridor.append(n)
        if len(new_corridor) <= 1:
            return []
        corridor = new_corridor

    # if corridor is occupied somewhere
    for n in corridor:
        if n.xy_name in occupied_to or n.xy_name in blocked_nodes_names:
            return []

    # Find ev-agents
    ev_agents: List[AgentAlg] = [occupied_from[node.xy_name] for node in corridor[1:] if node.xy_name in occupied_from]
    # if any agent from ev_agents is already with a plan
    for agent in ev_agents:
        if agent.name in config_to:
            return []

    # Find EV paths (with the goal node blocked around except the corridor edge)
    ev_paths_list: List[List[Node]] = []
    captured_free_nodes_names: List[str] = []
    main_goal_node: Node = main_agent.get_goal_node()

    # get edge_blocked_nodes
    edge_blocked_nodes_names: List[str] = []
    if to_block_edges_of_goal:
        if main_goal_node == corridor[-1]:
            # edge_blocked_nodes.append(main_goal_node)
            for n in main_goal_node.neighbours_nodes:
                if n not in corridor:
                    heapq.heappush(edge_blocked_nodes_names, n.xy_name)

    for ev_agent in ev_agents:
        ev_path, captured_free_node, blocked_is_involved = find_ev_path(
            config_from[ev_agent.name], corridor,
            blocked_nodes_names, main_from_node, main_goal_node, edge_blocked_nodes_names,
            captured_free_nodes_names,
            occupied_from, config_to, occupied_to
        )
        if ev_path is None:
            if params['alt_goal_flag'] == 'first':
                if not blocked_is_involved and main_agent.name == agents[0].name:
                    # print(f'\n{main_agent} (order: {agents.index(main_agent)}) got alt goal')
                    main_agent.alt_goal_node = get_alt_goal_node(
                        config_from[main_agent.name], occupied_from, non_sv_nodes_np, agents,
                    )
            elif params['alt_goal_flag'] == 'num':
                if not blocked_is_involved and main_agent in agents[:params['alt_goal_num']]:
                    # print(f'\n{main_agent} (order: {agents.index(main_agent)}) got alt goal')
                    main_agent.alt_goal_node = get_alt_goal_node(
                        config_from[main_agent.name], occupied_from, non_sv_nodes_np, agents,
                    )
            elif params['alt_goal_flag'] == 'all':
                if not blocked_is_involved:
                    # print(f'\n{main_agent} (order: {agents.index(main_agent)}) got alt goal')
                    main_agent.alt_goal_node = get_alt_goal_node(
                        config_from[main_agent.name], occupied_from, non_sv_nodes_np, agents,
                    )
            else:
                raise RuntimeError('nope!')
            return []

        heapq.heappush(captured_free_nodes_names, captured_free_node.xy_name)
        ev_paths_list.append(ev_path)

    # Evacuate ev-agents
    moved_agents = []
    # last_visit_dict: Dict[str, int] = {n.xy_name: 0 for n in nodes}
    ev_config_from: Dict[str, Node] = {}
    ev_occupied_from: Dict[str, AgentAlg] = {}
    for a in agents:
        ev_config_from[a.name] = a.path[-1]
        ev_occupied_from[a.path[-1].xy_name] = a
    for i_ev, ev_path in enumerate(ev_paths_list):
        max_len, assigned_agents = push_ev_agents(ev_path,
                                                  ev_config_from, ev_occupied_from, config_to, occupied_to, iteration,
                                                  main_agent, last_visit_dict)
        # assert main_agent not in assigned_agents
        for a in assigned_agents:
            next_step: Node = a.path[iteration + 1]
            config_to[a.name] = next_step
            occupied_to[next_step.xy_name] = a
            a.message += f'| [{iteration}] under: {main_agent.name} |'
            del ev_occupied_from[ev_config_from[a.name].xy_name]
            ev_config_from[a.name] = a.path[-1]
            ev_occupied_from[a.path[-1].xy_name] = a
        moved_agents.extend(assigned_agents)
        moved_agents = list(set(moved_agents))

    # # Build the steps in the corridor to the main agent + extend the path
    push_main_agent(main_agent, corridor, moved_agents, config_to, occupied_to, iteration)
    moved_agents.append(main_agent)

    return moved_agents


# config_to[main_agent.name] = main_from_node
# occupied_to[main_from_node.xy_name] = main_agent
# main_agent.path.append(main_from_node)
# return [main_agent]


# moved_agents = []
# for agent in ev_agents:
#     from_node = config_from[agent.name]
#     config_to[agent.name] = from_node
#     occupied_to[from_node.xy_name] = agent
#     agent.path.append(from_node)
#     moved_agents.append(agent)
# config_to[main_agent.name] = main_from_node
# occupied_to[main_from_node.xy_name] = main_agent
# main_agent.path.append(main_from_node)
# moved_agents.append(main_agent)


# def update_blocked_nodes_names_after_cga(
#         blocked_nodes_names: List[str],
#         agents: List[AgentAlg],
#         config_from: Dict[str, Node],
#         config_to: Dict[str, Node],
#         iteration: int
# ) -> List[str]:
#     for agent in agents:
#         if agent.name in config_to and config_to[agent.name].xy_name not in blocked_nodes_names:
#             heapq.heappush(blocked_nodes_names, config_from[agent.name].xy_name)
#             heapq.heappush(blocked_nodes_names, config_to[agent.name].xy_name)
#         if len(agent.path) - 1 >= iteration + 2:
#             for n in agent.path[iteration + 2:]:
#                 if n.xy_name not in blocked_nodes_names:
#                     heapq.heappush(blocked_nodes_names, n.xy_name)
#     return blocked_nodes_names
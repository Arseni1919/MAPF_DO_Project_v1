from globals import AgentAlg, Node, Dict, List, Deque, deque, Tuple
import numpy as np
from typing import Optional


class AgentOD(AgentAlg):
    def __init__(self, num: int, start_node: Node, goal_node: Optional[Node] = None):
        super().__init__(num, start_node, goal_node)
        self.pre_move_position: Node = start_node
        self.post_move_position: Node = start_node
        self.assigned_move: bool = False
        self.order: int = num
        self.heuristic: int = 0

    def clone(self):
        new_agent = AgentOD(
            num=self.num,
            start_node=self.start_node,
            goal_node=self.goal_node
        )
        new_agent.pre_move_position = self.pre_move_position
        new_agent.post_move_position = self.post_move_position
        new_agent.assigned_move = self.assigned_move
        new_agent.order = self.num
        new_agent.heuristic = self.heuristic
        new_agent.name = self.name
        new_agent.start_node_name = self.start_node_name  # ?
        new_agent.curr_node = self.curr_node
        new_agent.curr_node_name = self.curr_node.xy_name  # ?
        if self.goal_node is not None:
            new_agent.goal_node_name = self.goal_node.xy_name
        new_agent.path = self.path
        new_agent.k_path = self.k_path
        return new_agent


class ConfigNode:
    def __init__(self, config: Dict[str, Node],
                 config_agents: List[AgentOD],
                 g: int, h: int):
        self.config = config              # dict: {agent_name → Node}
        self.agents: List[AgentOD] = config_agents  # a list of AgentODs
        self.parent = None
        self.g = g
        self.h = h
        self.f = g + h
        self.name = ''
        self.update_name()  # optional: string identifier, might not need this
        self.is_intermediate: bool = False
        self.chosen_agent: str | None = None
        self.chose_to_stay: bool = False
        self.moved_yet: List = []

    def __lt__(self, other):              # required for heapq
        return self.f < other.f

    def __repr__(self):
        agent_positions = ', '.join(f"{agent}:{node.xy_name}" for agent, node in sorted(self.config.items()))
        return (f"ConfigNode(f={self.f}, g={self.g}, h={self.h}, "
                f"agents=[{agent_positions}], "
                f"is_intermediate={self.is_intermediate})")

    def get_name(self):
        return '_'.join([f"{a}:{n.xy_name}" for a, n in sorted(self.config.items())])

    def update_name(self):
        self.name = self.get_config_name(self.config)
        # self.name = '_'.join(f"{agent}:{self.config[agent].xy_name}" for agent in sorted(self.config))

    def get_config_name(self, config: Dict[str, Node]):
        assert len(config) > 0
        k_list = list(config.keys())
        k_list.sort()
        name = ''
        for k in k_list:
            v = config[k]
            name += v.xy_name + '-'
        return name[:-1]


def goal_test(config_node: ConfigNode) -> bool:
    for agent in config_node.agents:
        if agent.goal_node:
            if config_node.config[agent.name] != agent.goal_node:
                return False
    return True


def create_agents_od(
        start_nodes: List[Node], goal_nodes: List[Node],  h_dict: Dict[str, np.ndarray]
) -> Tuple[List[AgentOD], Dict[str, AgentOD]]:
    agents: List[AgentOD] = []
    agents_dict: Dict[str, AgentOD] = {}
    goals_len = len(goal_nodes)
    start_nodes_with_goals = start_nodes[0:goals_len]
    num = 0
    for s_node, g_node in zip(start_nodes_with_goals, goal_nodes):
        new_agent = AgentOD(num, s_node, g_node)
        new_agent.pre_move_position = s_node
        new_agent.post_move_position = None
        new_agent.order = num
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
        num += 1
    start_nodes_without_goals = start_nodes[goals_len:]
    for s_node in start_nodes_without_goals:
        new_agent = AgentOD(num, s_node)
        new_agent.pre_move_position = s_node
        new_agent.post_move_position = None
        new_agent.order = num
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
        new_agent.heuristic = 0
        num += 1
    return agents, agents_dict


def backtrack_od(n: ConfigNode) -> Dict[str, List[Node]]:
    paths_deque_dict: Dict[str, Deque[Node]] = {k: deque([]) for k, _ in n.config.items()}
    parent: ConfigNode = n
    while parent is not None:
        for k, v in parent.config.items():
            if k == parent.chosen_agent:
                paths_deque_dict[k].appendleft(v)
        parent = parent.parent

    # add the first node for each agent
    for i, agent in enumerate(paths_deque_dict):
        paths_deque_dict[agent].appendleft(n.agents[i].start_node)

    paths_dict: Dict[str, List[Node]] = {}
    for k, v in paths_deque_dict.items():
        paths_dict[k] = list(v)

    return paths_dict


def get_next_config(agent: AgentOD,
                    curr_config: ConfigNode,
                    neighbour: Node,
                    h_dict: Dict[str, np.ndarray],
                    intermediate: bool,
                    idx: int = 0,
                    cost_func: str = 'makespan',
                    weights=None) -> ConfigNode:

    chose_to_stay = False
    if agent.curr_node == neighbour:
        chose_to_stay = True

    agent_copy = AgentOD(
        num=agent.num,
        start_node=agent.start_node,
        goal_node=agent.goal_node
    )
    agent_copy.pre_move_position = agent.pre_move_position
    agent_copy.curr_node = neighbour
    agent_copy.post_move_position = neighbour
    agent_copy.assigned_move = True
    agent_copy.heuristic = agent.heuristic


    config_copy = dict(curr_config.config)
    config_copy[agent_copy.name] = agent_copy.curr_node

    agents_copy = [agent.clone() for agent in curr_config.agents]

    new_config_node_copy = ConfigNode(config=config_copy,
                                      config_agents=agents_copy,
                                      g=0,
                                      h=0)
    new_config_node_copy.chosen_agent = agent_copy.name  # ?
    new_config_node_copy.parent = curr_config
    new_config_node_copy.is_intermediate = True
    new_config_node_copy.agents[idx] = agent_copy  # ?
    new_config_node_copy.config[agent_copy.name] = neighbour
    new_config_node_copy.update_name()
    new_config_node_copy.chose_to_stay = chose_to_stay
    new_config_node_copy.moved_yet = [agent_name for agent_name in curr_config.moved_yet]

    if not chose_to_stay:
        if agent_copy.name not in new_config_node_copy.moved_yet:
            new_config_node_copy.moved_yet.append(agent_copy.name)

    # calculate the new h,g,f values
    new_config_node_copy.h = config_h(new_config_node_copy, cost_func, weights)
    new_config_node_copy.g = curr_config.g
    new_config_node_copy.f = new_config_node_copy.g + new_config_node_copy.h

    # standard nodes initiate a new step, therefore have plus 1 for g and f
    if cost_func == 'makespan':
        if not intermediate:
            new_config_node_copy.g += 1
            new_config_node_copy.f += 1

    elif cost_func == 'fuel':
        if not new_config_node_copy.chose_to_stay:
            new_config_node_copy.g += 1
            new_config_node_copy.f += 1

    elif cost_func == 'moved_agents':
        new_config_node_copy.g = get_moved_agents(new_config_node_copy)
        new_config_node_copy.f = new_config_node_copy.g + new_config_node_copy.h

    elif cost_func == 'average':
        g_moved_agents = get_moved_agents(new_config_node_copy) * weights['moved_agents'] / sum(weights.values())

        if not new_config_node_copy.chose_to_stay:
            g_fuel = 1 * weights['fuel'] / sum(weights.values())
        else:
            g_fuel = 0
        if not intermediate:
            g_makespan = 1 * weights['makespan'] / sum(weights.values())
        else:
            g_makespan = 0

        new_config_node_copy.g += (g_moved_agents + g_fuel + g_makespan)
        new_config_node_copy.f = new_config_node_copy.g + new_config_node_copy.h

    # when finished with the last intermediate node, reset for the next standard node
    if idx == len(curr_config.agents) - 1:
        new_config_node_copy.is_intermediate = False
        for moved_agent in new_config_node_copy.agents:
            moved_agent.pre_move_position = moved_agent.curr_node
            moved_agent.post_move_position = None
            moved_agent.assigned_move = False

    return new_config_node_copy
    # return new_config_node

# 460 iterations per min without deep copies


def neighbour_taken(idx: int,
                    curr_agent: AgentOD,
                    neighbour: Node,
                    curr_config: ConfigNode) -> bool:
    for i, agent in enumerate(curr_config.agents):
        if i >= idx:
            return False
        if agent.post_move_position == neighbour:
            return True
        if curr_agent.pre_move_position == agent.post_move_position and agent.pre_move_position == neighbour:
            return True


def config_h(config: ConfigNode, cost_func: str, weights: dict) -> int:
    h_makespan = 0
    h_fuel = 0
    h_moved_agents = 0
    for i, agent in enumerate(config.agents):
        if agent.heuristic is not None:

            h_fuel += agent.heuristic

            if agent.heuristic > h_makespan:
                h_makespan = agent.heuristic

            if agent.goal_node:
                if agent.curr_node != agent.goal_node:
                    h_moved_agents += 1

    if cost_func == 'moved_agents':
        h_moved_agents - len(config.moved_yet) # TODO Try to fix this heuristic
        return h_moved_agents
    elif cost_func == 'makespan':
        return h_makespan
    elif cost_func == 'fuel':
        return h_fuel
    elif cost_func == 'average':
        w_moved_agents = weights['moved_agents'] * h_moved_agents
        w_fuel = weights['fuel'] * h_fuel
        w_makespan = weights['makespan'] * h_makespan
        return (w_moved_agents + w_fuel + w_makespan) / sum(weights.values())


def get_config_name(config: Dict[str, Node]):
    # same from lacam to save imports
    assert len(config) > 0
    k_list = list(config.keys())
    k_list.sort()
    name = ''
    for k in k_list:
        v = config[k]
        if v is not None:
            name += v.xy_name + '-'
    return name[:-1]


def get_fuel(agents: List[AgentOD]) -> int:
    fuel = 0
    for agent in agents:
        curr_step = agent.path[0]
        for step in agent.path:
            if step != curr_step:
                fuel += 1
                curr_step = step
    return fuel


def get_moved_agents(config: ConfigNode) -> int:
    return len(config.moved_yet)

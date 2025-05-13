from globals import *
from functions_general import *
from functions_plotting import *
import copy


class AgentOD(AgentAlg):
    def __init__(self, num: int, start_node: Node, goal_node: Node | None = None):
        super().__init__(num, start_node, goal_node)

        self.pre_move_position: Node = start_node
        self.post_move_position: Node = start_node
        self.assigned_move: bool = False
        self.order: int = num
        self.heuristic: int = 0


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
            if agent.goal_node != agent.curr_node:
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
    paths_deque_dict: Dict[str, Deque[Node]] = {k: deque([v]) for k, v in n.config.items()}
    parent: ConfigNode = n.parent
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
                    idx: int = 0) -> ConfigNode:
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

    # make a copy of the first agent (only him moves in standard nodes)
    # agent_copy = copy.deepcopy(agent)
    #
    # # assign a move
    # agent_copy.curr_node = neighbour
    # agent_copy.post_move_position = neighbour
    # agent_copy.assigned_move = True

    # create a new config
    config_copy = copy.deepcopy(curr_config.config)
    agents_copy = copy.deepcopy(curr_config.agents)
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

    # calculate the new h,g,f values
    new_config_node_copy.h = config_h(new_config_node_copy)
    new_config_node_copy.g = curr_config.g
    new_config_node_copy.f = new_config_node_copy.g + new_config_node_copy.h

    # standard nodes initiate a new step, therefore have plus 1 for g and f
    if not intermediate:
        new_config_node_copy.g += 1
        new_config_node_copy.f += 1

    # new_config_node = copy.deepcopy(curr_config)
    # new_config_node.chosen_agent = agent_copy.name  # which agent just moved?
    # new_config_node.parent = curr_config  # who is the parent configNode?
    # new_config_node.is_intermediate = True  # is the node intermediate or standard?
    #
    # # replace the old agent with the new one
    # new_config_node.agents[idx] = agent_copy
    # new_config_node.config[agent_copy.name] = neighbour
    # new_config_node.update_name()
    # new_config_node.chose_to_stay = chose_to_stay
    #
    # # calculate the new h,g,f values
    # new_config_node.h = config_h(new_config_node)
    # new_config_node.g = curr_config.g
    # new_config_node.f = new_config_node.g + new_config_node.h
    #
    # # standard nodes initiate a new step, therefore have plus 1 for g and f
    # if not intermediate:
    #     new_config_node.g += 1
    #     new_config_node.f += 1

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



def config_h(config: ConfigNode) -> int:
    h = 0
    for i, agent in enumerate(config.agents):
        if agent.heuristic is not None:
           h += agent.heuristic
#             if agent.heuristic > h:
#                 h = agent.heuristic
    return h

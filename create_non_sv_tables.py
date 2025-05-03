from globals import *
from functions_general import *
from functions_plotting import *


def get_blocked_non_sv_nodes(img_dir: str, folder_dir: str = 'logs_for_freedom_maps'):
    possible_dir = f'{folder_dir}/blocked_{img_dir[:-4]}.npy'
    # assert os.path.exists(possible_dir)
    with open(possible_dir, 'a') as f:
        non_sv_nodes_with_blocked_np = np.load(f)
        return non_sv_nodes_with_blocked_np


def is_non_sv(node: Node, nodes_dict: Dict[str, Node], blocked_node: Node | None = None) -> bool:

    blocked_node_name = blocked_node.xy_name

    assert len(node.neighbours) != 0
    assert len(node.neighbours) != 1
    if len(node.neighbours) == 2:
        return True

    init_nei_names = node.neighbours[:]
    init_nei_names.remove(node.xy_name)

    if blocked_node_name in init_nei_names:
        return False

    first_nei_name = init_nei_names.pop(0)
    first_nei = nodes_dict[first_nei_name]

    open_list: Deque[Node] = deque([first_nei])
    open_names_list_heap = [f'{first_nei.xy_name}']
    closed_names_list_heap = [f'{node.xy_name}']

    iteration = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.pop()
        open_names_list_heap.remove(next_node.xy_name)
        if next_node.xy_name in init_nei_names:
            init_nei_names.remove(next_node.xy_name)
            if len(init_nei_names) == 0:
                return True
        for nei_name in next_node.neighbours:

            if nei_name == next_node.xy_name:
                continue

            if nei_name in closed_names_list_heap:
                continue

            if nei_name in open_names_list_heap:
                continue

            if nei_name == blocked_node_name:
                continue

            nei_node = nodes_dict[nei_name]
            open_list.appendleft(nei_node)
            heapq.heappush(open_names_list_heap, nei_name)
        heapq.heappush(closed_names_list_heap, next_node.xy_name)

    return False


def create_non_sv_nodes_with_blocked_np(nodes: List[Node], nodes_dict: Dict[str, Node], img_np: np.ndarray, img_dir: str):
    # x, y, x, y
    print(f'Started to create blocked_{img_dir[:-4]}.npy...')
    non_sv_nodes_with_blocked_np: np.ndarray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[0], img_np.shape[1]))
    for main_node in nodes:

        print(f'{main_node.xy_name} started...', end='')
        non_sv_nodes_np = np.zeros(img_np.shape)
        for i_node in nodes:
            if is_non_sv(i_node, nodes_dict, blocked_node=main_node):
                non_sv_nodes_np[i_node.x, i_node.y] = 1
        non_sv_nodes_with_blocked_np[main_node.x, main_node.y, :, :] = non_sv_nodes_np
        example = img_np + non_sv_nodes_np * 0.5
        print(f'finished.')

    assert os.path.exists('logs_for_freedom_maps')

    possible_dir = f'logs_for_freedom_maps/blocked_v2_{img_dir[:-4]}.npy'

    with open(possible_dir, 'wb') as f:
        np.save(f, non_sv_nodes_with_blocked_np)
        print(f'Saved freedom nodes of {img_dir} (with blocked options) to {possible_dir}.')


def main():

    img_dir = '10_10_my_corridor.map'
    # img_dir = '10_10_my_rand.map'
    # img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    # img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'

    # img_dir = 'empty-32-32.map'  # v
    # img_dir = 'random-32-32-10.map'  # v
    # img_dir = 'random-32-32-20.map'  # v
    # img_dir = 'maze-32-32-4.map'  # v
    # img_dir = 'maze-32-32-2.map'  # v
    # img_dir = 'room-32-32-4.map'  # v

    # img_dir = 'maze-128-128-1.map'

    # img_dir = 'random-64-64-10.map'
    # img_dir = 'random-64-64-20.map'
    # img_dir = 'room-64-64-8.map'
    # img_dir = 'den312d.map'


    path_to_maps: str = 'maps'
    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    create_non_sv_nodes_with_blocked_np(nodes, nodes_dict, img_np, img_dir=img_dir)

    non_sv_nodes_with_blocked_np = get_blocked_non_sv_nodes(img_dir=img_dir)
    print()


if __name__ == '__main__':
    main()


import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import concurrent.futures
import threading
import asyncio
import logging
from typing import *
from globals import *
from functions_general import *
from functions_plotting import *

# from simulator_objects import ListNodes
# from funcs_graph.map_dimensions import map_dimensions_dict
# from funcs_graph.nodes_from_pic import build_graph_nodes, get_dims_from_pic
# from funcs_plotter.plotter import Plotter


def dist_heuristic(from_node, to_node):
    return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
    # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)


def h_func_creator(h_dict):
    def h_func(from_node, to_node):
        if to_node.xy_name in h_dict:
            h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
            if h_value > 0:
                return h_value
        return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
        # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)
    return h_func


def get_node(successor_xy_name, node_current, nodes_dict):
    if node_current.xy_name == successor_xy_name:
        return None
    return nodes_dict[successor_xy_name]
    # for node in nodes:
    #     if node.xy_name == successor_xy_name and node_current.xy_name != successor_xy_name:
    #         return node
    # return None


def parallel_update_h_table(node, nodes, map_dim, to_save, plotter, middle_plot, h_dict, node_index):
    print(f'[HEURISTIC]: Thread {node_index} started.')
    h_table = build_heuristic_for_one_target(node, nodes, map_dim, to_save, plotter, middle_plot)
    h_dict[node.xy_name] = h_table
    print(f'[HEURISTIC]: Thread {node_index} finished.')


def parallel_build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    print('Started to build heuristic...')
    h_dict = {}
    reset_nodes(nodes, target_nodes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_nodes)) as executor:
        for node_index, node in enumerate(target_nodes):
            executor.submit(parallel_update_h_table, node, nodes, map_dim, to_save, plotter, middle_plot, h_dict, node_index)

    print(f'\nFinished to build heuristic for all nodes.')
    return h_dict


def load_h_dict(possible_dir: str) -> Dict[str, np.ndarray] | None:
    if os.path.exists(possible_dir):
        # Opening JSON file
        with open(possible_dir, 'r') as openfile:
            # Reading from json file
            h_dict = json.load(openfile)
            for k, v in h_dict.items():
                h_dict[k] = np.array(v)
            return h_dict
    return None


def save_h_dict(h_dict, possible_dir):
    for k, v in h_dict.items():
        h_dict[k] = v.tolist()
    json_object = json.dumps(h_dict, indent=2)
    Path(possible_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(possible_dir, "w") as outfile:
        outfile.write(json_object)
        print(f"Saving to: {os.path.abspath(possible_dir)}")


def parallel_build_heuristic_for_entire_map(nodes: List, nodes_dict: Dict[str, Any], map_dim: Tuple[int, int], **kwargs) -> Dict[str, np.ndarray]:
    # print(f'Started to build heuristic for {kwargs['img_dir'][:-4]}...')
    path = kwargs['path']
    possible_dir = f"{path}/h_dict_of_{kwargs['img_dir'][:-4]}.json"

    # if there is one
    h_dict = load_h_dict(possible_dir)
    if h_dict is not None:
        # print(f'\nFinished to build heuristic for all nodes.')
        return h_dict

    # else, create one
    h_dict = {}
    reset_nodes(nodes, nodes)
    # for node_index, node in enumerate(nodes):
    #     h_table = build_heuristic_for_one_target(node, nodes, map_dim, False, None, False)
    #     h_dict[node.xy_name] = h_table
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        for node_index, node in enumerate(nodes):
            # parallel_update_h_table(node, nodes, map_dim, to_save, plotter, middle_plot, h_dict, node_index)
            executor.submit(parallel_update_h_table, node, nodes, map_dim, False, None, False, h_dict, node_index)
    save_h_dict(h_dict, possible_dir)
    # print(f'\nFinished to build heuristic for all nodes.')

    h_dict = load_h_dict(possible_dir)
    if h_dict is not None:
        print(f'\nFinished to build heuristic for all nodes.')
        return h_dict

    raise RuntimeError('nu nu')


def build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    print('Started to build heuristic...')
    h_dict = {}
    # reset_nodes(nodes, target_nodes)
    # check what reset does and if is needed
    iteration = 0
    for node in target_nodes:
        h_table = build_heuristic_for_one_target(node, nodes, map_dim, to_save, plotter, middle_plot)
        h_dict[node.xy_name] = h_table

        print(f'\nFinished to build heuristic for node {iteration}.')
        iteration += 1
    return h_dict


def reset_nodes(nodes, target_nodes=None):
    _ = [node.reset(target_nodes) for node in nodes]


def build_heuristic_for_one_target(target_node, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    # print('Started to build heuristic...')
    h_table = np.zeros(map_dim)
    g_dict = {target_node.xy_name: 0}
    copy_nodes = nodes
    nodes_dict = {node.xy_name: node for node in copy_nodes}
    target_name = target_node.xy_name
    target_node = nodes_dict[target_name]
    # target_node = [node for node in copy_nodes if node.xy_name == target_node.xy_name][0]
    open_list = []
    close_list = []
    # open_nodes = List(target_name=target_node.xy_name)
    # closed_nodes = List(target_name=target_node.xy_name)
    open_list.append(target_node)
    #open_nodes.add(target_node)
    iteration = 0
    while len(open_list) > 0:
    # while len(open_nodes) > 0:
        iteration += 1
        node_current = open_list[0]
        open_list.remove(node_current)
        # node_current = get_node_from_open(open_list, target_name)
        # node_current = open_nodes.pop()
        # if node_current.xy_name == '30_12':
        #     print()
        for successor_xy_name in node_current.neighbours:
            node_successor = get_node(successor_xy_name, node_current, nodes_dict)
            if node_successor:
                successor_current_g = g_dict[node_current.xy_name] + 1  # h(now, next)

                # INSIDE OPEN LIST
                if node_successor in open_list:
                    if g_dict[node_successor.xy_name] <= successor_current_g:
                        continue
                    open_list.remove(node_successor)
                    g_dict[node_successor.xy_name] = successor_current_g
                    node_successor.parent = node_current
                    open_list.append(node_successor)

                # INSIDE CLOSED LIST
                elif node_successor in close_list:
                    if g_dict[node_successor.xy_name] <= successor_current_g:
                        continue
                    close_list.remove(node_successor)
                    g_dict[node_successor.xy_name] = successor_current_g
                    node_successor.parent = node_current
                    open_list.append(node_successor)

                # NOT IN CLOSED AND NOT IN OPEN LISTS
                else:
                    g_dict[node_successor.xy_name] = successor_current_g
                    node_successor.parent = node_current
                    open_list.append(node_successor)

                # node_successor.g_dict[target_name] = successor_current_g
                # node_successor.parent = node_current

        # open_nodes.remove(node_current, target_name=target_node.xy_name)
        close_list.append(node_current)
        # closed_nodes.add(node_current)

        # if plotter and middle_plot and iteration % 1000 == 0:
        #     plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
        #                        closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)
        if iteration % 100 == 0:
            print(f'\riter: {iteration}', end='')

    # if plotter and middle_plot:
    #     plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
    #                        closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)

    for node in copy_nodes:
        h_table[node.x, node.y] = g_dict[node.xy_name]
    # h_dict = {target_node.xy_name: h_table}
    # print(f'\rFinished to build heuristic at iter {iteration}.')
    return h_table


def main():
    # img_dir = 'den520d.png'
    path_to_maps: str = 'maps'
    img_dir = '3_3_empty.map'
    map_dim = get_dims_from_pic(img_dir=img_dir, path=path_to_maps)
    # build_graph_from_np
    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    # x_goal, y_goal = 38, 89
    # node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    # plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    # reset_nodes(nodes, target_nodes=[node_goal])
    # h_table = build_heuristic_for_one_target(node_goal, nodes, map_dim, plotter=plotter, middle_plot=True)
    # h_table = build_heuristic_for_one_target(node_goal, nodes, map_dim, plotter=None, middle_plot=False)
    # build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    h_dict = build_heuristic_for_multiple_targets(nodes,
                                                  nodes,
                                                  map_dim,
                                                  to_save=False,
                                                  plotter=None,
                                                  middle_plot=False)
    print(h_dict)


    # Save to logs_for_heuristics with a descriptive name
    json_save_path = Path(__file__).resolve().parent.parent / 'MAPF_DO_PROJECT_v1'/ 'logs_for_heuristics' / 'h_dict_of_3_3_empty.json'
    save_h_dict(h_dict, json_save_path)

    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()


# class ParallelHDict:
#     def __init__(self):
#         self.h_dict = {}
#         self._lock = threading.Lock()


# def get_node_from_open(open_list, target_name):
#     v_list = open_list
#     v_dict = {}
#     v_g_list = []
#     for node in v_list:
#         curr_g = node.g_dict[target_name]
#         v_g_list.append(curr_g)
#         if curr_g not in v_dict:
#             v_dict[curr_g] = []
#         v_dict[curr_g].append(node)
#
#     # some_v = min([node.g for node in v_list])
#     # out_nodes = [node for node in v_list if node.g == some_v]
#     # print([node.ID for node in t_nodes])
#     # next_node = random.choice(out_nodes)
#     next_node = random.choice(v_dict[min(v_g_list)])
#     return next_node

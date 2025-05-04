import re
import os
import math
import json
import time
import heapq
import random
import pstats
import cProfile
import itertools
from itertools import combinations, permutations, tee, pairwise
from datetime import datetime
from typing import *
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# GLOBAL OBJECTS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

color_names = [
    # 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # Single-letter abbreviations
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',  # Full names
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

markers = [
    ".",    # point marker
    ",",    # pixel marker
    "o",    # circle marker
    "v",    # triangle_down marker
    "^",    # triangle_up marker
    "<",    # triangle_left marker
    ">",    # triangle_right marker
    "1",    # tri_down marker
    "2",    # tri_up marker
    "3",    # tri_left marker
    "4",    # tri_right marker
    "s",    # square marker
    "p",    # pentagon marker
    "*",    # star marker
    "h",    # hexagon1 marker
    "H",    # hexagon2 marker
    "+",    # plus marker
    "x",    # x marker
    "D",    # diamond marker
    "d",    # thin_diamond marker
    "P",    # plus (filled) marker
    "X",    # x (filled) marker
]
lines = [
    "-",  # solid line
    "--", # dashed line
    "-.", # dash-dot line
    ":",  # dotted line
]

# markers_lines_dict = defaultdict(lambda: random.choice(markers))
markers_lines_dict = {}
colors_dict: DefaultDict[str, str | None] = defaultdict(lambda: None)
markers_iter = iter(markers)
colors_iter = iter(color_names)

mrc_dict = {
    "PrP-A*":
        {'color': 'blue',
         'marker-line': '-^',
         'marker': '^',
         },
    "LNS2-A*":
        {'color': 'teal',
         'marker-line': '-X',
         'marker': 'X',
         },
    "PrP":
        {'color': 'green',
         'marker-line': '-v',
         'marker': 'v',
         },
    "LNS2":
        {'color': 'blue',
         'marker-line': '-P',
         'marker': 'P',
         },
    "PIBT":
        {'color': 'salmon',
         'marker-line': '-h',
         'marker': 'h',
         },
    "LaCAM":
        {'color': 'indigo',
         'marker-line': '-1',
         'marker': '1',
         },
    "LaCAM*":
        {'color': 'plum',
         'marker-line': '-2',
         'marker': '2',
         },
    "MACGA":
        {'color': 'red',
         'marker-line': '-X',
         'marker': 'X',
         },
    "MACGA+PIBT":
        {'color': 'brown',
         'marker-line': '-d',
         'marker': 'd',
         },
}

# markers_lines_dict['LNS2'] = '-p'
# colors_dict['LNS2'] = 'blue'
#
# markers_lines_dict['PF-LNS2'] = '-*'
# colors_dict['PF-LNS2'] = 'red'
#
# markers_lines_dict['PrP'] = '-v'
# colors_dict['PrP'] = 'green'
#
# markers_lines_dict['PF-PrP'] = '-^'
# colors_dict['PF-PrP'] = 'orange'

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# GLOBAL CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class Node:
    def __init__(self, x: int, y: int, neighbours: List[str] | None = None):
        self.x: int = x
        self.y: int = y
        self.neighbours: List[str] = [] if neighbours is None else neighbours
        self.neighbours_nodes: List[Node] = []
        self.xy_name: str = f'{self.x}_{self.y}'

    @property
    def xy(self):
        return self.x, self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.xy_name < other.xy_name

    def __gt__(self, other):
        return self.xy_name > other.xy_name

    def __hash__(self):
        return hash(self.xy_name)

    def __str__(self):
        return self.xy_name

    def __repr__(self):
        return self.xy_name


class AgentAlg:
    def __init__(self, num: int, start_node: Node, goal_node: Node | None = None):
        self.num = num
        self.name = f'agent_{num}'
        self.start_node: Node = start_node
        self.start_node_name: str = self.start_node.xy_name
        self.curr_node: Node = start_node
        # self.curr_node_name: str = self.curr_node.xy_name
        self.goal_node: Node | None = goal_node
        if self.goal_node is not None:
            self.goal_node_name: str = self.goal_node.xy_name
        self.alt_goal_node: Node | None = None
        self.message: str = ''
        self.path: List[Node] | None = [self.start_node]
        self.k_path: List[Node] | None = [self.start_node]
        self.init_priority: float = random.random() if self.goal_node is not None else 0
        self.priority: float = self.init_priority

    @property
    def path_names(self):
        return [n.xy_name for n in self.path]

    def update_curr_node(self, i_time):
        if i_time >= len(self.path):
            self.curr_node = self.path[-1]
            return
        self.curr_node = self.path[i_time]

    def get_goal_node(self) -> Node:
        if self.alt_goal_node is not None:
            return self.alt_goal_node
        if self.goal_node is not None:
            return self.goal_node
        return self.curr_node

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other: Self):
        return self.priority < other.priority

    def __hash__(self):
        return hash(self.num)

    def __eq__(self, other):
        return self.num == other.num


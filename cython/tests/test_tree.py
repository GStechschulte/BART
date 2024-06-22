import os
import sys

from copy import copy

import numpy as np

# Access modules from parent directory since `pymc_bart` is not an installable package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from pymc_bart._tree import (
    Node, 
    Tree, 
    new_tree, 
    get_depth, 
    get_idx_left_child, 
    get_idx_right_child
)


def test_new_tree():

    # Initial leaf node value = mean
    leaf_node_value = 0.013728
    idx_data_points = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    num_observations = 10
    leaves_shape = 1

    # Test only __init__ method as `new_tree` is a static cdef method and
    # cannot be called from Python
    # tree = Tree(
    #     {
    #         0: Node.new_leaf_node(value=3.14, idx_data_points=np.array([1, 2, 3], dtype=np.int32)),
    #     },
    #     np.zeros((num_observations, leaves_shape)),
    #     [0]
    # )

    tree = new_tree(
        leaf_node_value,
        idx_data_points,
        num_observations,
        leaves_shape
    )

    print(tree.tree_structure)
    print(tree.output)
    cp_tree = copy(tree)


def test_split_node():
    index = 5
    split_node = Node(idx_split_variable=2, value=3.0)
    assert get_depth(index) == 2
    assert split_node.value == 3.0
    assert split_node.idx_split_variable == 2
    assert split_node.idx_data_points is None
    assert get_idx_left_child(index) == 11
    assert get_idx_right_child(index) == 12
    assert split_node.is_split_node() is True
    assert split_node.is_leaf_node() is False


def test_leaf_node():
    index = 5
    leaf_node = Node.new_leaf_node(value=3.14, idx_data_points=np.array([1, 2, 3], dtype=np.int32))
    assert get_depth(index) == 2
    assert leaf_node.value == 3.14
    assert leaf_node.idx_split_variable == -1
    assert np.array_equal(leaf_node.idx_data_points, np.array([1, 2, 3]))
    assert get_idx_left_child(index) == 11
    assert get_idx_right_child(index) == 12
    assert leaf_node.is_split_node() is False
    assert leaf_node.is_leaf_node() is True


def test_new_node():
    pass



def main():
    test_split_node()
    test_leaf_node()
    test_new_tree()
    print("All tests passed.")


if __name__ == "__main__":
    main()
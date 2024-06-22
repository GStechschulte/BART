from typing import Optional, List

cimport cython
import numpy as np
cimport numpy as cnp

# TODO: change `cpdef` funcs/methods to `cdef` once `pgbart.py` is cythonized
# TODO: inline certain functions/methods

# ? Should `Node` be converted to a struct?
# ? Should the `tree_structure` in `Tree` be converted to a struct?


cdef class Node:
 
    def __init__(
        self,
        value: float64_t = -1.0,
        nvalue: uint32_t = 0,
        idx_data_points: Optional[np.ndarray[intp_t]] = None,
        idx_split_variable: intp_t = -1,
        linear_params: Optional[List[np.ndarray[float64_t]]] = None
    ) -> None:
    
        self.value = value
        self.nvalue = nvalue
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable
        self.linear_params = linear_params
    
    @classmethod
    def new_leaf_node(
        cls, 
        value: np.ndarray[np.float_], 
        nvalue: int = 0,
        idx_data_points: Optional[np.ndarray[np.int_]] = None,
        idx_split_variable: int = -1,
        linear_params: Optional[List[np.ndarray[np.float_]]] = None
    ) -> "Node":
        """
        TODO: Cython currently does not support decorating cdef/@ccall methods with the @classmethod decorator.
        But creating a new instance of a class is a common operation. How to do this in Cython?
        """
        return cls(
            value=value, 
            nvalue=nvalue,
            idx_data_points=idx_data_points,
            idx_split_variable=idx_split_variable,
            linear_params=linear_params
            )

    cpdef bint is_split_node(self):
        return self.idx_split_variable >= 0

    cpdef bint is_leaf_node(self):
        return not self.is_split_node()

cpdef int get_idx_left_child(intp_t index):
    return index * 2 + 1

cpdef int get_idx_right_child(intp_t index):
    return index * 2 + 2

cpdef int get_depth(intp_t index):
    return (index + 1).bit_length() - 1

cdef class Tree:

    def __init__(
        self,
        dict tree_structure,
        cnp.ndarray[float64_t, ndim=2] output,
        list idx_leaf_nodes = None
    ):
        self.tree_structure = tree_structure
        self.output = output
        self.idx_leaf_nodes = idx_leaf_nodes

    cpdef Tree copy(self):
        cdef dict tree = {}
        for k, v in self.tree_structure.items():
            tree[k] = Node(
                value=v.value,
                nvalue=v.nvalue,
                idx_data_points=v.idx_data_points,
                idx_split_variable=v.idx_split_variable,
                linear_params=v.linear_params
            )
        
        if self.idx_leaf_nodes is not None:
            idx_leaf_nodes = self.idx_leaf_nodes.copy()
        
        if hasattr(self.output, "base"):
            output = self.output.base
        else:
            output = np.asarray(self.output)

        return Tree(
            tree,
            idx_leaf_nodes,
            output
        )

    cdef Node get_node(self, intp_t index):
        return self.tree_structure[index]

    cdef void set_node(self, intp_t index, Node node):
        self.tree_structure[index] = node
        if node.is_leaf_node() and self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.append(index)

    cdef void grow_leaf_node(
        self,
        Node current_node,
        int selected_predictor,
        cnp.ndarray[float64_t, ndim=1] split_value,
        intp_t index_leaf_node
    ):
        current_node.value = split_value
        current_node.idx_split_variable = selected_predictor
        current_node.idx_data_points = None
        if self.idx_leaf_nodes is not None:
            self.idx_leaf_nodes.remove(index_leaf_node)
    
    cpdef cnp.ndarray[float64_t, ndim=1] _predict(self):
        cdef cnp.ndarray[float64_t, ndim=2] output = np.asarray(self.output)
        cdef intp_t node_index
        if self.idx_leaf_nodes is not None:
            for node_index in self.idx_leaf_nodes:
                leaf_node = self.get_node(node_index)
                output[leaf_node.idx_data_points] = leaf_node.value
        
        return output.T
    
    cdef cnp.ndarray[float64_t, ndim=1] predict(
        self, 
        cnp.ndarray[float64_t, ndim=1] x,
        list excluded,
        int shape
    ):
        """
        Parameters:
        -----------
        x: np.ndarray
            The input data point to predict the output for.
        excluded: list = None
            The list of indices of the features to exclude from the prediction.
        shape: int = 1
            The shape of the output.
        """
        if excluded is None:
            excluded = [] 

    # TODO: Implement the `_traverse_tree` method
    # cdef cnp.ndarray[float64_t, ndim=1] _traverse_tree(
    #     self,
    #     cnp.ndarray[float64_t, ndim=1] X,
    #     list excluded,
    #     int shape
    # ):

    #     cdef int x_shape = (1,) if len(X.shape) == 1 else X.shape[:-1]
    #     cdef int nd_dims = (...,) + (None,) * len(x_shape)
    #     cdef list stack = [(0, np.ones(x_shape), 0)]
    #     cdef tuple p_d = (
    #         np.zeros(shape + x_shape) if isinstance(shape, tuple) else np.zeros((shape,) + x_shape)
    #     )

    #     cdef int node_index
    #     cdef int idx_split_variable
    #     cdef cnp.ndarray[float64_t, ndim=1] weights
    #     cdef Node node
        
    #     while stack:
    #         node_index, weights, idx_split_variable = stack.pop()
    #         node = self.get_node(node_index)

    cdef void _traverse_leaf_values(
        self,
        list leaf_values,
        list leaf_n_values,
        intp_t node_index
    ):
        cdef Node node = self.get_node(node_index)
        if node.is_leaf_node():
            leaf_values.append(node.value)
            leaf_n_values.append(node.nvalue)
        else:
            self._traverse_leaf_values(leaf_values, leaf_n_values, get_idx_left_child(node_index))
            self._traverse_leaf_values(leaf_values, leaf_n_values, get_idx_right_child(node_index))


cpdef Tree new_tree(
    float64_t leaf_node_value,
    cnp.ndarray[int32_t, ndim=1] idx_data_points,
    uint32_t num_obversations,
    uint32_t shape
):
    """Create a new tree with a single leaf node.

    Parameters:
    -----------
    leaf_node_value: float
        The value of the leaf node.
    idx_data_points: np.ndarray
        The indices of the data points in the leaf node.
    num_obversations: int
        The number of observations in the dataset.
    shape: int
        The shape of the output.

    Returns:
    --------
    Tree
        A new tree with a single leaf node.
    """
    cdef dict tree_structure = {
        0: Node.new_leaf_node(
            value=leaf_node_value, 
            idx_data_points=idx_data_points
        )
    }
    cdef list idx_leaf_nodes = [0]
    cdef cnp.ndarray[float64_t, ndim=2] output = np.zeros((num_obversations, shape), dtype=np.float64)#.squeeze()
    return Tree(tree_structure, output, idx_leaf_nodes)
from typing import Optional, List

cimport cython
import numpy as np
cimport numpy as cnp

from ._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t


cdef class Node:
    cdef public float64_t value
    cdef public uint32_t nvalue
    # Use a memoryview as buffer types (np.ndarray) can only be used as 
    # local variables within functions and not as an instance attribute
    # cdef public intp_t[:] idx_data_points
    cdef public int32_t[:] idx_data_points
    cdef public intp_t idx_split_variable
    cdef public list linear_params

    # Methods
    cpdef bint is_split_node(self)

    cpdef bint is_leaf_node(self)


cdef class Tree:
    cdef public dict tree_structure
    # Use a 2d memoryview as buffer types (np.ndarray) can only be used as 
    # local variables within functions and not as an instance attribute
    cdef public float64_t[:,:] output
    #cdef public list split_rules # TODO: type alias?
    cdef public list idx_leaf_nodes

    # Methods
    # @staticmethod
    # cdef Tree new_tree(
    #     float64_t leaf_node_value,
    #     intp_t[:] idx_data_points,
    #     uint32_t num_obversations,
    #     uint32_t shape
    # )

    cpdef Tree copy(self)

    cdef Node get_node(self, intp_t index)

    cdef void set_node(self, intp_t index, Node node)

    cdef void grow_leaf_node(
        self,
        Node current_node,
        int selected_predictor,
        cnp.ndarray[float64_t, ndim=1] split_value,
        intp_t index_leaf_node
    )
    
    cpdef cnp.ndarray[float64_t, ndim=1] _predict(self)
    
    cdef cnp.ndarray[float64_t, ndim=1] predict(
        self, 
        cnp.ndarray[float64_t, ndim=1] x,
        list excluded,
        int shape
        )

    # cdef cnp.ndarray[float64_t, ndim=1] _traverse_tree(
    #     self,
    #     cnp.ndarray[float64_t, ndim=1] X,
    #     list excluded,
    #     int shape
    # )

    cdef void _traverse_leaf_values(
        self,
        list leaf_values,
        list leaf_n_values,
        intp_t node_index
    )


cpdef Tree new_tree(
        float64_t leaf_node_value,
        # intp_t[:] idx_data_points,
        cnp.ndarray[int32_t, ndim=1] idx_data_points,
        uint32_t num_obversations,
        uint32_t shape
    )
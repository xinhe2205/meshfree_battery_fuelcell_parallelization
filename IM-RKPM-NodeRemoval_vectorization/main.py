import time
start_time = time.time()
import numpy as np
from numpy import sign

import scipy.sparse as sp

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit, njit, typed
import scipy.sparse as sp

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

from define_buttler_volmer import i_0_complex, alpha_lattice_complex, c_lattice_complex, Dn_complex, ocp_complex, i_se

from get_nodes_gauss_points import get_x_nodes_single_grain, get_x_nodes_multi_grain, x_G_and_def_J_time_weight_structured, x_G_b_and_det_J_b_structured, x_G_and_def_J_time_weight_multi_grains, x_G_b_and_det_J_b_multi_grains

from shape_function_in_domain import compute_phi_M, shape_grad_shape_func

from shape_func_interface_nodes import compute_phi_M_int, shape_grad_shape_func_int

from define_mechanical_stiffness_matrix import mechanical_stiffness_matrix, mechanical_force_matrix, mechanical_C_tensor
from define_diffusion_matrix_form import diffusion_matrix

from define_eva_at_gauss_points import evaluate_at_gauss_points

from read_image import read_in_image

from shape_func_correction_node_removal import modify_shape_func_node_removal

print('Define domain and parameters')
###############################
# Define domain 
###############################
x_min = -10e-6
x_max = 10e-6
y_min = -10e-6
y_max = 10e-6 

###############################
# Define time step
###############################
t = 1.0              # simulate for 10s
nt = 100             # nt is the number of time steps
dt = t/nt            # time step

IM_RKPM = 'True'  # if it is interfacial modified RKPM

Node_removal = 'True' # remove damaged nodes if it is true

###############################
# geometry
###############################

single_grain = 'False'   # True: single grain, False: multiple grains read from an image

if single_grain == 'True':
    angle = 0
else:
    angle = [26.0, np.pi, 75.0, np.pi/4.0, 121.0,np.pi*2.0/3.0, 149.0, \
             np.pi/2.0, 90.0, np.pi/3.0, 81.0, np.pi/4.0, 37.0, np.pi*2.0/3.0, 110.0, 0.0]

n_boundaries = 4   # number of boundaries

###############################
# Define material properties
###############################
Fday = 9.6485e4     # Faraday constant
R = 8.3145e0        # gas constant
Tk = 3.0515e2       # temperature in K

c_max = 49600.0     # maximum concentration

k_con = 10.0        # conductivity

Dx_div_Dy = 100.0

j_applied = -15.0     # j_applied

E = 138.87e9            # Youngs modulus (Pa)
nu = 0.3                # Poisson ratio
lambda_mechanical = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)         # lamme constants

k_i = 0.0125
k_f = 0.015

#######################################
# differential method
#######################################
    
differential_method = 'direct'    # 'implicite' or 'direct'    # specify which differential method to use, implicite: H1, H2, direct: directly differentiate
# if the IM_RKPM=True, differential method must be set to be direct.

integral_method = 'gauss'
damage_model = 'ON'       # ON or OFF

if integral_method == 'gauss':
# Define Guass int points and cells
    x_G_domain_rec = [[-3**0.5/3, -3**0.5/3],[-3**0.5/3, 3**0.5/3],\
                      [3**0.5/3, -3**0.5/3],[3**0.5/3, 3**0.5/3]] # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    x_G_domain_tri = [[1.0/6.0, 2.0/3.0],[1.0/6.0,1.0/6.0],[2.0/3.0, 1.0/6.0]]
    weight_G_domain_rec = [1.0,1.0,1.0,1.0]         # weight of each 2D Gauss points for rectangular
    weight_G_domain_tri = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    x_G_boundary = [-(3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5, -(3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, \
                    (3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, (3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5]#[-0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0,0.4058451513773972,0.7415311855993945,0.9491079123427585]#                   # coordinates of 1D Gauss points
    weight_G_boundary = [0.5-30**0.5/36, 0.5+30**0.5/36, 0.5+30**0.5/36, 0.5-30**0.5/36]#[0.1294849661688697,0.2797053914892766,0.3818300505051189,0.4179591836734694,0.3818300505051189,0.2797053914892766,0.1294849661688697]#         # weight of each 1D Gauss points

def_para_time = time.time()

print('time to define parameters = ' + "%s seconds" % (def_para_time - start_time))

############################################
# define RK nodes and gauss points(if needed)
############################################
print('define RK nodes')

if single_grain == 'True':
    n_intervals = 20      # number of intervals along each direction
    n_nodes = n_intervals+1 # number of nodes along each direction,
    x_nodes = np.array(get_x_nodes_single_grain(n_nodes,x_min,x_max,n_intervals,y_min,y_max)) # types: array
    num_interface_segments = 0
    interface_nodes = []
    BxByCxCy = []


else:
    file_name = 'reduced_8grain_ulm_large_grain_51x51.tiff'
    img_, unic_grain_id, num_pixels_x, num_pixels_y = read_in_image(file_name)
    # print(num_pixels_x, num_pixels_y)
    # x_nodes_interface_unique_id is the node id of interface nodes in whole node list
    cell_nodes_list, grain_id, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, cell_shape, num_rec_cell, num_tri_cell, x_nodes, nodes_grain_id, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, repeated_vertex, interface_segments, x_nodes_interface_unique, x_nodes_interface_unique_id  = get_x_nodes_multi_grain(x_min,x_max,y_min,y_max, num_pixels_x, num_pixels_y, img_)
    
    num_interface_segments = np.shape(interface_segments)[0]
    _ifs = np.asarray(interface_segments)
    interface_nodes = _ifs.reshape(num_interface_segments*2, 2) # interface nodes has repeated nodes on interface, while x_nodes_interface_unique is unique.
    BxByCxCy = _ifs.reshape(num_interface_segments, 4)  # the first column is x coordinates of point B, ......

    boundary_nodes = np.asarray(bottom_boundary_cell_nodes_list+top_boundary_cell_nodes_list+left_boundary_cell_nodes_list+right_boundary_cell_nodes_list)
    
    # x_nodes is array, all others are lists
    num_of_cell = int(num_rec_cell + num_tri_cell)    
    print('number of rectangular cell: '+str(num_rec_cell)) 
    print('number of triangular cell: '+str(num_tri_cell)) 

num_nodes = np.shape(x_nodes)[0]

print('number of nodes: ' + str(num_nodes))
# print('number of nodes on interface: ' + str(np.shape(x_nodes_interface_unique)[0]))

print('define gauss points')
# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian
if integral_method == 'gauss':
    if single_grain == 'True':
        x_G, det_J_time_weight = x_G_and_def_J_time_weight_structured(n_intervals, x_min,x_max,y_min,y_max,x_G_domain_rec,weight_G_domain_rec)
        x_G_b, det_J_b_time_weight = x_G_b_and_det_J_b_structured(n_boundaries, n_intervals, x_min, x_max, y_min, y_max, x_G_boundary, weight_G_boundary)
        gauss_angle = angle*np.ones(len(x_G))
        gauss_angle_b = angle*np.ones(len(x_G_b))
        Gauss_grain_id = np.zeros(len(x_G), dtype=int)
        Gauss_b_grain_id = np.zeros(len(x_G_b), dtype=int)
    else:
        # x_G_domain_tri = typed.List([typed.List(x) for x in x_G_domain_tri])
        # x_G_domain_rec = typed.List([typed.List(x) for x in x_G_domain_rec])
        # cell_nodes_list = typed.List([typed.List(x) for x in cell_nodes_list])
        # print('multi in domain')
        x_G, det_J_time_weight, gauss_angle, Gauss_grain_id= x_G_and_def_J_time_weight_multi_grains(num_of_cell,x_G_domain_rec, x_G_domain_tri,weight_G_domain_rec, weight_G_domain_tri, cell_shape, cell_nodes_list, grain_id, angle, repeated_vertex)
        # bottom_boundary_cell_nodes_list = typed.List([typed.List(x) for x in bottom_boundary_cell_nodes_list])
        # right_boundary_cell_nodes_list = typed.List([typed.List(x) for x in right_boundary_cell_nodes_list])
        # left_boundary_cell_nodes_list = typed.List([typed.List(x) for x in left_boundary_cell_nodes_list])
        # top_boundary_cell_nodes_list = typed.List([typed.List(x) for x in top_boundary_cell_nodes_list])
        # print('multi on boundary')
        x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id = x_G_b_and_det_J_b_multi_grains(x_min, x_max, y_min, y_max, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, x_G_boundary, weight_G_boundary, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, angle)
    x_G = np.array(x_G)
    x_G_b = np.array(x_G_b)
    num_gauss_points_in_domain = np.shape(x_G)[0]
    num_gauss_points_on_boundary = np.shape(x_G_b)[0]
    gauss_angle = np.array(gauss_angle)
    gauss_angle_b = np.array(gauss_angle_b)
    Gauss_grain_id = np.array(Gauss_grain_id)
    Gauss_b_grain_id = np.array(Gauss_b_grain_id)

# print(np.max(det_J_time_weight), np.min(det_J_time_weight), np.average(det_J_time_weight))
print('number of Gauss points in domain: ' + str(num_gauss_points_in_domain))
print('number of Gauss points on boundaries: ' + str(num_gauss_points_on_boundary))

def_nodes_gauss_points_time = time.time()

print('time to define nodes and Gauss points = ' + "%s seconds" % (def_nodes_gauss_points_time-def_para_time))

####################################################
# Compute shape function and its gradient in domain
#####################################################
print('Compute shape function and its gradient in domain')


HT0 = np.array([1,0,0],dtype=np.float64)     # transpose of the basis vector H
HT1 = np.array([0,-1,0],dtype=np.float64)   # for computation of gradient of shape function, d/dx
HT2 = np.array([0,0,-1],dtype=np.float64)   # for computation of gradient of shape function, d/dy

c = 2        # support size

if single_grain == 'True':
    a = c*(x_max-x_min)/n_intervals*np.ones(num_nodes)       # compact support size, shape: (num_nodes,)
else:

    h = np.zeros(num_nodes)
    for i in range(num_nodes):
        dist = ((x_nodes[i,0]-x_nodes[:,0])**2+(x_nodes[i,1]-x_nodes[:,1])**2)**0.5
        
        index_four_smallest = sorted(range(len(dist)), key=lambda sub: dist[sub])[:5]  # get the index of the four smallest index, the first one is always zero, so 5 here

        h[i] = dist[index_four_smallest][dist[index_four_smallest].tolist().index(max(dist[index_four_smallest]))]

    a = c*h #shape: (num_nodes,)

M = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_in_domain)],dtype=np.float64)
M_P_x = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_in_domain)],dtype=np.float64) # partial M partial x
M_P_y = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_in_domain)],dtype=np.float64) # partial M partial y


save_distance_function, save_distance_function_dx, save_distance_function_dy, save_point_D_coor, save_heavyside, save_heavyside_px, save_heavyside_py, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data, phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y = compute_phi_M(x_G, Gauss_grain_id, x_nodes,nodes_grain_id, a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM)


phi = csc_matrix((np.array(phi_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
phi_x = csc_matrix((np.array(phi_P_x_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
phi_y = csc_matrix((np.array(phi_P_x_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))


np.savetxt('distance_func_in_domain.txt', save_distance_function)
np.savetxt('distance_func_dx_in_domain.txt', save_distance_function_dx)
np.savetxt('distance_func_dy_in_domain.txt', save_distance_function_dy)
np.savetxt('points_D_coor.txt', save_point_D_coor)
np.savetxt('heavyside_124.txt', save_heavyside)
np.savetxt('heavyside_px_124.txt', save_heavyside_px)
np.savetxt('heavyside_py_124.txt', save_heavyside_py)


num_non_zero_phi_a = np.shape(np.array(phi_nonzero_index_row))[0]

shape_func_value, shape_func_times_det_J_time_weight_value, grad_shape_func_x_value, grad_shape_func_y_value, grad_shape_func_x_times_det_J_time_weight_value, grad_shape_func_y_times_det_J_time_weight_value = shape_grad_shape_func(x_G,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight, IM_RKPM)

# numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
shape_func = csc_matrix((np.array(shape_func_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
shape_func_times_det_J_time_weight = csc_matrix((np.array(shape_func_times_det_J_time_weight_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
grad_shape_func_x = csc_matrix((np.array(grad_shape_func_x_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
grad_shape_func_y = csc_matrix((np.array(grad_shape_func_y_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
grad_shape_func_x_times_det_J_time_weight = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
grad_shape_func_y_times_det_J_time_weight = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))

shape_func_weak_dis_times_det_J_time_weight_array = shape_func_times_det_J_time_weight.toarray()

comp_shape_func_grad_shape_func_in_domain = time.time()

print('time to compute the shape function and grad of shape function in domain = ' + "%s seconds" % (comp_shape_func_grad_shape_func_in_domain-def_nodes_gauss_points_time))


#####################################################################
# Compute shape function and its gradient on nodes on interfaces
######################################################################
num_interface_nodes = np.shape(x_nodes_interface_unique)[0]

M_int = np.array([np.zeros((3,3)) for _ in range(num_interface_nodes)],dtype=np.float64)
M_int_P_x = np.array([np.zeros((3,3)) for _ in range(num_interface_nodes)],dtype=np.float64) # partial M partial x
M_int_P_y = np.array([np.zeros((3,3)) for _ in range(num_interface_nodes)],dtype=np.float64) # partial M partial y


phi_nonzero_index_row_int, phi_nonzero_index_column_int, phi_nonzerovalue_data_int, phi_P_x_nonzerovalue_data_int, phi_P_y_nonzerovalue_data_int, M_int, M_int_P_x, M_int_P_y = compute_phi_M_int(x_nodes_interface_unique, x_nodes, a, M_int, M_int_P_x, M_int_P_y)
print('Computed phi and M on interface nodes')


phi_int = csc_matrix((np.array(phi_nonzerovalue_data_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
phi_x_int = csc_matrix((np.array(phi_P_x_nonzerovalue_data_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
phi_y_int = csc_matrix((np.array(phi_P_x_nonzerovalue_data_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))


num_non_zero_phi_a_int = np.shape(np.array(phi_nonzero_index_row_int))[0]

shape_func_value_int, shape_func_times_det_J_time_weight_value_int, grad_shape_func_x_value_int, grad_shape_func_y_value_int, grad_shape_func_x_times_det_J_time_weight_value_int, grad_shape_func_y_times_det_J_time_weight_value_int = shape_grad_shape_func_int(x_nodes_interface_unique,x_nodes, num_non_zero_phi_a_int,HT0, M_int, M_int_P_x, M_int_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data_int,phi_P_x_nonzerovalue_data_int,phi_P_y_nonzerovalue_data_int, phi_nonzero_index_row_int, phi_nonzero_index_column_int, det_J_time_weight)

# numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
shape_func_int = csc_matrix((np.array(shape_func_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
shape_func_times_det_J_time_weight_int = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
grad_shape_func_x_int = csc_matrix((np.array(grad_shape_func_x_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
grad_shape_func_y_int = csc_matrix((np.array(grad_shape_func_y_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
grad_shape_func_x_times_det_J_time_weight_int = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))
grad_shape_func_y_times_det_J_time_weight_int = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_int), (np.array(phi_nonzero_index_row_int),np.array(phi_nonzero_index_column_int))), shape = (num_interface_nodes, num_nodes))

np.savetxt('shape_func_int.txt', shape_func_int.toarray())  # i row correspomndes to ith nodes on interface
np.savetxt('grad_shape_func_x_int.txt', grad_shape_func_x_int.toarray())
np.savetxt('grad_shape_func_y_int.txt', grad_shape_func_y_int.toarray())
np.savetxt('x_nodes_interface.txt', x_nodes_interface_unique)

comp_shape_func_grad_shape_func_on_interface = time.time()
print('time to compute the shape function and grad of shape function on interfaces = ' + "%s seconds" % (comp_shape_func_grad_shape_func_on_interface - comp_shape_func_grad_shape_func_in_domain))


#######################################################
# Compute shape function and its gradient on boundaries
########################################################

print('Compute shape function and its gradient on boundaries')

M_b = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_on_boundary)],dtype=np.float64)
M_b_P_x = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_on_boundary)],dtype=np.float64)
M_b_P_y = np.array([np.zeros((3,3)) for _ in range(num_gauss_points_on_boundary)],dtype=np.float64)

save_distance_function_b, save_distance_function_dx_b, save_distance_function_dy_b, save_point_D_coor_b, save_heavyside_b, save_heavyside_px_b, save_heavyside_py_b, phi_b_nonzero_index_row, phi_b_nonzero_index_column, phi_b_nonzerovalue_data, phi_b_P_x_nonzerovalue_data, phi_b_P_y_nonzerovalue_data, M_b, M_b_P_x, M_b_P_y = compute_phi_M(x_G_b, Gauss_b_grain_id, x_nodes, nodes_grain_id, a, M_b, M_b_P_x, M_b_P_y, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM)

phi_b = csc_matrix((np.array(phi_b_nonzerovalue_data), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
phi_x_b = csc_matrix((np.array(phi_b_P_x_nonzerovalue_data), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
phi_y_b = csc_matrix((np.array(phi_b_P_x_nonzerovalue_data), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))

np.savetxt('heavyside_b_124.txt', save_heavyside_b)
np.savetxt('heavyside_px_b_124.txt', save_heavyside_px_b)
np.savetxt('heavyside_py_b_124.txt', save_heavyside_py_b)

num_non_zero_phi_a_b = np.shape(np.array(phi_b_nonzero_index_row))[0]

shape_func_b_value, shape_func_b_times_det_J_b_time_weight_value, grad_shape_func_b_x_value, grad_shape_func_b_y_value, grad_shape_func_b_x_times_det_J_b_time_weight_value, grad_shape_func_b_y_times_det_J_b_time_weight_value = shape_grad_shape_func(x_G_b, x_nodes, num_non_zero_phi_a_b, HT0, M_b, M_b_P_x, M_b_P_y, differential_method, HT1, HT2, phi_b_nonzerovalue_data, phi_b_P_x_nonzerovalue_data, phi_b_P_y_nonzerovalue_data, phi_b_nonzero_index_row, phi_b_nonzero_index_column, det_J_b_time_weight, IM_RKPM)


shape_func_b = csc_matrix((np.array(shape_func_b_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
shape_func_b_times_det_J_b_time_weight = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
grad_shape_func_b_x = csc_matrix((np.array(grad_shape_func_b_x_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
grad_shape_func_b_y = csc_matrix((np.array(grad_shape_func_b_y_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
grad_shape_func_b_x_times_det_J_b_time_weight = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))
grad_shape_func_b_y_times_det_J_b_time_weight = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value), (np.array(phi_b_nonzero_index_row),np.array(phi_b_nonzero_index_column))), shape = (num_gauss_points_on_boundary, num_nodes))

comp_shape_func_grad_shape_func_on_boundaries = time.time()

print('time to compute the shape function and grad of shape function on baoundaries = ' + "%s seconds" % (comp_shape_func_grad_shape_func_on_boundaries-comp_shape_func_grad_shape_func_in_domain))

print('assemble the matrix and solve all')


###################################################
# assemble the stiffness matrix for mechanical part
###################################################

# initialize the damage factor D_damage
D_damage = np.zeros((num_gauss_points_in_domain, 1))

# initialize the damage factor D_damage on the interfaces
D_damage_int = np.zeros((num_interface_nodes, 1))

# initialize the history parameter k
k = np.ones((num_gauss_points_in_domain, 1))*k_i

# initialize the history parameter k on interface
k_int = np.ones((num_interface_nodes, 1))*k_i

########################################################################
# assemble the matrix for diffusion problem and solve all
########################################################################

ini_charge_state = 0.92   

c_ini = np.array(np.ones((num_gauss_points_in_domain,1)))*c_max*ini_charge_state    # initial concentration at all gauss points, shape:(num_gauss_points_in_domain,1)

a_lattice_ini, da_lattice_ini = alpha_lattice_complex(c_ini/c_max)        # initial value of alpha_lattice and dalph_lattice/dx

c_lattice_ini, dc_lattice_ini = c_lattice_complex(c_ini/c_max)        # initial value of c_lattice and dc_lattice/dx

c_n = np.array(np.ones((num_nodes,1)))*c_max*ini_charge_state      # initial concentration at nodes
x_n = c_n/c_max                                                          # inital x

ini_potential = 3.712
phi_n = np.array(np.ones((num_nodes,1)))*ini_potential               # initial potential

dc_threshold = 1.0e-9
dphi_threshold = 1.0e-9     #when the norm of dc and dphi smaller than the threshold, stop newton iteratio nand move to next time step

dc = np.array(np.zeros((num_nodes,1)))
dphi = np.array(np.zeros((num_nodes,1)))      # give an initial value for dc and dphi to start the newton iteration

c_n1 = c_n+dc       # solutiona from previous newton interation
phi_n1 = phi_n+dphi

c_mean_domain = []        # concentration at the point whose index is 'index' at different time
sigma_x_mean_domain = []
sigma_y_mean_domain = []
tau_xy_mean_domain = []
VM_mean_domain = []
max_VM_domain = []
min_VM_domain = []

time_list = []

c_min_t = []
c_max_t = []

c_min_t_gauss = []
c_max_t_gauss = []

D_damage_mean = []
D_damage_min = []
D_damage_max = []

phi_min = []
phi_max = []
phi_mean = []

def_initial = time.time()

print('time to define initial condition = ' + "%s seconds" % (def_initial-comp_shape_func_grad_shape_func_on_boundaries))

K_min = []
K_max = []
K_mean = []

f_min = []
f_max = []
f_mean = []

if Node_removal == 'True':
    all_damaged_interface_nodes_id = np.array([]).astype(int)    # including all damaged interface nodes across all time steps
    damaged_interface_nodes_id = np.array([]).astype(int)        # including damaged interface nodes from current time step
    all_damaged_interface_nodes_array = np.zeros((nt, num_interface_nodes))

for ii in range(50):
    print('time_step:' +str(ii))

    t= dt+dt*ii

    newton_iter_num = 0

    if Node_removal == 'True':

        #  if there are new damaged nodes, modified the shape functions
        if np.shape(damaged_interface_nodes_id)[0] != 0:
            print('damaged_interface_nodes_id', damaged_interface_nodes_id)
        
            ###################################################################
            # updates the shape function in domain after node removal
            ###################################################################
            unique_row_index_to_be_modified = np.unique(np.where(shape_func.toarray()[:, damaged_interface_nodes_id] != 0)[0])
            shape_func_row_index_to_be_modified = unique_row_index_to_be_modified[np.where(shape_func.toarray()[unique_row_index_to_be_modified, :] != 0)[0]]
            shape_func_column_index_to_be_modified = np.where(shape_func.toarray()[unique_row_index_to_be_modified, :] != 0)[1]

            num_shape_func_tobe_modified = np.shape(shape_func_column_index_to_be_modified)[0]

            if num_shape_func_tobe_modified != 0:

                M_modi = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified)],dtype=np.float64)
                M_modi_P_x = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified)],dtype=np.float64) # partial M partial x
                M_modi_P_y = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified)],dtype=np.float64) # partial M partial y

                phi_array, phi_x_array, phi_y_array, shape_func_array, shape_func_times_det_J_time_weight_array, grad_shape_func_x_array, grad_shape_func_y_array, grad_shape_func_x_times_det_J_time_weight_array, grad_shape_func_y_times_det_J_time_weight_array = modify_shape_func_node_removal(ii, x_nodes, x_G, M_modi, M_modi_P_x, M_modi_P_y, phi.toarray(), phi_x.toarray(), phi_y.toarray(), shape_func_row_index_to_be_modified, shape_func_column_index_to_be_modified, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_time_weight, shape_func.toarray(), shape_func_times_det_J_time_weight.toarray(), grad_shape_func_x.toarray(), grad_shape_func_y.toarray(), grad_shape_func_x_times_det_J_time_weight.toarray(), grad_shape_func_y_times_det_J_time_weight.toarray(), damaged_interface_nodes_id)
                
                phi = csc_matrix(phi_array)
                phi_x = csc_matrix(phi_x_array)
                phi_y = csc_matrix(phi_y_array)

                shape_func = csc_matrix(shape_func_array)
                shape_func_times_det_J_time_weight = csc_matrix(shape_func_times_det_J_time_weight_array)
                grad_shape_func_x = csc_matrix(grad_shape_func_x_array)
                grad_shape_func_y = csc_matrix(grad_shape_func_y_array)
                grad_shape_func_x_times_det_J_time_weight = csc_matrix(grad_shape_func_x_times_det_J_time_weight_array)
                grad_shape_func_y_times_det_J_time_weight = csc_matrix(grad_shape_func_y_times_det_J_time_weight_array)
                
            ###################################################################
            # updates the shape function on boundary after node removal
            ###################################################################
            unique_row_index_to_be_modified_b = np.unique(np.where(shape_func_b.toarray()[:, damaged_interface_nodes_id] != 0)[0])
            shape_func_row_index_to_be_modified_b = unique_row_index_to_be_modified_b[np.where(shape_func_b.toarray()[unique_row_index_to_be_modified_b, :] != 0)[0]]
            shape_func_column_index_to_be_modified_b = np.where(shape_func_b.toarray()[unique_row_index_to_be_modified_b, :] != 0)[1]

            num_shape_func_tobe_modified_b = np.shape(shape_func_column_index_to_be_modified_b)[0]
            
            if num_shape_func_tobe_modified_b != 0:
                M_modi_b = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_b)],dtype=np.float64)
                M_modi_P_x_b = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_b)],dtype=np.float64) # partial M partial x
                M_modi_P_y_b = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_b)],dtype=np.float64) # partial M partial y

                phi_b_array, phi_x_b_array, phi_y_b_array, shape_func_b_array, shape_func_b_times_det_J_b_time_weight_array, grad_shape_func_b_x_array, grad_shape_func_b_y_array, grad_shape_func_b_x_times_det_J_b_time_weight_array, grad_shape_func_b_y_times_det_J_b_time_weight_array = modify_shape_func_node_removal(ii, x_nodes, x_G_b, M_modi_b, M_modi_P_x_b, M_modi_P_y_b, phi_b.toarray(), phi_x_b.toarray(), phi_y_b.toarray(), shape_func_row_index_to_be_modified_b, shape_func_column_index_to_be_modified_b, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_b_time_weight, shape_func_b.toarray(), shape_func_b_times_det_J_b_time_weight.toarray(), grad_shape_func_b_x.toarray(), grad_shape_func_b_y.toarray(), grad_shape_func_b_x_times_det_J_b_time_weight.toarray(), grad_shape_func_b_y_times_det_J_b_time_weight.toarray(), damaged_interface_nodes_id)
                
                phi_b = csc_matrix(phi_b_array)
                phi_x_b = csc_matrix(phi_x_b_array)
                phi_y_b = csc_matrix(phi_y_b_array)
                
                shape_func_b = csc_matrix(shape_func_b_array)
                shape_func_b_times_det_J_b_time_weight = csc_matrix(shape_func_b_times_det_J_b_time_weight_array)
                grad_shape_func_b_x = csc_matrix(grad_shape_func_b_x_array)
                grad_shape_func_b_y = csc_matrix(grad_shape_func_b_y_array)
                grad_shape_func_b_x_times_det_J_b_time_weight = csc_matrix(grad_shape_func_b_x_times_det_J_b_time_weight_array)
                grad_shape_func_b_y_times_det_J_b_time_weight = csc_matrix(grad_shape_func_b_y_times_det_J_b_time_weight_array)

            ###################################################################
            # updates the shape function on interface after node removal
            ###################################################################
            unique_row_index_to_be_modified_int = np.unique(np.where(shape_func_int.toarray()[:, damaged_interface_nodes_id] != 0)[0])
            shape_func_row_index_to_be_modified_int = unique_row_index_to_be_modified_int[np.where(shape_func_int.toarray()[unique_row_index_to_be_modified_int, :] != 0)[0]]
            shape_func_column_index_to_be_modified_int = np.where(shape_func_int.toarray()[unique_row_index_to_be_modified_int, :] != 0)[1]

            num_shape_func_tobe_modified_int = np.shape(shape_func_column_index_to_be_modified_int)[0]

            if num_shape_func_tobe_modified_int != 0:

                M_modi_int = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_int)],dtype=np.float64)
                M_modi_P_x_int = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_int)],dtype=np.float64) # partial M partial x
                M_modi_P_y_int = np.array([np.zeros((3,3)) for _ in range(num_shape_func_tobe_modified_int)],dtype=np.float64) # partial M partial y

                phi_int_array, phi_x_int_array, phi_y_int_array, shape_func_int_array, shape_func_times_det_J_time_weight_int_array, grad_shape_func_x_int_array, grad_shape_func_y_int_array, grad_shape_func_x_times_det_J_time_weight_int_array, grad_shape_func_y_times_det_J_time_weight_int_array = modify_shape_func_node_removal(ii, x_nodes, x_nodes_interface_unique, M_modi_int, M_modi_P_x_int, M_modi_P_y_int, phi_int.toarray(), phi_x_int.toarray(), phi_y_int.toarray(), shape_func_row_index_to_be_modified_int, shape_func_column_index_to_be_modified_int, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_time_weight, shape_func_int.toarray(), shape_func_times_det_J_time_weight_int.toarray(), grad_shape_func_x_int.toarray(), grad_shape_func_y_int.toarray(), grad_shape_func_x_times_det_J_time_weight_int.toarray(), grad_shape_func_y_times_det_J_time_weight_int.toarray(), damaged_interface_nodes_id)
                
                phi_int = csc_matrix(phi_int_array)
                phi_x_int = csc_matrix(phi_x_int_array)
                phi_y_int = csc_matrix(phi_y_int_array)

               
                shape_func_int = csc_matrix(shape_func_int_array)
                shape_func_times_det_J_time_weight_int = csc_matrix(shape_func_times_det_J_time_weight_int_array)
                grad_shape_func_x_int = csc_matrix(grad_shape_func_x_int_array)
                grad_shape_func_y_int = csc_matrix(grad_shape_func_y_int_array)
                grad_shape_func_x_times_det_J_time_weight_int = csc_matrix(grad_shape_func_x_times_det_J_time_weight_int_array)
                grad_shape_func_y_times_det_J_time_weight_int = csc_matrix(grad_shape_func_y_times_det_J_time_weight_int_array)

            # modify the node coefficient, the previous node coefficient is based on old shape func, in new time step, need to update it.
            c_n1 = np.dot(np.linalg.pinv(np.dot(np.transpose(shape_func.toarray()), shape_func_times_det_J_time_weight.toarray())), np.dot(np.dot(np.transpose(shape_func.toarray()), shape_func_weak_dis_times_det_J_time_weight_array), c_n1))
            c_n1[all_damaged_interface_nodes_id] = 0.0
            phi_n1 = np.dot(np.linalg.pinv(np.dot(np.transpose(shape_func.toarray()), shape_func_times_det_J_time_weight.toarray())), np.dot(np.dot(np.transpose(shape_func.toarray()), shape_func_weak_dis_times_det_J_time_weight_array), phi_n1))
            phi_n1[all_damaged_interface_nodes_id] = 0.0

    while newton_iter_num<10:#(np.linalg.norm(dc)/c_max/ini_charge_state>dc_threshold or np.linalg.norm(dphi)>dphi_threshold) or newton_iter_num==0:

        def_initial = time.time()


        newton_iter_num = newton_iter_num+1

        Dx,dDx_dx = Dn_complex(shape_func*(c_n1/c_max), D_damage)
        
        dDx_dc = dDx_dx/c_max                                    # diffucivity and dD/dc, size=(n_nodes*n_nodes, 1)

        Dy = Dx/Dx_div_Dy
        dDy_dc = dDx_dc/Dx_div_Dy

        j0,dj0_dx = i_0_complex(shape_func_b*(c_n1/c_max))
        dj0_dc = dj0_dx/c_max      


        E_eq, dE_eq_dx = ocp_complex(shape_func_b*(c_n1/c_max))
        dE_eq_dc = dE_eq_dx/c_max         
        

        djbv_deta, djbv_dj0, j_BV = i_se(shape_func_b*phi_n1, j0, E_eq, Fday, R, Tk)  # di/d\eta and dj/dj0 at cn and phi_n;


        #!!!! for all parameters denpend on cencentration or potential, if you want to investigate them at the gauss point, always calculate the concentration 
        #!!!! and potential at the gauss point, then use the concentration or potential at gauss point to calculate parameters that depend on concentration and potential.
        #!!!! if you calsulate the parameter at the nodes using concentration and potential at nodes, then interpolate the computed parameters at nodes to gauss point by times the 
        #!!!! shape function, it is not accurate!!!!

        # rotate the diffusivity
        R11 = (np.cos(gauss_angle)).reshape(num_gauss_points_in_domain,1)
        R12 = (np.sin(gauss_angle)).reshape(num_gauss_points_in_domain,1)
        R21 = (-np.sin(gauss_angle)).reshape(num_gauss_points_in_domain,1)
        R22 = (np.cos(gauss_angle)).reshape(num_gauss_points_in_domain,1)


        dD_dc_R11 = (dDx_dc)*(R11**2)+(dDy_dc)*(R12**2)
        dD_dc_R12 = (dDx_dc)*(R11*R21)+(dDy_dc)*(R12*R22)
        dD_dc_R21 = (dDx_dc)*(R11*R21)+(dDy_dc)*(R12*R22)
        dD_dc_R22 = (dDx_dc)*(R21**2)+(dDy_dc)*(R22**2)

        D_R11 = (Dx)*(R11**2)+(Dy)*(R12**2)
        D_R12 = (Dx)*(R11*R21)+(Dy)*(R12*R22)
        D_R21 = (Dx)*(R11*R21)+(Dy)*(R12*R22)
        D_R22 = (Dx)*(R21**2)+(Dy)*(R22**2)                # a sparse matrix dot an array is an array.
        
        # for a sparse matrix A of shape n*m, if want to times the ith column of A by ith component of 1d array B with shape of m, do A.multiple(B), this returns a sparse matrix
        # if you want to times the ith row of A by ith component of 1d array B with shape of n, you need do scipy.sparse.diags(B).dot(A), which also return a sparse matrix
        # if you want to times the ith row of A by ith component of 1d array B with shape of n*1, you can do A.multiply(B), which also return a sparse matrix

        #########################################
        # define the matrix form for diffusion
        #########################################

        K,f = diffusion_matrix(shape_func, shape_func_times_det_J_time_weight,grad_shape_func_x,D_R11,grad_shape_func_y,D_R21,grad_shape_func_x_times_det_J_time_weight,D_R12,D_R22,grad_shape_func_y_times_det_J_time_weight,\
                     dD_dc_R11,dD_dc_R21,c_n1,dD_dc_R12,dD_dc_R22,shape_func_b,djbv_deta,dE_eq_dc,shape_func_b_times_det_J_b_time_weight,djbv_dj0,dj0_dc,j_BV,j_applied,x_G_b,k_con,\
                        dt,Fday,c_n,phi_n1)
        
        if Node_removal == 'True' and np.shape(all_damaged_interface_nodes_id)[0] != 0:
            K = K.toarray()
            
            K[:, all_damaged_interface_nodes_id] = 0.0
            K[all_damaged_interface_nodes_id, :] = 0.0
            K[all_damaged_interface_nodes_id, all_damaged_interface_nodes_id] = 1.0

            K[:, all_damaged_interface_nodes_id+num_nodes] = 0.0
            K[all_damaged_interface_nodes_id+num_nodes, :] = 0.0
            K[all_damaged_interface_nodes_id+num_nodes, all_damaged_interface_nodes_id+num_nodes] = 1.0
            K = csc_matrix(K)

            f[all_damaged_interface_nodes_id] = 0.0
            f[all_damaged_interface_nodes_id+num_nodes] = 0.0
        
        def_diffusion_matrix = time.time()
        print('time to define diffusion matrix= ' + "%s seconds" % (def_diffusion_matrix - def_initial))

        #########################################
        # calculate the value at RPK nodes
        #########################################
            
        # Small diagonal regularization to prevent exact singularity
        K = K + sp.eye(K.shape[0], format='csc') * 1.0e-6
        du = spsolve(K, f)
        # du = np.dot(np.linalg.inv(K.toarray()), f)
        # Apply damping to improve robustness of Newton updates
        damping = 1.0e-3
        dc[:,0] = damping * du[0:num_nodes]
        dphi[:,0] = damping * du[num_nodes:]

        print('Number of Newton Iteration: ' + str(newton_iter_num), 'in Time Step: '+str(ii), np.linalg.norm(dc)/(c_max*ini_charge_state), np.linalg.norm(dphi))
        
        c_n1 = c_n1+dc
        phi_n1 = phi_n1+dphi                                       # tentative c and phi for n+1 time step for next newton iteration
        
        # if ii==0 and newton_iter_num==1:
        solv_diffusion = time.time()
        print('time to solve and update diffusion matrix= ' + "%s seconds" % (solv_diffusion - def_diffusion_matrix))


    c_n[:,0] = c_n1[:,0]
    phi_n[:,0] = phi_n1[:,0]             # update the concentration and potential after each time step
    
    ##################################################
    # evaluate the predicted value at all gauss points
    ##################################################
    c_G_domain, c_G_boundary = evaluate_at_gauss_points(shape_func.toarray(), shape_func_b.toarray(), c_n)
    phi_G_domain, phi_G_boundary = evaluate_at_gauss_points(shape_func.toarray(), shape_func_b.toarray(), phi_n)

    phi_mean.append(np.mean(phi_G_domain))
    phi_max.append(np.max(phi_G_domain))
    phi_min.append(np.min(phi_G_domain))

    c_mean_domain.append(np.mean(c_G_domain))

    c_min_t.append(np.min(c_n))
    c_max_t.append(np.max(c_n))

    c_min_t_gauss.append(np.min(c_G_domain))
    c_max_t_gauss.append(np.max(c_G_domain))

    time_list.append(t)

    K_min.append(np.min(K))
    K_max.append(np.max(K))
    K_mean.append(np.mean(K))

    f_min.append(np.min(f))
    f_max.append(np.max(f))
    f_mean.append(np.mean(f))

    ####################################################################
    # assemble matrix for mechanical simulation and solve
    ####################################################################
    # if ii==0:
    start_mechanical_time = time.time()

    print('define mechanical stiffness matrix')

    C11, C12, C13, C22, C23, C33 = mechanical_C_tensor(num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle)
    
    K_mechanical = mechanical_stiffness_matrix(C11, C12, C13, C22, C23, C33,E, x_nodes,num_gauss_points_in_domain, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y_times_det_J_time_weight, grad_shape_func_y)
    
    if Node_removal == 'True' and np.shape(all_damaged_interface_nodes_id)[0] != 0:
        K_mechanical = K_mechanical.toarray()

        K_mechanical[:, all_damaged_interface_nodes_id] = 0.0
        K_mechanical[all_damaged_interface_nodes_id, :] = 0.0
        K_mechanical[all_damaged_interface_nodes_id, all_damaged_interface_nodes_id] = 1.0

        K_mechanical[:, all_damaged_interface_nodes_id+num_nodes] = 0.0
        K_mechanical[all_damaged_interface_nodes_id+num_nodes, :] = 0.0
        K_mechanical[all_damaged_interface_nodes_id+num_nodes, all_damaged_interface_nodes_id+num_nodes] = 1.0

        K_mechanical = csc_matrix(K_mechanical)

    comp_mechanical_stiffness_matrix = time.time()

    print('time to compute the mechanical stiffness matrix = ' + "%s seconds" % (comp_mechanical_stiffness_matrix-start_mechanical_time))


    # compute Beta1
    a_lattice, da_lattice_dx = alpha_lattice_complex(c_G_domain/c_max)
    delta_a_lattice = a_lattice-a_lattice_ini
    delta_c = c_G_domain-c_ini
    beta_1 = delta_a_lattice/a_lattice_ini       # all of beta_1 and delta_c are np array
    #compute beta_2
    c_lattice, dc_lattice_dx = c_lattice_complex(c_G_domain/c_max)
    delta_c_lattice = c_lattice-c_lattice_ini
    beta_2 = delta_c_lattice/c_lattice_ini     # all of beta_2 and delta_c are np array

    epsilon_D1 = (R11**2*beta_1 + R12**2*beta_2)
    epsilon_D2 = (R21**2*beta_1 + R22**2*beta_2)
    epsilon_D3 = (R21*R11*2*beta_1 + R22*R12*2*beta_2)

    # solve the mechenical part without damage
    f_mechanical = mechanical_force_matrix(x_G, C11, C12, C13, C22, C23, C33, epsilon_D1, epsilon_D2, epsilon_D3, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight)
    
    if Node_removal == 'True' and np.shape(all_damaged_interface_nodes_id)[0] != 0:
        f_mechanical[all_damaged_interface_nodes_id] = 0.0
        f_mechanical[all_damaged_interface_nodes_id+num_nodes] = 0.0

    # if ii==0:
    comp_mechanical_force_matrix = time.time()

    print('time to compute mechanical force matrix= ' + "%s seconds" % (comp_mechanical_force_matrix-comp_mechanical_stiffness_matrix))

    
    # solve displacement field
    u_disp = spsolve(K_mechanical, f_mechanical)    #1d array

    ux = u_disp[0:num_nodes]          # disp at nodes along x
    uy = u_disp[num_nodes:]           # disp at nodes along y


    ux_gauss = shape_func*ux                 # disp at all gauss points along x
    uy_gauss = shape_func*uy                 # disp at all gauss points along y

    ##################################################
    #calculate the damage factor on all gauss points
    ##################################################

    """
    (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
    epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
    !!!!! reshape (grad_shape_func_x*ux)
    """
    epsilon_x = (grad_shape_func_x*ux).reshape(num_gauss_points_in_domain,1)       # normal strain along x at all gauss points
    epsilon_y = (grad_shape_func_y*uy).reshape(num_gauss_points_in_domain,1)       # normal strain along y at all gauss points
    gamma_xy = ((grad_shape_func_x*uy+ grad_shape_func_y*ux)*0.5).reshape(num_gauss_points_in_domain,1)      # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

    epsilon_x_mechanical = epsilon_x - epsilon_D1
    epsilon_y_mechanical = epsilon_y - epsilon_D2
    gamma_xy_mechanical = gamma_xy-epsilon_D3/2


    # calculate the principle strain:
    epsilon_1 = (epsilon_x_mechanical+epsilon_y_mechanical)/2+(((epsilon_x_mechanical-epsilon_y_mechanical)/2)**2+gamma_xy_mechanical**2)**0.5
    epsilon_2 = (epsilon_x_mechanical+epsilon_y_mechanical)/2-(((epsilon_x_mechanical-epsilon_y_mechanical)/2)**2+gamma_xy_mechanical**2)**0.5

    epsilon_1[epsilon_1<0] = 0
    epsilon_2[epsilon_2<0] = 0

    # print('epsilon_1:', np.mean(epsilon_1), np.max(epsilon_1), np.min(epsilon_1))
    # print('epsilon_2:', np.mean(epsilon_2), np.max(epsilon_2), np.min(epsilon_2))

    epsilon_e_eq = (epsilon_1**2+epsilon_2**2)**0.5

    # print('epsilon_eq:', np.mean(epsilon_e_eq), np.max(epsilon_e_eq), np.min(epsilon_e_eq))


    k=np.fmax(epsilon_e_eq, k)

    D_damage[np.logical_and(k>k_i, k<=k_f)] = (k[np.logical_and(k>k_i, k<=k_f)]-k_i)/(k_f-k_i)*k_f/k[np.logical_and(k>k_i, k<=k_f)]
    D_damage[k>k_f] = 1.0
    D_damage[k<=k_i] = 0.0

    #############################################################
    #calculate the damage factor on all nodes on interfaces
    #############################################################

    """
    (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
    epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
    !!!!! reshape (grad_shape_func_x*ux)
    """
    epsilon_x_int = (grad_shape_func_x_int*ux).reshape(num_interface_nodes,1)       # normal strain along x at nodes on interfaces
    epsilon_y_int = (grad_shape_func_y_int*uy).reshape(num_interface_nodes,1)       # normal strain along y at nodes on interfaces
    gamma_xy_int = ((grad_shape_func_x_int*uy+ grad_shape_func_y_int*ux)*0.5).reshape(num_interface_nodes,1)      # shear strain at nodes on interfaces, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

    c_int = np.dot(shape_func_int.toarray(), c_n)
    a_lattice_int, da_lattice_dx_int = alpha_lattice_complex(c_int/c_max)

    c_ini_int = np.array(np.ones((num_interface_nodes,1)))*c_max*ini_charge_state    # initial concentration at all gauss points, shape:(num_gauss_points_in_domain,1)
    a_lattice_ini_int, da_lattice_ini_int = alpha_lattice_complex(c_ini_int/c_max)
    
    delta_a_lattice_int = a_lattice_int-a_lattice_ini_int
    delta_c_int = c_int-c_ini_int
    beta_1_int = delta_a_lattice_int/a_lattice_ini_int       # all of beta_1 and delta_c are np array
    
    #compute beta_2
    c_lattice_int, dc_lattice_dx_int = c_lattice_complex(c_int/c_max)
    c_lattice_ini_int, dc_lattice_dx_ini_int = c_lattice_complex(c_ini_int/c_max)

    delta_c_lattice_int = c_lattice_int-c_lattice_ini_int
    beta_2_int = delta_c_lattice_int/c_lattice_ini_int     # all of beta_2 and delta_c are np array

    epsilon_D1_int = beta_1_int
    epsilon_D2_int = beta_2_int
    
    epsilon_x_mechanical_int = epsilon_x_int - epsilon_D1_int
    epsilon_y_mechanical_int = epsilon_y_int - epsilon_D2_int
    gamma_xy_mechanical_int = gamma_xy_int


    # calculate the principle strain:
    epsilon_1_int = (epsilon_x_mechanical_int+epsilon_y_mechanical_int)/2+(((epsilon_x_mechanical_int-epsilon_y_mechanical_int)/2)**2+gamma_xy_mechanical_int**2)**0.5
    epsilon_2_int = (epsilon_x_mechanical_int+epsilon_y_mechanical_int)/2-(((epsilon_x_mechanical_int-epsilon_y_mechanical_int)/2)**2+gamma_xy_mechanical_int**2)**0.5

    epsilon_1_int[epsilon_1_int<0] = 0
    epsilon_2_int[epsilon_2_int<0] = 0

    epsilon_e_eq_int = (epsilon_1_int**2+epsilon_2_int**2)**0.5
    
    k_int=np.fmax(epsilon_e_eq_int, k_int)

    D_damage_int[np.logical_and(k_int>k_i, k_int<=k_f)] = (k_int[np.logical_and(k_int>k_i, k_int<=k_f)]-k_i)/(k_f-k_i)*k_f/k_int[np.logical_and(k_int>k_i, k_int<=k_f)]
    D_damage_int[k_int>k_f] = 1.0
    D_damage_int[k_int<=k_i] = 0.0

    if Node_removal == 'True':
        # # for da_in in all_damaged_interface_nodes_id:
        # #     D_damage_int[np.where(x_nodes_interface_unique_id==da_in)[0]] = 1.0
        damaged_interface_nodes_id = np.array(x_nodes_interface_unique_id)[np.where(D_damage_int == 1.0)[0].astype(int)]

        # new damaged nodes
        damaged_interface_nodes_id = np.array([element for element in damaged_interface_nodes_id if element not in all_damaged_interface_nodes_id])
        # add new damaged nodes id to all damaged node id array
        all_damaged_interface_nodes_id = np.append(all_damaged_interface_nodes_id, damaged_interface_nodes_id).astype(int)
        
        #save all damaged interface node id to array
        all_damaged_interface_nodes_array[ii, 0:np.shape(all_damaged_interface_nodes_id)[0]] = all_damaged_interface_nodes_id


        
        # for i_curr_damaged_index in damaged_interface_nodes_id:
        #     if i_curr_damaged_index not in all_damaged_interface_nodes_id:
        #         all_damaged_interface_nodes_id = np.array(all_damaged_interface_nodes_id.tolist() + [i_curr_damaged_index])

        shape_func_weak_dis_times_det_J_time_weight_array = shape_func_times_det_J_time_weight.toarray()


    if damage_model == 'OFF':
        D_damage[:] = 0.0
        D_damage_int[:] = 0
        damaged_interface_nodes_id = np.array([]).astype(int)

    # update the damge factor

    C11, C12, C13, C22, C23, C33 = mechanical_C_tensor(num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle)

    sigma_x = epsilon_x_mechanical*C11+epsilon_y_mechanical*C12+gamma_xy_mechanical*2*C13   # normal stress along x-direction at all gauess points
    sigma_y = epsilon_x_mechanical*C12+epsilon_y_mechanical*C22+gamma_xy_mechanical*2*C23  # normal stress along y-direction at all gauess points
    tau_xy =  epsilon_x_mechanical*C13+epsilon_y_mechanical*C23+gamma_xy_mechanical*2*C33  # shear stress at all gauess points


    Von_mises_stress = (sigma_x**2+sigma_y**2-sigma_x*sigma_y+3*tau_xy**2)**0.5

    sigma_x_mean_domain.append(np.mean(sigma_x))
    sigma_y_mean_domain.append(np.mean(sigma_y))
    tau_xy_mean_domain.append(np.mean(tau_xy))
    VM_mean_domain.append(np.mean(Von_mises_stress))
    max_VM_domain.append(np.max(Von_mises_stress))
    min_VM_domain.append(np.min(Von_mises_stress))
    D_damage_mean.append(np.mean(D_damage))
    D_damage_min.append(np.min(D_damage))
    D_damage_max.append(np.max(D_damage))

    if ii==0:
        comp_solve_mechanical_stress = time.time()
        print('time to solve mechanical = ' + "%s seconds" % (comp_solve_mechanical_stress-comp_mechanical_force_matrix))




# # ######################################
# #     post-process
# # #######################################
# # print('averaged damage factor on Gauss points: ', D_damage_mean)
# # print('averaged VM in domain:', VM_mean_domain)
# # print('averaged concentration on gauss points:', c_mean_domain)
# # print('maximum concentration on nodes:', c_max_t)
# # print('minimum concentration on nodes:', c_min_t)
# # print('maximum concentration on Gauss points:', c_max_t_gauss)
# # print('minimum concentration on Gauss points:', c_min_t_gauss)
# # print('maximum VM on Gauss points:', max_VM_domain)
# # print('minimum VM on Gauss points:', min_VM_domain)
np.savetxt('all_damaged_interface_nodes_history229.txt', all_damaged_interface_nodes_array)
np.savetxt('ave_damage_G_DM_damage_long229_direct_distance.txt', D_damage_mean)
np.savetxt('min_damage_G_DM_damage_long229_direct_distance.txt', D_damage_min)
np.savetxt('max_damage_G_DM_damage_long229_direct_distance.txt', D_damage_max)
np.savetxt('ave_VM_DM_damage_long229_direct_distance.txt', VM_mean_domain)
np.savetxt('max_VM_DM_damage_long229_direct_distance.txt', max_VM_domain)
np.savetxt('min_VM_DM_damage_long229_direct_distance.txt', min_VM_domain)
np.savetxt('ave_con_DM_damage_long229_direct_distance.txt', c_mean_domain)
np.savetxt('max_con_DM_damage_long229_direct_distance.txt', c_max_t_gauss)
np.savetxt('min_con_DM_damage_long229_direct_distance.txt', c_min_t_gauss)
np.savetxt('ave_phi_damage_long229_direct_distance.txt', phi_mean)
np.savetxt('max_phi_damage_long229_direct_distance.txt', phi_max)
np.savetxt('min_phi_damage_long229_direct_distance.txt', phi_min)

# _DM_nodamage means the multigrain case with different angles, direct gradient, no damage, only concentration and mechanical simulation,.

# # fig1 = plt.plot()

# # plt.plot(time_list, sigma_x_mean_domain, 'b', label = 'Meshfree')

# # plt.xlabel('time(s)')
# # plt.ylabel('sigma_x(mean)')
# # plt.legend()

# # plt.grid()

# # # plt.show()
# # plt.savefig('verify_sigma_x',dpi=300)

# # fig2 = plt.plot()

# # plt.plot(time_list, sigma_y_mean_domain, 'b', label = 'Meshfree')

# # plt.xlabel('time(s)')
# # plt.ylabel('sigma_y(mean)')
# # plt.legend()

# # plt.grid()

# # # plt.show()
# # plt.savefig('verify_sigma_y',dpi=300)

# # fig3 = plt.plot()

# # plt.plot(time_list, tau_xy_mean_domain, 'b', label = 'Meshfree')

# # plt.xlabel('time(s)')
# # plt.ylabel('tau_xy(mean)')
# # plt.legend()

# # plt.grid()

# # # plt.show()
# # plt.savefig('verify_mean_tau_xy',dpi=300)

# # fig4 = plt.plot()

# # plt.plot(time_list, VM_mean_domain, 'b', label = 'Meshfree')

# # plt.xlabel('time(s)')
# # plt.ylabel('VM(mean)')
# # plt.legend()

# # plt.grid()

# # # plt.show()
# # plt.savefig('verify_mean_VM',dpi=300)

# finish_time = time.time()

# print('time to assemble matrix and solve equation = ' + "%s seconds" % (finish_time-comp_shape_func_grad_shape_func_on_boundaries))


# print('total running time = ' + "%s seconds" % (finish_time - start_time))


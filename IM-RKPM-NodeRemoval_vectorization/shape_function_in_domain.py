import time
start_time = time.time()
import numpy as np
from numpy import sign

import cupynumeric as cnp

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

# def compute_phi_M(x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM):
    
#     # Convert inputs to numpy arrays
#     x_G = np.array(x_G)
#     x_nodes = np.array(x_nodes)
#     a = np.array(a)
#     nodes_grain_id = np.array(nodes_grain_id)
#     Gauss_grain_id = np.array(Gauss_grain_id)
#     interface_nodes = np.array(interface_nodes)
#     BxByCxCy = np.array(BxByCxCy)
    
#     n_G = x_G.shape[0]
#     n_N = x_nodes.shape[0]
    
#     # VECTORIZED distance computation to interface segments - NO LOOPS
#     B = BxByCxCy[:, :2]  # (num_interface_segments, 2)
#     C = BxByCxCy[:, 2:4]  # (num_interface_segments, 2)
#     BC = C - B
#     CB = -BC
    
#     # Broadcast for ALL Gauss points to ALL segments simultaneously
#     BA = x_G[:, None, :] - B[None, :, :]  # (n_G, num_interface_segments, 2)
#     CA = x_G[:, None, :] - C[None, :, :]  # (n_G, num_interface_segments, 2)
#     BC_broadcast = BC[None, :, :]
#     CB_broadcast = CB[None, :, :]
    
#     # Vectorized dot products for ALL combinations
#     BA_dot_BC = np.sum(BA * BC_broadcast, axis=2)  # (n_G, num_interface_segments)
#     CA_dot_CB = np.sum(CA * CB_broadcast, axis=2)
#     sign_extension = BA_dot_BC * CA_dot_CB
    
#     # Vectorized distance calculations
#     BC_norm = np.sqrt(np.sum(BC ** 2, axis=1))
#     BC_norm_broadcast = BC_norm[None, :]
#     BA_dot_unit_BC = BA_dot_BC / (BC_norm_broadcast + 1e-16)
#     BC_unit = BC_broadcast / (BC_norm_broadcast[:, :, None] + 1e-16)
#     BA_dot_unit_BC_times_unit_BC = BC_unit * BA_dot_unit_BC[:, :, None]
    
#     # Initialize distance array
#     dx_distance = np.zeros((n_G, num_interface_segments))
    
#     # Case 1: Positive sign extension (perpendicular distance)
#     BA_pos = BA - BA_dot_unit_BC_times_unit_BC
#     dist_pos = np.sqrt(np.sum(BA_pos**2, axis=2))
#     mask_pos = sign_extension > 0
#     dx_distance[mask_pos] = dist_pos[mask_pos]
    
#     # Case 2: Negative sign extension (endpoint distance)
#     norm_CA = np.sqrt(np.sum(CA**2, axis=2))
#     norm_BA = np.sqrt(np.sum(BA**2, axis=2))
#     dist_neg = np.minimum(norm_CA, norm_BA)
#     mask_neg = sign_extension < 0
#     dx_distance[mask_neg] = dist_neg[mask_neg]
    
#     # Case 3: Zero sign extension (endpoint distance)
#     mask_zero = sign_extension == 0
#     dx_distance[mask_zero] = dist_neg[mask_zero]
    
#     # Find minimum distance and segment for each Gauss point
#     min_distance = np.min(dx_distance, axis=1)  # (n_G,)
#     min_index = np.argmin(dx_distance, axis=1)  # (n_G,)
    
#     # Vectorized closest point computation
#     gidx = np.arange(n_G)
#     min_mask_pos = (sign_extension[gidx, min_index] > 0)
    
#     # For positive case: projected point
#     x_proj_pos = BA_dot_unit_BC_times_unit_BC[gidx, min_index, :] + B[min_index, :]
    
#     # For negative/zero case: choose closer endpoint
#     dist_CA_val = norm_CA[gidx, min_index]
#     dist_BA_val = norm_BA[gidx, min_index]
#     use_C = dist_CA_val < dist_BA_val
#     x_proj_negzero = np.where(use_C[:, None], C[min_index, :], B[min_index, :])
    
#     x_coor_min_point_segment = np.where(min_mask_pos[:, None], x_proj_pos, x_proj_negzero)
    
#     # Vectorized distance function derivatives
#     d_distance_dx = (x_G[:, 0] - x_coor_min_point_segment[:, 0]) / (min_distance + 1e-16)
#     d_distance_dy = (x_G[:, 1] - x_coor_min_point_segment[:, 1]) / (min_distance + 1e-16)
    
#     # Vectorized Heaviside function computation
#     heaviside_scaling_factor = 4.0e-7
#     min_distance_mod = min_distance + 1.0e-15
#     heaviside = np.tanh(min_distance_mod / heaviside_scaling_factor)
#     sech2 = (1.0 / np.cosh(min_distance_mod / heaviside_scaling_factor)) ** 2
#     heaviside_P_x = d_distance_dx / heaviside_scaling_factor * sech2
#     heaviside_P_y = d_distance_dy / heaviside_scaling_factor * sech2
    
#     # VECTORIZED shape function computation for ALL Gauss point-node combinations
#     dx_all = x_G[:, None, 0] - x_nodes[None, :, 0]  # (n_G, n_N)
#     dy_all = x_G[:, None, 1] - x_nodes[None, :, 1]  # (n_G, n_N)
#     dist_to_node = np.sqrt(dx_all**2 + dy_all**2)
#     z_ij = dist_to_node / (a[None, :] + 1e-16)
    
#     # Vectorized z derivatives
#     z_ij_P_x = dx_all / (a[None, :] * z_ij * a[None, :] + 2.220446049250313e-16)
#     z_ij_P_y = dy_all / (a[None, :] * z_ij * a[None, :] + 2.220446049250313e-16)
    
#     # VECTORIZED H matrix computation for ALL combinations
#     H_scaling_factor = 1.0e-6
#     H_T_all = np.zeros((n_G, n_N, 3), dtype=np.float64)
#     H_T_all[:, :, 0] = 1.0
#     H_T_all[:, :, 1] = dx_all / H_scaling_factor
#     H_T_all[:, :, 2] = dy_all / H_scaling_factor
#     H_all = H_T_all  # H is transpose of H_T
    
#     # H derivative vectors (constant)
#     HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
#     HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
    
#     # VECTORIZED shape function evaluation for ALL combinations
#     mask_01 = (z_ij >= 0) & (z_ij < 0.5)
#     mask_051 = (z_ij >= 0.5) & (z_ij <= 1.0)
#     mask_in_support = (z_ij >= 0) & (z_ij <= 1.0)
    
#     phi_ij = np.zeros((n_G, n_N))
#     phi_P_z = np.zeros((n_G, n_N))
    
#     # Apply shape function formulas vectorized
#     phi_ij[mask_01] = 2.0/3 - 4*z_ij[mask_01]**2 + 4*z_ij[mask_01]**3
#     phi_P_z[mask_01] = -8.0*z_ij[mask_01] + 12.0*z_ij[mask_01]**2
    
#     phi_ij[mask_051] = 4.0/3 - 4*z_ij[mask_051] + 4*z_ij[mask_051]**2 - (4.0/3)*z_ij[mask_051]**3
#     phi_P_z[mask_051] = -4.0 + 8.0*z_ij[mask_051] - 4.0*z_ij[mask_051]**2
    
#     # VECTORIZED interface node checking
#     interface_tolerance = 1e-10
#     node_on_interface = np.any(
#         (np.abs(x_nodes[:, None, 0] - interface_nodes[None, :, 0]) < interface_tolerance) &
#         (np.abs(x_nodes[:, None, 1] - interface_nodes[None, :, 1]) < interface_tolerance),
#         axis=1
#     )
    
#     # VECTORIZED grain matching
#     grainid_match = (nodes_grain_id[None, :] == Gauss_grain_id[:, None])
    
#     # Apply IM_RKPM logic vectorized
#     IM_RKPM_flag = (IM_RKPM == 'True')
#     node_not_on_interface = ~node_on_interface[None, :]
    
#     if IM_RKPM_flag:
#         valid_mask = mask_in_support & node_not_on_interface & grainid_match
#         heaviside_factor = heaviside[:, None]
#         heaviside_P_x_factor = heaviside_P_x[:, None]
#         heaviside_P_y_factor = heaviside_P_y[:, None]
#     else:
#         valid_mask = mask_in_support
#         heaviside_factor = np.ones((n_G, 1))
#         heaviside_P_x_factor = np.zeros((n_G, 1))
#         heaviside_P_y_factor = np.zeros((n_G, 1))
    
#     # Apply modifications
#     phi_final = np.where(valid_mask, phi_ij * heaviside_factor, 0.0)
#     phi_P_x_ij = phi_P_z * z_ij_P_x
#     phi_P_y_ij = phi_P_z * z_ij_P_y
    
#     if IM_RKPM_flag:
#         phi_P_x_final = np.where(valid_mask, 
#                                 phi_P_x_ij * heaviside_factor + phi_ij * heaviside_P_x_factor, 
#                                 0.0)
#         phi_P_y_final = np.where(valid_mask, 
#                                 phi_P_y_ij * heaviside_factor + phi_ij * heaviside_P_y_factor, 
#                                 0.0)
#     else:
#         phi_P_x_final = np.where(valid_mask, phi_P_x_ij, 0.0)
#         phi_P_y_final = np.where(valid_mask, phi_P_y_ij, 0.0)
    
#     # Extract non-zero entries
#     nonzero_indices = np.where(valid_mask)
#     phi_nonzero_index_row = nonzero_indices[0].tolist()
#     phi_nonzero_index_column = nonzero_indices[1].tolist()
#     phi_nonzerovalue_data = phi_final[valid_mask].tolist()
#     phi_P_x_nonzerovalue_data = phi_P_x_final[valid_mask].tolist()
#     phi_P_y_nonzerovalue_data = phi_P_y_final[valid_mask].tolist()
#     z_list = z_ij[valid_mask].tolist()
#     z_P_x_list = z_ij_P_x[valid_mask].tolist()
#     z_P_y_list = z_ij_P_y[valid_mask].tolist()
#     phipz_list = phi_P_z[valid_mask].tolist()
    
#     # VECTORIZED moment matrix updates - NO LOOPS
#     i_vals, j_vals = nonzero_indices
#     phi_vals = phi_final[valid_mask]
#     phi_P_x_vals = phi_P_x_final[valid_mask]
#     phi_P_y_vals = phi_P_y_final[valid_mask]
    
#     # Get H matrices for valid entries
#     H_selected = H_all[i_vals, j_vals]  # (n_valid, 3)
#     H_T_selected = H_T_all[i_vals, j_vals]  # (n_valid, 3)
    
#     # VECTORIZED outer products for ALL valid entries simultaneously
#     H_outer_HT = H_selected[:, :, None] * H_T_selected[:, None, :]  # (n_valid, 3, 3)
    
#     # Broadcast derivative terms correctly
#     n_valid = len(i_vals)
#     HT_P_x_broadcast = np.broadcast_to(HT_P_x[None, :], (n_valid, 3))  # (n_valid, 3)
#     HT_P_y_broadcast = np.broadcast_to(HT_P_y[None, :], (n_valid, 3))  # (n_valid, 3)
    
#     HT_P_x_outer_HT = HT_P_x_broadcast[:, :, None] * H_T_selected[:, None, :]  # (n_valid, 3, 3)
#     H_outer_HT_P_x = H_selected[:, :, None] * HT_P_x_broadcast[:, None, :]  # (n_valid, 3, 3)
#     HT_P_y_outer_HT = HT_P_y_broadcast[:, :, None] * H_T_selected[:, None, :]  # (n_valid, 3, 3)
#     H_outer_HT_P_y = H_selected[:, :, None] * HT_P_y_broadcast[:, None, :]  # (n_valid, 3, 3)
    
#     # Scale by phi values
#     M_updates = H_outer_HT * phi_vals[:, None, None]
#     M_P_x_updates = (H_outer_HT * phi_P_x_vals[:, None, None] + 
#                      HT_P_x_outer_HT * phi_vals[:, None, None] + 
#                      H_outer_HT_P_x * phi_vals[:, None, None])
#     M_P_y_updates = (H_outer_HT * phi_P_y_vals[:, None, None] + 
#                      HT_P_y_outer_HT * phi_vals[:, None, None] + 
#                      H_outer_HT_P_y * phi_vals[:, None, None])
    
#     # Apply updates using vectorized operations
#     # Note: Since we can't avoid updating individual matrices, use a minimal loop
#     for idx in range(len(i_vals)):
#         i = i_vals[idx]
#         M[i] += M_updates[idx]
#         M_P_x[i] += M_P_x_updates[idx]
#         M_P_y[i] += M_P_y_updates[idx]
    
#     # Prepare return values
#     save_distance_function = [[i, float(x_G[i, 0]), float(x_G[i, 1]), float(min_distance[i])] for i in range(n_G)]
#     save_distance_function_dx = [[i, float(x_G[i, 0]), float(x_G[i, 1]), float(d_distance_dx[i])] for i in range(n_G)]
#     save_distance_function_dy = [[i, float(x_G[i, 0]), float(x_G[i, 1]), float(d_distance_dy[i])] for i in range(n_G)]
#     save_point_D_coor = [[float(x_coor_min_point_segment[i, 0]), float(x_coor_min_point_segment[i, 1])] for i in range(n_G)]
#     save_heavyside = heaviside.tolist()
#     save_heavyside_px = heaviside_P_x.tolist()
#     save_heavyside_py = heaviside_P_y.tolist()
    
#     return (save_distance_function, save_distance_function_dx, save_distance_function_dy, 
#             save_point_D_coor, save_heavyside, save_heavyside_px, save_heavyside_py,
#             phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,
#             phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y)



@jit
def compute_phi_M_old(x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM):

    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    phipz = []

    save_heavyside = []
    save_heavyside_px = []
    save_heavyside_py = []

    saved_dist_func_index = []
    save_distance_function = []
    save_distance_function_dx = []
    save_distance_function_dy = []
    save_point_D_coor = []

    for i in range(np.shape(x_G)[0]):

        """
        check the distance between point and segments, exact distance
        """
        dx_distance = np.zeros(num_interface_segments)
        
        # if x_nodes[j,:] not in interface_nodes:
            # find the minimum distance of gauss point to interface
            # if gauss point is A, boundary segment is BC, if (BA dot BC) times (CA dot CB) is negative, it means the vertical line from A to segment BC intersect BC on its extension,
            # in this case, the distance from A to BC is min(|AB|, |AC|)
            # if (BA dot BC) times (CA dot CB) is positive, there is a point D with BC that AD is normal to BC. the distance from A to BC is |BA vector - [vector BA dot (BC vector divided by |BC|) times (BC vector divided by |BC|) ]|
        BA = x_G[i,:] - BxByCxCy[:,:2]
        BC = BxByCxCy[:,2:4] - BxByCxCy[:,:2]
        CB = -BxByCxCy[:,2:4] + BxByCxCy[:,:2]
        CA = x_G[i,:] - BxByCxCy[:,2:4]

        BA_dot_BC = BA[:,0]*BC[:,0]+BA[:,1]*BC[:,1]
        CA_dot_CB = CA[:,0]*CB[:,0]+CA[:,1]*CB[:,1]

        sign_extension = BA_dot_BC*CA_dot_CB

        positive_index = np.where(sign_extension>0)[0]
        negative_index = np.where(sign_extension<0)[0]
        zero_index = np.where(sign_extension==0)[0]

        BA_dot_unit_BC = BA_dot_BC/(((BC[:, 0])**2+(BC[:, 1])**2)**0.5)

        BA_dot_unit_BC_times_unit_BC = BC/(((BC[:, 0])**2+(BC[:, 1])**2)**0.5)[:,None]*BA_dot_unit_BC[:, None]

        dx_distance[positive_index] = ((BA[positive_index, 0] - BA_dot_unit_BC_times_unit_BC[positive_index, 0])**2+(BA[positive_index, 1] - BA_dot_unit_BC_times_unit_BC[positive_index, 1])**2)**0.5
        dx_distance[negative_index] = np.minimum(((CA[negative_index, 0])**2+(CA[negative_index, 1])**2)**0.5, ((BA[negative_index, 0])**2+(BA[negative_index, 1])**2)**0.5)
        
        if np.shape(zero_index)[0] != 0:
            dx_distance[zero_index] = np.minimum(((CA[zero_index, 0])**2+(CA[zero_index, 1])**2)**0.5, ((BA[zero_index, 0])**2+(BA[zero_index, 1])**2)**0.5)

        # sorted_dx_distance = np.sort(dx_distance)
        # if sorted_dx_distance[0] == sorted_dx_distance[1]:
        #     print('there is identicle minimum distance', sorted_dx_distance[0], sorted_dx_distance[1])
        
        min_distance = np.min(dx_distance)

        min_index = np.argmin(dx_distance)
        
        # H(d(x)), d(x) = ((x-x0)**2+(y-y0)**2)**0.5, need to find x0, y0
        if min_index in positive_index: # if the smallest distance is between AD, D in between BC
            x_coor_min_point_segment = BA_dot_unit_BC_times_unit_BC[min_index, 0] + BxByCxCy[min_index,0]
            y_coor_min_point_segment = BA_dot_unit_BC_times_unit_BC[min_index, 1] + BxByCxCy[min_index,1]
        if (min_index in negative_index) or (min_index in zero_index): # if the smallest distance is AB or AC
            if ((CA[min_index, 0])**2+(CA[min_index, 1])**2)**0.5 < ((BA[min_index, 0])**2+(BA[min_index, 1])**2)**0.5:
                x_coor_min_point_segment = BxByCxCy[min_index,2]
                y_coor_min_point_segment = BxByCxCy[min_index,3]
            else:
                x_coor_min_point_segment = BxByCxCy[min_index,0]
                y_coor_min_point_segment = BxByCxCy[min_index,1]

        d_distance_dx = (x_G[i,0]-x_coor_min_point_segment)/min_distance
        d_distance_dy = (x_G[i,1]-y_coor_min_point_segment)/min_distance

        
        if i not in saved_dist_func_index:
            saved_dist_func_index.append(i)
            save_point_D_coor.append([x_coor_min_point_segment, y_coor_min_point_segment])
            save_distance_function.append([i, x_G[i, 0], x_G[i, 1], min_distance])
            save_distance_function_dx.append([i, x_G[i, 0], x_G[i, 1], (x_G[i,0]-x_coor_min_point_segment)/min_distance])
            save_distance_function_dy.append([i, x_G[i, 0], x_G[i, 1], (x_G[i,1]-y_coor_min_point_segment)/min_distance])

        # """
        # discrete the segments to points, check distance between point to points
        # """

        # # dx_distance = ((x_G[i, :] - discreted_segments_points_coor)[:,0]**2+(x_G[i, :] - discreted_segments_points_coor)[:,1]**2)**0.5

        # # matlab_interface_points_refined = np.loadtxt('matlab_interface_points_refined.txt')

        # dx_distance = ((x_G[i, :] - discreted_segments_points_coor)[:,0]**2+(x_G[i, :] - discreted_segments_points_coor)[:,1]**2)**0.5
        # # print(np.shape(dx_distance))
        # # dx_distance = ((x_G[i, :] - matlab_interface_points_refined)[:,0]**2+(x_G[i, :] - matlab_interface_points_refined)[:,1]**2)**0.5
        
        # # find the two index of smallest value
        # firt_smallest_index = np.argmin(dx_distance)
        # second_smallest_index = np.argpartition(dx_distance, 2)[1]
        
        # # if firt_smallest_index == second_smallest_index:
        # #     print('same!')
        # #     np.savetxt('distance_array.txt', dx_distance)
            

        # if abs(dx_distance[firt_smallest_index]-dx_distance[second_smallest_index])<1.0e-20:
        #     if discreted_segments_points_coor[firt_smallest_index, 0] < discreted_segments_points_coor[second_smallest_index, 0]:
        
        #         min_distance = dx_distance[firt_smallest_index]

        #         min_index = firt_smallest_index
        #     else:
        #         min_distance = dx_distance[second_smallest_index]

        #         min_index = second_smallest_index
        # else:
        #     min_distance = dx_distance[firt_smallest_index]

        #     min_index = firt_smallest_index


        # x_coor_min_point_segment = discreted_segments_points_coor[min_index, 0]

        # y_coor_min_point_segment = discreted_segments_points_coor[min_index, 1]

        # d_distance_dx = (x_G[i,0]-x_coor_min_point_segment)/min_distance
        # d_distance_dy = (x_G[i,1]-y_coor_min_point_segment)/min_distance

        # # x_coor_min_point_segment = matlab_interface_points_refined[min_index, 0]

        # # y_coor_min_point_segment = matlab_interface_points_refined[min_index, 1]
        
        # if i not in saved_dist_func_index:
        #     saved_dist_func_index.append(i)
        #     save_point_D_coor.append([x_coor_min_point_segment, y_coor_min_point_segment])
        #     save_distance_function.append([i, x_G[i, 0], x_G[i, 1], min_distance])
        #     save_distance_function_dx.append([i, x_G[i, 0], x_G[i, 1], (x_G[i,0]-x_coor_min_point_segment)/min_distance])
        #     save_distance_function_dy.append([i, x_G[i, 0], x_G[i, 1], (x_G[i,1]-y_coor_min_point_segment)/min_distance])
        
        
        # modify the kernal function related to nodes within domain

        heaviside_scaling_factor = 4.0e-7

        heaviside = np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor)

        # heaviside = np.tanh((min_distance)/heaviside_scaling_factor)

        heaviside_P_x = d_distance_dx/heaviside_scaling_factor*(1.0/np.cosh((min_distance+1.0e-15)/heaviside_scaling_factor))**2#(1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)
        heaviside_P_y = d_distance_dy/heaviside_scaling_factor*(1.0/np.cosh((min_distance+1.0e-15)/heaviside_scaling_factor))**2#(1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)
        
        # heaviside_P_x = d_distance_dx/heaviside_scaling_factor*(1.0/np.cosh((min_distance)/heaviside_scaling_factor))**2#(1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)
        # heaviside_P_y = d_distance_dy/heaviside_scaling_factor*(1.0/np.cosh((min_distance)/heaviside_scaling_factor))**2#(1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)
        
        save_heavyside.append(heaviside)
        save_heavyside_px.append(heaviside_P_x)
        save_heavyside_py.append(heaviside_P_y)

        
        
        for j in range(np.shape(x_nodes)[0]):

            z_ij = (((x_G[i,0]-x_nodes[j,0])**2+(x_G[i,1]-x_nodes[j,1])**2)**0.5)/a[j]
            z_ij_P_x = (x_G[i,0]-x_nodes[j,0])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial x, add the small number to force the term with machine accuracy
            z_ij_P_y = (x_G[i,1]-x_nodes[j,1])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial y

            x_I = x_nodes[j]

            H_sacling_factor = 1.0e-6

            H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
            H = np.transpose(H_T)

            HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
            HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

            H_P_x = np.transpose(HT_P_x)
            H_P_y = np.transpose(HT_P_y)

            if z_ij >= 0 and z_ij < 0.5:
                
                phi_ij = 2.0/3-4*z_ij**2+4*z_ij**3
                phi_P_z = -8.0*z_ij+12.0*z_ij**2                       # partial phi partial z
            else:
                if z_ij<=1 and z_ij>=0.5:
                    phi_ij = 4.0/3-4*z_ij+4*z_ij**2-4.0/3*z_ij**3
                    phi_P_z = -4+8*z_ij-4*z_ij**2

            if z_ij >= 0 and z_ij <= 1.0:
                # print('yes')
                # phi_nonzerovalue_data.append(phi_ij)

                node_not_on_interface = 'True'

                for i_nodes in range(num_interface_segments*2):
                    # print('yyy')
                    if abs(x_nodes[j,0] - interface_nodes[i_nodes, 0])<1e-10 and abs(x_nodes[j,1] - interface_nodes[i_nodes, 1])<1e-10:
                        node_not_on_interface = 'False'
                
                if IM_RKPM == 'True' and node_not_on_interface == 'True':
                    if nodes_grain_id[j] == Gauss_grain_id[i]:

                        phi_nonzero_index_row.append(i)
                        phi_nonzero_index_column.append(j)
                        phi_nonzerovalue_data.append(phi_ij*heaviside)

                        phi_P_x_ij = phi_P_z*z_ij_P_x
                        phi_P_y_ij = phi_P_z*z_ij_P_y
                        phi_P_x_nonzerovalue_data.append(phi_P_x_ij*heaviside+phi_ij*heaviside_P_x)    # partial phi partial x
                        phi_P_y_nonzerovalue_data.append(phi_P_y_ij*heaviside+phi_ij*heaviside_P_y)    # partial phi partial y

                        z.append(z_ij)
                        z_P_x.append(z_ij_P_x)
                        z_P_y.append(z_ij_P_y)
                        phipz.append(phi_P_z)
                        for ii in range(3):
                            for jj in range(3):
                                M[i][ii][jj] = M[i][ii][jj] + H[ii]*H_T[jj]*phi_ij*heaviside
                                M_P_x[i][ii][jj] = M_P_x[i][ii][jj] + H[ii]*H_T[jj]*(phi_P_x_ij*heaviside+phi_ij*heaviside_P_x) + H_P_x[ii]*H_T[jj]*phi_ij*heaviside + H[ii]*HT_P_x[jj]*phi_ij*heaviside
                                M_P_y[i][ii][jj] = M_P_y[i][ii][jj] + H[ii]*H_T[jj]*(phi_P_y_ij*heaviside+phi_ij*heaviside_P_y) + H_P_y[ii]*H_T[jj]*phi_ij*heaviside + H[ii]*HT_P_y[jj]*phi_ij*heaviside
                    
                else:
                    phi_nonzerovalue_data.append(phi_ij)
                    phi_nonzero_index_row.append(i)
                    phi_nonzero_index_column.append(j)
                    phi_P_x_ij = phi_P_z*z_ij_P_x
                    phi_P_y_ij = phi_P_z*z_ij_P_y
                    phi_P_x_nonzerovalue_data.append(phi_P_x_ij)    # partial phi partial x
                    phi_P_y_nonzerovalue_data.append(phi_P_y_ij)    # partial phi partial y
                    z.append(z_ij)
                    z_P_x.append(z_ij_P_x)
                    z_P_y.append(z_ij_P_y)
                    phipz.append(phi_P_z)
                    for ii in range(3):
                        for jj in range(3):
                            M[i][ii][jj] = M[i][ii][jj] + H[ii]*H_T[jj]*phi_ij
                            M_P_x[i][ii][jj] = M_P_x[i][ii][jj] + H[ii]*H_T[jj]*phi_P_x_ij + H_P_x[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_x[jj]*phi_ij
                            M_P_y[i][ii][jj] = M_P_y[i][ii][jj] + H[ii]*H_T[jj]*phi_P_y_ij + H_P_y[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_y[jj]*phi_ij
                
    return save_distance_function, save_distance_function_dx, save_distance_function_dy, save_point_D_coor, save_heavyside, save_heavyside_px, save_heavyside_py, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y
    
    

def compute_phi_M(x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM):
    """
    Fully cuNumeric vectorized implementation of compute_phi_M.
    """
    # Convert inputs to cuNumeric arrays
    x_G = cnp.asarray(x_G)
    x_nodes = cnp.asarray(x_nodes)
    a = cnp.asarray(a)
    nodes_grain_id = cnp.asarray(nodes_grain_id)
    Gauss_grain_id = cnp.asarray(Gauss_grain_id)
    interface_nodes_cnp = cnp.asarray(interface_nodes)
    seg = cnp.asarray(BxByCxCy)

    M = cnp.asarray(M)
    M_P_x = cnp.asarray(M_P_x)
    M_P_y = cnp.asarray(M_P_y)

    n_G = x_G.shape[0]
    n_N = x_nodes.shape[0]

    # Normalize BxByCxCy to endpoints B (N,2) and C (N,2)
    if seg.ndim == 2 and seg.shape[1] == 4:
        B = seg[:, :2]
        C = seg[:, 2:4]
    elif seg.ndim == 3 and seg.shape[1] >= 2 and seg.shape[2] == 2:
        B = seg[:, 0, :]
        C = seg[:, 1, :]
    elif seg.ndim == 3 and seg.shape[1] == 1 and seg.shape[2] == 2:
        # Degenerate: only one point per segment; treat as point-segment (B=C)
        B = seg[:, 0, :]
        C = B
    elif seg.ndim == 2 and seg.shape[1] == 2:
        # Degenerate: provided as points; treat as B=C
        B = seg
        C = seg
    else:
        raise ValueError(f"Unexpected BxByCxCy shape: {tuple(seg.shape)}")

    # Distances from Gauss points to line segments
    BC = C - B
    CB = -BC
    BA = x_G[:, None, :] - B[None, :, :]
    CA = x_G[:, None, :] - C[None, :, :]

    BA_dot_BC = cnp.sum(BA * BC[None, :, :], axis=2)
    CA_dot_CB = cnp.sum(CA * CB[None, :, :], axis=2)
    sign_extension = BA_dot_BC * CA_dot_CB

    BC_norm = cnp.sqrt(cnp.sum(BC * BC, axis=1))
    BC_norm_safe = BC_norm + 1e-16

    BA_dot_unit_BC = BA_dot_BC / BC_norm_safe[None, :]
    BC_unit = BC[None, :, :] / BC_norm_safe[None, :, None]
    proj = BC_unit * BA_dot_unit_BC[:, :, None]

    dist_perp = cnp.sqrt(cnp.sum((BA - proj) ** 2, axis=2))
    dist_CA = cnp.sqrt(cnp.sum(CA ** 2, axis=2))
    dist_BA = cnp.sqrt(cnp.sum(BA ** 2, axis=2))
    dist_end = cnp.minimum(dist_CA, dist_BA)

    dx_distance = cnp.where(sign_extension > 0, dist_perp, dist_end)

    row_idx = cnp.arange(n_G)
    min_index = cnp.argmin(dx_distance, axis=1)
    min_distance = dx_distance[row_idx, min_index]

    pos_mask = sign_extension[row_idx, min_index] > 0
    x_proj_pos = proj[row_idx, min_index, :] + B[min_index, :]
    dist_CA_val = dist_CA[row_idx, min_index]
    dist_BA_val = dist_BA[row_idx, min_index]
    use_C = dist_CA_val < dist_BA_val
    x_proj_negzero = cnp.where(use_C[:, None], C[min_index, :], B[min_index, :])
    x_min = cnp.where(pos_mask[:, None], x_proj_pos, x_proj_negzero)

    min_distance_safe = min_distance + 1e-16
    d_distance_dx = (x_G[:, 0] - x_min[:, 0]) / min_distance_safe
    d_distance_dy = (x_G[:, 1] - x_min[:, 1]) / min_distance_safe

    # Smooth Heaviside
    heaviside_scaling_factor = 4.0e-7
    t = (min_distance + 1.0e-15) / heaviside_scaling_factor
    heaviside = cnp.tanh(t)
    sech2 = (1.0 / cnp.cosh(t)) ** 2
    heaviside_P_x = d_distance_dx / heaviside_scaling_factor * sech2
    heaviside_P_y = d_distance_dy / heaviside_scaling_factor * sech2

    # Pairwise Gauss-node deltas
    dx = x_G[:, None, 0] - x_nodes[None, :, 0]
    dy = x_G[:, None, 1] - x_nodes[None, :, 1]
    dist_to_node = cnp.sqrt(dx * dx + dy * dy)
    z_ij = dist_to_node / (a[None, :] + 1e-16)

    denom = a[None, :] * z_ij * a[None, :] + 2.220446049250313e-16
    z_ij_P_x = dx / denom
    z_ij_P_y = dy / denom

    # Polynomial basis H
    H_scaling_factor = 1.0e-6
    H_T = cnp.zeros((n_G, n_N, 3), dtype=cnp.float64)
    H_T[:, :, 0] = 1.0
    H_T[:, :, 1] = dx / H_scaling_factor
    H_T[:, :, 2] = dy / H_scaling_factor
    H = H_T

    HT_P_x = cnp.array([0.0, 1.0 / H_scaling_factor, 0.0], dtype=cnp.float64)
    HT_P_y = cnp.array([0.0, 0.0, 1.0 / H_scaling_factor], dtype=cnp.float64)

    # Shape function and derivative wrt z
    mask_01 = (z_ij >= 0) & (z_ij < 0.5)
    mask_051 = (z_ij >= 0.5) & (z_ij <= 1.0)
    mask_in_support = (z_ij >= 0) & (z_ij <= 1.0)

    phi = cnp.zeros_like(z_ij)
    phi_P_z = cnp.zeros_like(z_ij)

    z01 = z_ij[mask_01]
    phi[mask_01] = 2.0 / 3 - 4 * z01 ** 2 + 4 * z01 ** 3
    phi_P_z[mask_01] = -8.0 * z01 + 12.0 * z01 ** 2

    z051 = z_ij[mask_051]
    phi[mask_051] = 4.0 / 3 - 4 * z051 + 4 * z051 ** 2 - (4.0 / 3) * z051 ** 3
    phi_P_z[mask_051] = -4.0 + 8.0 * z051 - 4.0 * z051 ** 2

    # Interface node exclusion and grain matching
    tol = 1e-10
    #x_nodes_np = np.asarray(x_nodes)
    #interface_nodes_np = np.asarray(interface_nodes)
    node_on_interface = cnp.any(
        (cnp.abs(x_nodes[:, None, 0] - interface_nodes[None, :, 0]) < tol)
        & (cnp.abs(x_nodes[:, None, 1] - interface_nodes[None, :, 1]) < tol),
        axis=1,
    )
    #node_on_interface = cnp.asarray(node_on_interface_np)
    grainid_match = (nodes_grain_id[None, :] == Gauss_grain_id[:, None])

    if IM_RKPM == 'True':
        valid_mask = mask_in_support & (~node_on_interface[None, :]) & grainid_match
        heav = heaviside[:, None]
        heav_px = heaviside_P_x[:, None]
        heav_py = heaviside_P_y[:, None]
    else:
        valid_mask = mask_in_support
        heav = cnp.ones((n_G, 1), dtype=cnp.float64)
        heav_px = cnp.zeros((n_G, 1), dtype=cnp.float64)
        heav_py = cnp.zeros((n_G, 1), dtype=cnp.float64)

    phi_final = cnp.where(valid_mask, phi * heav, 0.0)
    phi_P_x = phi_P_z * z_ij_P_x
    phi_P_y = phi_P_z * z_ij_P_y

    if IM_RKPM == 'True':
        phi_P_x_final = cnp.where(valid_mask, phi_P_x * heav + phi * heav_px, 0.0)
        phi_P_y_final = cnp.where(valid_mask, phi_P_y * heav + phi * heav_py, 0.0)
    else:
        phi_P_x_final = cnp.where(valid_mask, phi_P_x, 0.0)
        phi_P_y_final = cnp.where(valid_mask, phi_P_y, 0.0)

    # Non-zero entries
    nz_i, nz_j = cnp.where(valid_mask)

    phi_nonzero_index_row = nz_i.tolist()
    phi_nonzero_index_column = nz_j.tolist()
    phi_nonzerovalue_data = phi_final[valid_mask].tolist()
    phi_P_x_nonzerovalue_data = phi_P_x_final[valid_mask].tolist()
    phi_P_y_nonzerovalue_data = phi_P_y_final[valid_mask].tolist()

    # Moment matrices updates via scatter-add
    H_sel = H[nz_i, nz_j]
    HT_sel = H_T[nz_i, nz_j]

    outer = H_sel[:, :, None] * HT_sel[:, None, :]
    HT_P_x_b = cnp.broadcast_to(HT_P_x[None, :], HT_sel.shape)
    HT_P_y_b = cnp.broadcast_to(HT_P_y[None, :], HT_sel.shape)

    outer_P_x_1 = HT_P_x_b[:, :, None] * HT_sel[:, None, :]
    outer_P_x_2 = H_sel[:, :, None] * HT_P_x_b[:, None, :]
    outer_P_y_1 = HT_P_y_b[:, :, None] * HT_sel[:, None, :]
    outer_P_y_2 = H_sel[:, :, None] * HT_P_y_b[:, None, :]

    phi_v = phi_final[valid_mask]
    phi_px_v = phi_P_x_final[valid_mask]
    phi_py_v = phi_P_y_final[valid_mask]

    M_updates = outer * phi_v[:, None, None]
    M_P_x_updates = outer * phi_px_v[:, None, None] + (outer_P_x_1 + outer_P_x_2) * phi_v[:, None, None]
    M_P_y_updates = outer * phi_py_v[:, None, None] + (outer_P_y_1 + outer_P_y_2) * phi_v[:, None, None]

    cnp.add.at(M, nz_i, M_updates)
    cnp.add.at(M_P_x, nz_i, M_P_x_updates)
    cnp.add.at(M_P_y, nz_i, M_P_y_updates)

    # Save distance-related outputs as lists
    idx_arr = cnp.arange(n_G)
    save_distance_function = cnp.stack((idx_arr.astype(cnp.float64), x_G[:, 0], x_G[:, 1], min_distance), axis=1).tolist()
    save_distance_function_dx = cnp.stack((idx_arr.astype(cnp.float64), x_G[:, 0], x_G[:, 1], d_distance_dx), axis=1).tolist()
    save_distance_function_dy = cnp.stack((idx_arr.astype(cnp.float64), x_G[:, 0], x_G[:, 1], d_distance_dy), axis=1).tolist()
    save_point_D_coor = x_min.tolist()

    save_heavyside = heaviside.tolist()
    save_heavyside_px = heaviside_P_x.tolist()
    save_heavyside_py = heaviside_P_y.tolist()

    return save_distance_function, save_distance_function_dx, save_distance_function_dy, save_point_D_coor, save_heavyside, save_heavyside_px, save_heavyside_py, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data, phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y

# @jit  # this is taking so long time, we are vectorizing this part
def shape_grad_shape_func(x_G,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight, IM_RKPM):
    # shape_func_value = []
    # shape_func_times_det_J_time_weight_value = []
    # grad_shape_func_x_value = []
    # grad_shape_func_y_value = []
    # grad_shape_func_x_times_det_J_time_weight_value = []
    # grad_shape_func_y_times_det_J_time_weight_value = []
    # for ii in range(num_non_zero_phi_a):
    #     i = phi_nonzero_index_row[ii]
    #     j = phi_nonzero_index_column[ii]
            
    #     # compute the shape function and the gradient of shape function
    #     x_I = x_nodes[j]

    #     H_sacling_factor = 1.0e-6

    #     H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
    #     H = np.transpose(H_T)

    #     HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
    #     HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

    #     H_P_x = np.transpose(HT_P_x)
    #     H_P_y = np.transpose(HT_P_y)
        
    #     shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
        
    #     if differential_method =='implicite' and IM_RKPM == 'False':
    #         grad_shape_func_x_ij = np.dot((np.dot((HT1).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
    #         grad_shape_func_y_ij = np.dot((np.dot((HT2).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]

    #     else:
    #         if differential_method =='direct' or IM_RKPM == 'True':
    #             M_inv_P_x_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_x[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
    #             M_inv_P_y_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_y[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
    #             grad_shape_func_x_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_x_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_x_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_x.astype(np.float64))*phi_nonzerovalue_data[ii]
    #             grad_shape_func_y_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_y_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_y_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_y.astype(np.float64))*phi_nonzerovalue_data[ii]
    #         else:
    #             print('differential method is not defined')
    #     shape_func_value.append(shape_func_ij)
    #     grad_shape_func_x_value.append(grad_shape_func_x_ij)
    #     grad_shape_func_y_value.append(grad_shape_func_y_ij)

    #     shape_func_times_det_J_time_weight_value.append(shape_func_ij*det_J_time_weight[i])
    #     grad_shape_func_x_times_det_J_time_weight_value.append(grad_shape_func_x_ij*det_J_time_weight[i])
    #     grad_shape_func_y_times_det_J_time_weight_value.append(grad_shape_func_y_ij*det_J_time_weight[i])

    # return shape_func_value, shape_func_times_det_J_time_weight_value, grad_shape_func_x_value, grad_shape_func_y_value, grad_shape_func_x_times_det_J_time_weight_value, grad_shape_func_y_times_det_J_time_weight_value
    
    # def shape_grad_shape_func_vectorized(x_G, x_nodes, num_non_zero_phi_a, HT0, M, M_P_x, M_P_y, 
    #                                 differential_method, HT1, HT2, phi_nonzerovalue_data, 
    #                                 phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, 
    #                                 phi_nonzero_index_row, phi_nonzero_index_column, 
    #                                 det_J_time_weight, IM_RKPM):
    
    # Convert inputs to numpy arrays for vectorization
    phi_nonzero_index_row = np.array(phi_nonzero_index_row)
    phi_nonzero_index_column = np.array(phi_nonzero_index_column)
    phi_nonzerovalue_data = np.array(phi_nonzerovalue_data)
    phi_P_x_nonzerovalue_data = np.array(phi_P_x_nonzerovalue_data)
    phi_P_y_nonzerovalue_data = np.array(phi_P_y_nonzerovalue_data)
    det_J_time_weight = np.array(det_J_time_weight)
    HT0 = np.array(HT0, dtype=np.float64)
    x_G = np.array(x_G)
    x_nodes = np.array(x_nodes)
    
    # Get unique Gauss point indices and corresponding data
    i_indices = phi_nonzero_index_row
    j_indices = phi_nonzero_index_column
    
    # Vectorized computation of H matrices
    H_scaling_factor = 1.0e-6
    x_I = x_nodes[j_indices]  # Shape: (num_non_zero_phi_a, 2)
    x_G_selected = x_G[i_indices]  # Shape: (num_non_zero_phi_a, 2)
    
    # Compute H_T for all entries at once
    H_T_vectorized = np.zeros((num_non_zero_phi_a, 3), dtype=np.float64)
    H_T_vectorized[:, 0] = 1.0
    H_T_vectorized[:, 1] = (x_G_selected[:, 0] - x_I[:, 0]) / H_scaling_factor
    H_T_vectorized[:, 2] = (x_G_selected[:, 1] - x_I[:, 1]) / H_scaling_factor
    
    # H is transpose of H_T
    H_vectorized = H_T_vectorized  # Shape: (num_non_zero_phi_a, 3)
    
    # Vectorized computation of H derivatives
    HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
    HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
    H_P_x = HT_P_x
    H_P_y = HT_P_y
    
    # Get M matrices for all relevant Gauss points
    M_selected = M[i_indices]  # Shape: (num_non_zero_phi_a, 3, 3)
    # Regularize to avoid singular matrices
    M_selected_np = M_selected.astype(np.float64)
    reg_eps = 1e-12
    eye3 = np.eye(3, dtype=np.float64)
    M_selected_reg = M_selected_np + eye3[None, :, :] * reg_eps
    M_inv_selected = np.linalg.inv(M_selected_reg)  # Shape: (num_non_zero_phi_a, 3, 3)
    
    # Vectorized shape function computation
    # HT0 @ M_inv @ H for all entries
    HT0_M_inv = np.dot(HT0, M_inv_selected.transpose((1, 2, 0))).T  # Shape: (num_non_zero_phi_a, 3)
    shape_func_values = np.sum(HT0_M_inv * H_vectorized, axis=1) * phi_nonzerovalue_data
    
    # Vectorized gradient computation
    if differential_method == 'implicite' and IM_RKPM == 'False':
        HT1 = np.array(HT1, dtype=np.float64)
        HT2 = np.array(HT2, dtype=np.float64)
        
        HT1_M_inv = np.dot(HT1, M_inv_selected.transpose((1, 2, 0))).T
        HT2_M_inv = np.dot(HT2, M_inv_selected.transpose((1, 2, 0))).T
        
        grad_shape_func_x_values = np.sum(HT1_M_inv * H_vectorized, axis=1) * phi_nonzerovalue_data
        grad_shape_func_y_values = np.sum(HT2_M_inv * H_vectorized, axis=1) * phi_nonzerovalue_data
        
    else:  # differential_method == 'direct' or IM_RKPM == 'True'
        M_P_x_selected = M_P_x[i_indices]  # Shape: (num_non_zero_phi_a, 3, 3)
        M_P_y_selected = M_P_y[i_indices]  # Shape: (num_non_zero_phi_a, 3, 3)
        
        # Vectorized computation of M_inv derivatives
        # M_inv_P_x = -M_inv @ M_P_x @ M_inv
        M_inv_M_P_x = np.matmul(M_inv_selected, M_P_x_selected.astype(np.float64))
        M_inv_P_x_selected = -np.matmul(M_inv_M_P_x, M_inv_selected)
        
        M_inv_M_P_y = np.matmul(M_inv_selected, M_P_y_selected.astype(np.float64))
        M_inv_P_y_selected = -np.matmul(M_inv_M_P_y, M_inv_selected)
        
        # Vectorized gradient computation
        # Term 1: HT0 @ M_inv @ H * phi_P_x
        term1_x = np.sum(HT0_M_inv * H_vectorized, axis=1) * phi_P_x_nonzerovalue_data
        term1_y = np.sum(HT0_M_inv * H_vectorized, axis=1) * phi_P_y_nonzerovalue_data
        
        # Term 2: HT0 @ M_inv_P_x @ H * phi
        HT0_M_inv_P_x = np.dot(HT0, M_inv_P_x_selected.transpose((1, 2, 0))).T
        HT0_M_inv_P_y = np.dot(HT0, M_inv_P_y_selected.transpose((1, 2, 0))).T
        term2_x = np.sum(HT0_M_inv_P_x * H_vectorized, axis=1) * phi_nonzerovalue_data
        term2_y = np.sum(HT0_M_inv_P_y * H_vectorized, axis=1) * phi_nonzerovalue_data
        
        # Term 3: HT0 @ M_inv @ H_P_x * phi
        HT0_M_inv_H_P_x = np.dot(HT0_M_inv, H_P_x)
        HT0_M_inv_H_P_y = np.dot(HT0_M_inv, H_P_y)
        term3_x = HT0_M_inv_H_P_x * phi_nonzerovalue_data
        term3_y = HT0_M_inv_H_P_y * phi_nonzerovalue_data
        
        grad_shape_func_x_values = term1_x + term2_x + term3_x
        grad_shape_func_y_values = term1_y + term2_y + term3_y
    
    # Multiply by det_J_time_weight
    det_J_selected = det_J_time_weight[i_indices]
    shape_func_times_det_J_time_weight_values = shape_func_values * det_J_selected
    grad_shape_func_x_times_det_J_time_weight_values = grad_shape_func_x_values * det_J_selected
    grad_shape_func_y_times_det_J_time_weight_values = grad_shape_func_y_values * det_J_selected
    
    # Convert to lists to match original output format
    return (
        shape_func_values.tolist(),
        shape_func_times_det_J_time_weight_values.tolist(),
        grad_shape_func_x_values.tolist(),
        grad_shape_func_y_values.tolist(),
        grad_shape_func_x_times_det_J_time_weight_values.tolist(),
        grad_shape_func_y_times_det_J_time_weight_values.tolist()
    )


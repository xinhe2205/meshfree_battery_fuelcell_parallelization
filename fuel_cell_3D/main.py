import time
start_time = time.time()
import numpy as np
from numpy import sign

import scipy.sparse as sp

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit, njit, typed
import scipy.sparse as sp

from scipy.sparse import csc_matrix, csr_matrix, bmat, block_diag,vstack
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig


from get_nodes_gauss_points import x_G_and_def_J_time_weight_3d_fuelcell_domain, x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary,x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface, x_G_and_det_J_line_3d_fuelcell_1d_boundary, get_x_nodes_fuel_cell_3d_toy_image

from shape_function_in_domain import compute_phi_M, shape_grad_shape_func, shape_func_n_nodes_by_n_nodes

from define_mechanical_stiffness_matrix import mechanical_stiffness_matrix_3d_fuel_cell, mechanical_force_matrix_3d, mechanical_C_tensor_3d
from define_diffusion_matrix_form import diffusion_matrix_fuel_cell, diffusion_matrix_fuel_cell_distributed_point_source

from read_image import read_in_image

from scipy.sparse.linalg import inv

###################################
# define geometry and analysis type 
###################################
print('define geometry and analysis types')

single_grain = 'False'   # True: single grain, False: read from an image
dimention = 3   # 3d or 2d

IM_RKPM = 'False'  # if it is interfacial modified RKPM, only available for battery
studied_physics = "fuel cell"  # fuel cell or battery
damage_model = 'OFF'   # ON or OFF

delta_point_source = 'True' # if point source is delta function. If the point source is distributed, set it to be 'False'

#########################################
# define differential and integral method
#########################################
differential_method = 'direct'    # 'implicite' or 'direct'    # specify which differential method to use, implicite: H1, H2, direct: directly differentiate
# if the IM_RKPM=True, differential method must be set to be direct.
integral_method = 'gauss'

################
# Define domain 
################
print('Define domain and parameters')

if studied_physics == 'fuel cell':
    if dimention == 3:
        x_min = 0
        x_max = 10e-6
        y_min = 0
        y_max = 20e-6
        z_min = 0 
        z_max = 10e-6

##############
# grain angle
##############
print('define grain angle')
angle = 0

###############################
# Define material properties
###############################
    
Fday = 9.6485e4     # Faraday constant
R = 8.3145e0        # gas constant

# for damage
k_i = 0.0125
k_f = 0.015


diffusion_electrolyte = 0.035
diffusion_electrode = 1.0e-9
diffusion_pore = 1.0e-7

k_gas = 1.0e-4

i_0 = 1.0e-1
i_0_solid = 1.0e1

T = 1273.2

E_0 = 1.0

V_app = 1.5

c_boundary = 1000.0

c_boundary_pore = 9.572

# for mechanical
E_electrolyte = 132.69e9            # Youngs modulus (Pa)
nu_electrolyte = 0.33                # Poisson ratio
lambda_mechanical_electrolyte = E_electrolyte*nu_electrolyte/(1+nu_electrolyte)/(1-2*nu_electrolyte)
mu_electrolyte = E_electrolyte/2/(1+nu_electrolyte)         # lamme constants

E_electrode = 130.0e9            # Youngs modulus (Pa)
nu_electrode = 0.33                # Poisson ratio
lambda_mechanical_electrode = E_electrode*nu_electrode/(1+nu_electrode)/(1-2*nu_electrode)
mu_electrode = E_electrode/2/(1+nu_electrode)         # lamme constants

beta_fuelcell_expansion_coefficient = 4.0e-6 # m^3/mol


######################
# Gauss integral
######################
    
if integral_method == 'gauss':
# Define Guass int points and weights

# 3d cube:
    x_G_cube = [[-3**0.5/3, -3**0.5/3, -3**0.5/3],[3**0.5/3, -3**0.5/3, -3**0.5/3],\
                        [-3**0.5/3, 3**0.5/3, -3**0.5/3],[3**0.5/3, 3**0.5/3, -3**0.5/3],\
                        [-3**0.5/3, -3**0.5/3, 3**0.5/3],[3**0.5/3, -3**0.5/3, 3**0.5/3],\
                        [-3**0.5/3, 3**0.5/3, 3**0.5/3],[3**0.5/3, 3**0.5/3, 3**0.5/3]] # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    weight_G_cube = [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]         # weight of each 2D Gauss points for rectangular

# 2d rectangle or triangle
    x_G_rec = [[-3**0.5/3, -3**0.5/3],[-3**0.5/3, 3**0.5/3],[3**0.5/3, -3**0.5/3],[3**0.5/3, 3**0.5/3]] # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    x_G_tri = [[1.0/6.0, 2.0/3.0],[1.0/6.0,1.0/6.0],[2.0/3.0, 1.0/6.0]]
    weight_G_rec = [1.0,1.0,1.0,1.0]         # weight of each 2D Gauss points for rectangular
    weight_G_tri = [1.0/3.0, 1.0/3.0, 1.0/3.0]

# 1d line
    x_G_line = [-(3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5, -(3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, (3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, (3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5]#[-0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0,0.4058451513773972,0.7415311855993945,0.9491079123427585]#                   # coordinates of 1D Gauss points
    weight_G_line = [0.5-30**0.5/36, 0.5+30**0.5/36, 0.5+30**0.5/36, 0.5-30**0.5/36]#[0.1294849661688697,0.2797053914892766,0.3818300505051189,0.4179591836734694,0.3818300505051189,0.2797053914892766,0.1294849661688697]#         # weight of each 1D Gauss points
        
def_para_time = time.time()

print('time to define parameters = ' + "%s seconds" % (def_para_time - start_time))

##################
# define RK nodes 
##################
print('define RK nodes')


if studied_physics == "fuel cell":

    if single_grain == 'False':

        if dimention == 3:
            # file_name = 'micro_3d_connected.tif'#'M_3d_3phases_simple.tif'# real geometry
            file_name = 'M_3d_3phases_simple.tif' # simple geometry
            img_, unic_grain_id, num_pixels_xyz = read_in_image(file_name, studied_physics, dimention)

            x_nodes_mechanical, x_nodes_electrolyte, x_nodes_electrode, x_nodes_pore, segments_source_coords, cell_nodes_fixed_x, cell_nodes_fixed_z, nodes_id_left_electrolyte, nodes_id_right_electrode, nodes_id_right_pore, cell_nodes_electrolyte_x, cell_nodes_electrolyte_y, cell_nodes_electrolyte_z, cell_nodes_electrode_x, cell_nodes_electrode_y, cell_nodes_electrode_z,cell_nodes_pore_x,cell_nodes_pore_y,cell_nodes_pore_z, cell_nodes_left_electrolyte_x, cell_nodes_left_electrolyte_z, cell_nodes_right_electrode_x, cell_nodes_right_electrode_z, cell_nodes_right_pore_x, cell_nodes_right_pore_z,\
            cell_nodes_interface_electrode_electrolyte_x,cell_nodes_interface_electrode_electrolyte_y,cell_nodes_interface_electrode_electrolyte_z,\
            cell_nodes_interface_electrode_pore_x, cell_nodes_interface_electrode_pore_y, cell_nodes_interface_electrode_pore_z = get_x_nodes_fuel_cell_3d_toy_image(x_min,x_max,y_min,y_max,z_min, z_max, num_pixels_xyz, img_)
            
            x_nodes_mechanical, x_nodes_electrolyte, x_nodes_electrode, x_nodes_pore, segments_source_coords, cell_nodes_fixed_x, cell_nodes_fixed_z, nodes_id_left_electrolyte, nodes_id_right_electrode, nodes_id_right_pore, cell_nodes_electrolyte_x, cell_nodes_electrolyte_y, cell_nodes_electrolyte_z, cell_nodes_electrode_x, cell_nodes_electrode_y, cell_nodes_electrode_z,cell_nodes_pore_x,cell_nodes_pore_y,cell_nodes_pore_z, cell_nodes_left_electrolyte_x, cell_nodes_left_electrolyte_z, cell_nodes_right_electrode_x, cell_nodes_right_electrode_z, cell_nodes_right_pore_x, cell_nodes_right_pore_z,\
            cell_nodes_interface_electrode_electrolyte_x,cell_nodes_interface_electrode_electrolyte_y,cell_nodes_interface_electrode_electrolyte_z,\
            cell_nodes_interface_electrode_pore_x, cell_nodes_interface_electrode_pore_y, cell_nodes_interface_electrode_pore_z= [np.asarray(lst) for lst in [x_nodes_mechanical, x_nodes_electrolyte, x_nodes_electrode, x_nodes_pore, segments_source_coords, cell_nodes_fixed_x, cell_nodes_fixed_z, nodes_id_left_electrolyte, nodes_id_right_electrode, nodes_id_right_pore, cell_nodes_electrolyte_x, cell_nodes_electrolyte_y, cell_nodes_electrolyte_z, cell_nodes_electrode_x, cell_nodes_electrode_y, cell_nodes_electrode_z,cell_nodes_pore_x,cell_nodes_pore_y,cell_nodes_pore_z, cell_nodes_left_electrolyte_x, cell_nodes_left_electrolyte_z, cell_nodes_right_electrode_x, cell_nodes_right_electrode_z, cell_nodes_right_pore_x, cell_nodes_right_pore_z,\
            cell_nodes_interface_electrode_electrolyte_x,cell_nodes_interface_electrode_electrolyte_y,cell_nodes_interface_electrode_electrolyte_z,\
            cell_nodes_interface_electrode_pore_x, cell_nodes_interface_electrode_pore_y, cell_nodes_interface_electrode_pore_z]]   
            
            if delta_point_source == 'False':
                cell_nodes_distributed_point_source_surface_x = []
                cell_nodes_distributed_point_source_surface_y = []
                cell_nodes_distributed_point_source_surface_z = []
                # if 10 volxels on top right surface of electrolyte, 20% is used to distribute the point source, this is 2 cells
                for i_dis in range(2):
                    for j_dis in range(20):                    
                        cell_nodes_distributed_point_source_surface_x.append([x_min+(x_max-x_min)/(20)*j_dis, x_min+(x_max-x_min)/(20)*(j_dis+1), x_min+(x_max-x_min)/(20)*(j_dis+1), x_min+(x_max-x_min)/(20)*j_dis])
                        cell_nodes_distributed_point_source_surface_y.append([(y_max+y_min)/2, (y_max+y_min)/2, (y_max+y_min)/2, (y_max+y_min)/2])
                        cell_nodes_distributed_point_source_surface_z.append([(z_max+z_min)/2+(z_max-z_min)/20*i_dis, (z_max+z_min)/2+(z_max-z_min)/20*i_dis, (z_max+z_min)/2+(z_max-z_min)/20*(i_dis+1), (z_max+z_min)/2+(z_max-z_min)/20*(i_dis+1)])

            
    num_interface_segments = 0
    interface_nodes = np.zeros((1,1))
    BxByCxCy = np.zeros((1,1))

    num_nodes_electrolyte = np.shape(x_nodes_electrolyte)[0]
    num_nodes_electrode = np.shape(x_nodes_electrode)[0]
    num_nodes_pore = np.shape(x_nodes_pore)[0]
    num_nodes_mechanical = np.shape(x_nodes_mechanical)[0]

    nodes_grain_id_electrolyte = 1*np.ones(num_nodes_electrolyte)
    nodes_grain_id_electrode = 1*np.ones(num_nodes_electrode)
    nodes_grain_id_pore = 1*np.ones(num_nodes_pore)

    nodes_grain_id_mechanical = 1*np.ones(num_nodes_mechanical)

    print('number of nodes in electrolyte: ' + str(num_nodes_electrolyte))
    print('number of nodes in electrode: ' + str(num_nodes_electrode))
    print('number of nodes in pore: ' + str(num_nodes_pore))
    print('number of nodes in whole domain: ' + str(num_nodes_mechanical))

    print('number of cells in electrolyte: ' + str(np.shape(cell_nodes_electrolyte_x)[0]))
    print('number of cells in electrode: ' + str(np.shape(cell_nodes_electrode_x)[0]))
    print('number of cells in pore: ' + str(np.shape(cell_nodes_pore_x)[0]))
    # print('number of cells in whole domain: ' + str(np.shape(cell_nodes_mechanical_x)[0]))

# 9110 electrolyte cells
# 2939 electrode cells
# 3951 voids


##########################
# define gauss points
##########################
print('define gauss points')
# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian
if integral_method == 'gauss':

    if studied_physics == "fuel cell":
        
        if dimention == 3:
            x_G_electrolyte, det_J_time_weight_electrolyte = x_G_and_def_J_time_weight_3d_fuelcell_domain(cell_nodes_electrolyte_x,cell_nodes_electrolyte_y,cell_nodes_electrolyte_z,x_G_cube,weight_G_cube)
            x_G_electrode, det_J_time_weight_electrode = x_G_and_def_J_time_weight_3d_fuelcell_domain(cell_nodes_electrode_x,cell_nodes_electrode_y,cell_nodes_electrode_z,x_G_cube,weight_G_cube)
            x_G_pore, det_J_time_weight_pore = x_G_and_def_J_time_weight_3d_fuelcell_domain(cell_nodes_pore_x,cell_nodes_pore_y,cell_nodes_pore_z,x_G_cube,weight_G_cube)
            
            if single_grain == 'True':
                pass
            else:
                # on left boundary of electrolyte
                x_G_b_electrolyte, det_J_b_time_weight_electrolyte = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(cell_nodes_left_electrolyte_x,cell_nodes_left_electrolyte_z, y_min, x_G_rec, weight_G_rec)
                # on right boundary of electrode
                x_G_b_electrode, det_J_b_time_weight_electrode = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(cell_nodes_right_electrode_x,cell_nodes_right_electrode_z, y_max, x_G_rec, weight_G_rec)
                # on right boundary of pore
                x_G_b_pore, det_J_b_time_weight_pore = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(cell_nodes_right_pore_x,cell_nodes_right_pore_z, y_max, x_G_rec, weight_G_rec)
                # on electrolyte boundary and electrolyte/electrode interface
                x_G_b_interface_electrode_electrolyte, det_J_b_time_weight_interface_electrode_electrolyte = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_electrolyte_x,cell_nodes_interface_electrode_electrolyte_y, cell_nodes_interface_electrode_electrolyte_z, x_G_rec, weight_G_rec)
                # # on electrode boundary and electrolyte/electrode interface
                # x_G_b_interface_electrode_electrolyte_electrode, det_J_b_time_weight_interface_electrode_electrolyte_electrode = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_electrolyte_electrode_x,cell_nodes_interface_electrode_electrolyte_electrode_y, cell_nodes_interface_electrode_electrolyte_electrode_z, x_G_rec, weight_G_rec)
                # on electrode boundary and pore/electrode interface
                x_G_b_interface_electrode_pore, det_J_b_time_weight_interface_electrode_pore = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_pore_x,cell_nodes_interface_electrode_pore_y, cell_nodes_interface_electrode_pore_z, x_G_rec, weight_G_rec)
                # # on pore boundary and pore/electrode interface
                # x_G_b_interface_electrode_pore_pore, det_J_b_time_weight_interface_electrode_pore_pore = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_pore_pore_x,cell_nodes_interface_electrode_pore_pore_y, cell_nodes_interface_electrode_pore_pore_z, x_G_rec, weight_G_rec)
                if delta_point_source == 'False':
                    x_G_b_distributed_point_source_surface, det_J_b_time_weight_distributed_point_source_surface = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(np.asarray(cell_nodes_distributed_point_source_surface_x), np.asarray(cell_nodes_distributed_point_source_surface_y), np.asarray(cell_nodes_distributed_point_source_surface_z), x_G_rec, weight_G_rec)

            x_G_b_line, det_J_b_time_weight_line = x_G_and_det_J_line_3d_fuelcell_1d_boundary(segments_source_coords, x_G_line, weight_G_line)
            x_G_b_line = np.array(x_G_b_line) 
            num_source_line_gauss_points = np.shape(x_G_b_line)[0]
            det_J_b_time_weight_line = np.array(det_J_b_time_weight_line)

            x_G_b_fixed, det_J_b_time_weight_fixed = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(cell_nodes_fixed_x, cell_nodes_fixed_z, y_min, x_G_rec, weight_G_rec)
            x_G_b_fixed = np.array(x_G_b_fixed) 
            num_fixed_gauss_points = np.shape(x_G_b_fixed)[0]
            det_J_b_time_weight_fixed = np.array(det_J_b_time_weight_fixed)

            gauss_rotation_axis_electrolyte = np.zeros((len(x_G_electrolyte), 3))
            gauss_rotation_axis_electrode = np.zeros((len(x_G_electrode), 3))
            gauss_rotation_axis_electrolyte[:,0] = 1.0
            gauss_rotation_axis_electrode[:,0] = 1.0

            gauss_rotation_axis = np.concatenate((gauss_rotation_axis_electrolyte, gauss_rotation_axis_electrode), axis=0)

        num_gauss_points_in_domain_electrolyte = np.shape(x_G_electrolyte)[0]
        num_gauss_points_in_domain_electrode = np.shape(x_G_electrode)[0]
        num_gauss_points_in_domain_pore = np.shape(x_G_pore)[0]
        num_gauss_points_on_boundary_electrolyte = np.shape(x_G_b_electrolyte)[0]
        num_gauss_points_on_boundary_electrode = np.shape(x_G_b_electrode)[0]
        num_gauss_points_on_boundary_pore = np.shape(x_G_b_pore)[0]
        num_gauss_points_on_electrolyte_electrode_interface = np.shape(x_G_b_interface_electrode_electrolyte)[0]
        num_gauss_points_on_electrode_pore_interface = np.shape(x_G_b_interface_electrode_pore)[0]


        gauss_angle_electrolyte = angle*np.ones(num_gauss_points_in_domain_electrolyte)
        gauss_angle_electrode = angle*np.ones(num_gauss_points_in_domain_electrode)
        gauss_angle_pore = angle*np.ones(num_gauss_points_in_domain_pore)
        gauss_angle_b_electrolyte = angle*np.ones(num_gauss_points_on_boundary_electrolyte)
        gauss_angle_b_electrode = angle*np.ones(num_gauss_points_on_boundary_electrode)
        gauss_angle_b_pore = angle*np.ones(num_gauss_points_on_boundary_pore)
        gauss_angle_electrolyte_electrode_interface = angle*np.ones(num_gauss_points_on_electrolyte_electrode_interface)
        gauss_angle_electrode_pore_interface = angle*np.ones(num_gauss_points_on_electrode_pore_interface)


        Gauss_grain_id_electrolyte = 1*np.ones(num_gauss_points_in_domain_electrolyte)
        Gauss_grain_id_electrode = 1*np.ones(num_gauss_points_in_domain_electrode)
        Gauss_grain_id_pore = 1*np.ones(num_gauss_points_in_domain_pore)
        Gauss_b_grain_id_electrolyte = 1*np.ones(num_gauss_points_on_boundary_electrolyte)
        Gauss_b_grain_id_electrode = 1*np.ones(num_gauss_points_on_boundary_electrode)
        Gauss_b_grain_id_pore = 1*np.ones(num_gauss_points_on_boundary_pore)
        Gauss_b_grain_id_electrolyte_electrode_interace = 1*np.ones(num_gauss_points_on_electrolyte_electrode_interface)
        Gauss_b_grain_id_electrode_pore_interace = 1*np.ones(num_gauss_points_on_electrode_pore_interface)

        x_G_electrolyte,x_G_b_electrolyte,x_G_electrode,x_G_b_electrode,x_G_pore,x_G_b_pore,\
        x_G_b_interface_electrode_electrolyte,\
        x_G_b_interface_electrode_pore\
        = [np.array(lst) for lst in [x_G_electrolyte,x_G_b_electrolyte,x_G_electrode,x_G_b_electrode,x_G_pore,x_G_b_pore,\
            x_G_b_interface_electrode_electrolyte,\
        x_G_b_interface_electrode_pore]]

        gauss_angle_electrolyte,gauss_angle_b_electrolyte,gauss_angle_electrode,gauss_angle_b_electrode,gauss_angle_pore,gauss_angle_b_pore, gauss_angle_electrolyte_electrode_interface, gauss_angle_electrode_pore_interface = [np.array(lst) for lst in [gauss_angle_electrolyte,gauss_angle_b_electrolyte,gauss_angle_electrode,gauss_angle_b_electrode,gauss_angle_pore,gauss_angle_b_pore, gauss_angle_electrolyte_electrode_interface, gauss_angle_electrode_pore_interface]]
        Gauss_grain_id_electrolyte,Gauss_b_grain_id_electrolyte,Gauss_grain_id_electrode,Gauss_b_grain_id_electrode,Gauss_grain_id_pore,Gauss_b_grain_id_pore,Gauss_b_grain_id_electrolyte_electrode_interace,Gauss_b_grain_id_electrode_pore_interace = [np.array(lst) for lst in [Gauss_grain_id_electrolyte,Gauss_b_grain_id_electrolyte,Gauss_grain_id_electrode,Gauss_b_grain_id_electrode,Gauss_grain_id_pore,Gauss_b_grain_id_pore,Gauss_b_grain_id_electrolyte_electrode_interace,Gauss_b_grain_id_electrode_pore_interace]]

    
        # all gauss points in domain, used for mechanical simulation
        x_G_mechanical = np.concatenate((x_G_electrolyte, x_G_electrode), axis=0)
        det_J_time_weight_mechanical = np.concatenate((np.asarray(det_J_time_weight_electrolyte), np.asarray(det_J_time_weight_electrode)), axis=0)
        Gauss_grain_id_mechanical = np.concatenate((Gauss_grain_id_electrolyte, Gauss_grain_id_electrode), axis=0)
        Gauss_angle_mechanical = np.concatenate((gauss_angle_electrolyte, gauss_angle_electrode), axis=0)
        num_gauss_points_in_domain_mechanical = np.shape(x_G_mechanical)[0]

        print('number of Gauss points in electrolyte domain: ' + str(num_gauss_points_in_domain_electrolyte))
        print('number of Gauss points on electrolyte boundaries: ' + str(num_gauss_points_on_boundary_electrolyte))
        print('number of Gauss points in electrode domain: ' + str(num_gauss_points_in_domain_electrode))
        print('number of Gauss points on electrode boundaries: ' + str(num_gauss_points_on_boundary_electrode))
    
def_nodes_gauss_points_time = time.time()
print('time to define nodes and Gauss points = ' + "%s seconds" % (def_nodes_gauss_points_time-def_para_time))

####################################################
# Compute shape function and its gradient in domain
#####################################################
print('Compute shape function and its gradient in domain')

c = 2        # support size

if dimention == 2:
    HT0 = np.array([1,0,0],dtype=np.float64)     # transpose of the basis vector H
    HT1 = np.array([0,-1,0],dtype=np.float64)   # for computation of gradient of shape function, d/dx
    HT2 = np.array([0,0,-1],dtype=np.float64)   # for computation of gradient of shape function, d/dy
if dimention == 3:
    HT0 = np.array([1,0,0,0],dtype=np.float64)     # transpose of the basis vector H
    HT1 = np.array([0,-1,0,0],dtype=np.float64)   # for computation of gradient of shape function, d/dx
    HT2 = np.array([0,0,-1,0],dtype=np.float64)   # for computation of gradient of shape function, d/dy
    HT3 = np.array([0,0,0,-1],dtype=np.float64)   # for computation of gradient of shape function, d/dy

if studied_physics == "fuel cell" and single_grain == 'False':
    a_electrolyte = c*(x_max-x_min)/num_pixels_xyz[0]*np.ones(num_nodes_electrolyte)       # compact support size, shape: (num_nodes,)  
    a_electrode = c*(x_max-x_min)/num_pixels_xyz[0]*np.ones(num_nodes_electrode)
    a_pore = c*(x_max-x_min)/num_pixels_xyz[0]*np.ones(num_nodes_pore)
    a_mechanical = c*(x_max-x_min)/num_pixels_xyz[0]*np.ones(num_nodes_mechanical)




if dimention == 3:
    M_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrolyte)])
    M_P_x_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrolyte)]) # partial M partial x
    M_P_y_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrolyte)]) # partial M partial y
    M_P_z_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrolyte)]) # partial M partial y

    M_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrode)])
    M_P_x_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrode)]) # partial M partial x
    M_P_y_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrode)]) # partial M partial y
    M_P_z_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_electrolyte)]) # partial M partial y
    
    M_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_pore)])
    M_P_x_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_pore)]) # partial M partial x
    M_P_y_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_pore)]) # partial M partial y
    M_P_z_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_pore)]) # partial M partial y

    M_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_mechanical)])
    M_P_x_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_mechanical)]) # partial M partial x
    M_P_y_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_mechanical)]) # partial M partial y
    M_P_z_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_in_domain_mechanical)]) # partial M partial y
    
    phi_nonzero_index_row_electrolyte, phi_nonzero_index_column_electrolyte, phi_nonzerovalue_data_electrolyte, phi_P_x_nonzerovalue_data_electrolyte, phi_P_y_nonzerovalue_data_electrolyte,phi_P_z_nonzerovalue_data_electrolyte, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte,M_P_z_electrolyte = compute_phi_M(x_G_electrolyte, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte)
    phi_nonzero_index_row_electrode, phi_nonzero_index_column_electrode, phi_nonzerovalue_data_electrode, phi_P_x_nonzerovalue_data_electrode, phi_P_y_nonzerovalue_data_electrode,phi_P_z_nonzerovalue_data_electrode, M_electrode, M_P_x_electrode, M_P_y_electrode,M_P_z_electrode = compute_phi_M(x_G_electrode, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode, M_P_x_electrode, M_P_y_electrode, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode)
    phi_nonzero_index_row_pore, phi_nonzero_index_column_pore, phi_nonzerovalue_data_pore, phi_P_x_nonzerovalue_data_pore, phi_P_y_nonzerovalue_data_pore,phi_P_z_nonzerovalue_data_pore, M_pore, M_P_x_pore, M_P_y_pore,M_P_z_pore = compute_phi_M(x_G_pore, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore, M_P_x_pore, M_P_y_pore, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore)
    phi_nonzero_index_row_mechanical, phi_nonzero_index_column_mechanical, phi_nonzerovalue_data_mechanical, phi_P_x_nonzerovalue_data_mechanical, phi_P_y_nonzerovalue_data_mechanical,phi_P_z_nonzerovalue_data_mechanical, M_mechanical, M_P_x_mechanical, M_P_y_mechanical,M_P_z_mechanical = compute_phi_M(x_G_mechanical, Gauss_grain_id_mechanical, x_nodes_mechanical,nodes_grain_id_mechanical, a_mechanical, M_mechanical, M_P_x_mechanical, M_P_y_mechanical, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_mechanical)

    
num_non_zero_phi_a_electrolyte = np.shape(np.array(phi_nonzero_index_row_electrolyte))[0]
num_non_zero_phi_a_electrode = np.shape(np.array(phi_nonzero_index_row_electrode))[0]
num_non_zero_phi_a_pore = np.shape(np.array(phi_nonzero_index_row_pore))[0]
num_non_zero_phi_a_mechanical = np.shape(np.array(phi_nonzero_index_row_mechanical))[0]

    
    
if dimention == 3: 
    # print('yes')
    shape_func_value_electrolyte, shape_func_times_det_J_time_weight_value_electrolyte, grad_shape_func_x_value_electrolyte, grad_shape_func_y_value_electrolyte,grad_shape_func_z_value_electrolyte, grad_shape_func_x_times_det_J_time_weight_value_electrolyte, grad_shape_func_y_times_det_J_time_weight_value_electrolyte,grad_shape_func_z_times_det_J_time_weight_value_electrolyte = shape_grad_shape_func(x_G_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte,HT0, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte, differential_method, HT1, HT2, phi_nonzerovalue_data_electrolyte,phi_P_x_nonzerovalue_data_electrolyte,phi_P_y_nonzerovalue_data_electrolyte, phi_nonzero_index_row_electrolyte, phi_nonzero_index_column_electrolyte, det_J_time_weight_electrolyte, IM_RKPM, M_P_z_electrolyte, HT3, phi_P_z_nonzerovalue_data_electrolyte)
    shape_func_value_electrode, shape_func_times_det_J_time_weight_value_electrode, grad_shape_func_x_value_electrode,grad_shape_func_y_value_electrode, grad_shape_func_z_value_electrode, grad_shape_func_x_times_det_J_time_weight_value_electrode, grad_shape_func_y_times_det_J_time_weight_value_electrode, grad_shape_func_z_times_det_J_time_weight_value_electrode = shape_grad_shape_func(x_G_electrode,x_nodes_electrode, num_non_zero_phi_a_electrode,HT0, M_electrode, M_P_x_electrode, M_P_y_electrode, differential_method, HT1, HT2, phi_nonzerovalue_data_electrode,phi_P_x_nonzerovalue_data_electrode,phi_P_y_nonzerovalue_data_electrode, phi_nonzero_index_row_electrode, phi_nonzero_index_column_electrode, det_J_time_weight_electrode, IM_RKPM, M_P_z_electrode, HT3, phi_P_z_nonzerovalue_data_electrode)
    shape_func_value_pore, shape_func_times_det_J_time_weight_value_pore, grad_shape_func_x_value_pore,grad_shape_func_y_value_pore, grad_shape_func_z_value_pore, grad_shape_func_x_times_det_J_time_weight_value_pore, grad_shape_func_y_times_det_J_time_weight_value_pore, grad_shape_func_z_times_det_J_time_weight_value_pore = shape_grad_shape_func(x_G_pore,x_nodes_pore, num_non_zero_phi_a_pore,HT0, M_pore, M_P_x_pore, M_P_y_pore, differential_method, HT1, HT2, phi_nonzerovalue_data_pore,phi_P_x_nonzerovalue_data_pore,phi_P_y_nonzerovalue_data_pore, phi_nonzero_index_row_pore, phi_nonzero_index_column_pore, det_J_time_weight_pore, IM_RKPM, M_P_z_pore, HT3, phi_P_z_nonzerovalue_data_pore)
    shape_func_value_mechanical, shape_func_times_det_J_time_weight_value_mechanical, grad_shape_func_x_value_mechanical,grad_shape_func_y_value_mechanical, grad_shape_func_z_value_mechanical, grad_shape_func_x_times_det_J_time_weight_value_mechanical, grad_shape_func_y_times_det_J_time_weight_value_mechanical, grad_shape_func_z_times_det_J_time_weight_value_mechanical = shape_grad_shape_func(x_G_mechanical,x_nodes_mechanical, num_non_zero_phi_a_mechanical,HT0, M_mechanical, M_P_x_mechanical, M_P_y_mechanical, differential_method, HT1, HT2, phi_nonzerovalue_data_mechanical,phi_P_x_nonzerovalue_data_mechanical,phi_P_y_nonzerovalue_data_mechanical, phi_nonzero_index_row_mechanical, phi_nonzero_index_column_mechanical, det_J_time_weight_mechanical, IM_RKPM, M_P_z_mechanical, HT3, phi_P_z_nonzerovalue_data_mechanical)

# numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
shape_func_electrolyte = csc_matrix((np.array(shape_func_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
shape_func_times_det_J_time_weight_electrolyte = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
grad_shape_func_x_electrolyte = csc_matrix((np.array(grad_shape_func_x_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
grad_shape_func_y_electrolyte = csc_matrix((np.array(grad_shape_func_y_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
grad_shape_func_x_times_det_J_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
grad_shape_func_y_times_det_J_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))

shape_func_electrode = csc_matrix((np.array(shape_func_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
shape_func_times_det_J_time_weight_electrode = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
grad_shape_func_x_electrode = csc_matrix((np.array(grad_shape_func_x_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
grad_shape_func_y_electrode = csc_matrix((np.array(grad_shape_func_y_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
grad_shape_func_x_times_det_J_time_weight_electrode = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
grad_shape_func_y_times_det_J_time_weight_electrode = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))

shape_func_pore = csc_matrix((np.array(shape_func_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
shape_func_times_det_J_time_weight_pore = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
grad_shape_func_x_pore = csc_matrix((np.array(grad_shape_func_x_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
grad_shape_func_y_pore = csc_matrix((np.array(grad_shape_func_y_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
grad_shape_func_x_times_det_J_time_weight_pore = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
grad_shape_func_y_times_det_J_time_weight_pore = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))

shape_func_mechanical = csc_matrix((np.array(shape_func_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
shape_func_times_det_J_time_weight_mechanical = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
grad_shape_func_x_mechanical = csc_matrix((np.array(grad_shape_func_x_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
grad_shape_func_y_mechanical = csc_matrix((np.array(grad_shape_func_y_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
grad_shape_func_x_times_det_J_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
grad_shape_func_y_times_det_J_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))

if dimention == 3:
    grad_shape_func_z_electrolyte = csc_matrix((np.array(grad_shape_func_z_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
    grad_shape_func_z_times_det_J_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_z_times_det_J_time_weight_value_electrolyte), (np.array(phi_nonzero_index_row_electrolyte),np.array(phi_nonzero_index_column_electrolyte))), shape = (num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte))
    
    grad_shape_func_z_electrode = csc_matrix((np.array(grad_shape_func_z_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
    grad_shape_func_z_times_det_J_time_weight_electrode = csc_matrix((np.array(grad_shape_func_z_times_det_J_time_weight_value_electrode), (np.array(phi_nonzero_index_row_electrode),np.array(phi_nonzero_index_column_electrode))), shape = (num_gauss_points_in_domain_electrode, num_nodes_electrode))
    
    grad_shape_func_z_pore = csc_matrix((np.array(grad_shape_func_z_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
    grad_shape_func_z_times_det_J_time_weight_pore = csc_matrix((np.array(grad_shape_func_z_times_det_J_time_weight_value_pore), (np.array(phi_nonzero_index_row_pore),np.array(phi_nonzero_index_column_pore))), shape = (num_gauss_points_in_domain_pore, num_nodes_pore))
    
    grad_shape_func_z_mechanical = csc_matrix((np.array(grad_shape_func_z_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))
    grad_shape_func_z_times_det_J_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_z_times_det_J_time_weight_value_mechanical), (np.array(phi_nonzero_index_row_mechanical),np.array(phi_nonzero_index_column_mechanical))), shape = (num_gauss_points_in_domain_mechanical, num_nodes_mechanical))

comp_shape_func_grad_shape_func_in_domain = time.time()

print('time to compute the shape function and grad of shape function in domain = ' + "%s seconds" % (comp_shape_func_grad_shape_func_in_domain-def_nodes_gauss_points_time))


#######################################################
# Compute shape function and its gradient on boundaries
########################################################

print('Compute shape function and its gradient on boundaries')


if dimention == 3:
    M_b_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_x_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_y_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_z_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    phi_b_nonzero_index_row_electrolyte, phi_b_nonzero_index_column_electrolyte, phi_b_nonzerovalue_data_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte,phi_b_P_z_nonzerovalue_data_electrolyte, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, M_b_P_z_electrolyte = compute_phi_M(x_G_b_electrolyte, Gauss_b_grain_id_electrolyte, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte)
    
    M_b_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_x_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_y_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    M_b_P_z_mechanical = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],dtype=np.float64)
    phi_b_nonzero_index_row_mechanical, phi_b_nonzero_index_column_mechanical, phi_b_nonzerovalue_data_mechanical, phi_b_P_x_nonzerovalue_data_mechanical, phi_b_P_y_nonzerovalue_data_mechanical,phi_b_P_z_nonzerovalue_data_mechanical, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, M_b_P_z_mechanical = compute_phi_M(x_G_b_electrolyte, Gauss_b_grain_id_electrolyte, x_nodes_mechanical, nodes_grain_id_mechanical, a_mechanical, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_mechanical)

num_non_zero_phi_a_b_electrolyte = np.shape(np.array(phi_b_nonzero_index_row_electrolyte))[0]
num_non_zero_phi_a_b_mechanical = np.shape(np.array(phi_b_nonzero_index_row_mechanical))[0]

if dimention == 3:
    shape_func_b_value_electrolyte, shape_func_b_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_x_value_electrolyte, grad_shape_func_b_y_value_electrolyte,grad_shape_func_b_z_value_electrolyte, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte = shape_grad_shape_func(x_G_b_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_b_electrolyte,HT0, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte, phi_b_nonzero_index_row_electrolyte, phi_b_nonzero_index_column_electrolyte, det_J_b_time_weight_electrolyte, IM_RKPM, M_b_P_z_electrolyte, HT3, phi_b_P_z_nonzerovalue_data_electrolyte)
    shape_func_b_value_mechanical, shape_func_b_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_x_value_mechanical, grad_shape_func_b_y_value_mechanical,grad_shape_func_b_z_value_mechanical, grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical = shape_grad_shape_func(x_G_b_electrolyte,x_nodes_mechanical, num_non_zero_phi_a_b_mechanical,HT0, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, differential_method, HT1, HT2, phi_b_nonzerovalue_data_mechanical, phi_b_P_x_nonzerovalue_data_mechanical, phi_b_P_y_nonzerovalue_data_mechanical, phi_b_nonzero_index_row_mechanical, phi_b_nonzero_index_column_mechanical, det_J_b_time_weight_electrolyte, IM_RKPM, M_b_P_z_mechanical, HT3, phi_b_P_z_nonzerovalue_data_mechanical)

shape_func_b_electrolyte = csc_matrix((np.array(shape_func_b_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
shape_func_b_times_det_J_b_time_weight_electrolyte = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
grad_shape_func_b_x_electrolyte = csc_matrix((np.array(grad_shape_func_b_x_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
grad_shape_func_b_y_electrolyte = csc_matrix((np.array(grad_shape_func_b_y_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))

shape_func_b_mechanical = csc_matrix((np.array(shape_func_b_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
shape_func_b_times_det_J_b_time_weight_mechanical = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
grad_shape_func_b_x_mechanical = csc_matrix((np.array(grad_shape_func_b_x_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
grad_shape_func_b_y_mechanical = csc_matrix((np.array(grad_shape_func_b_y_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
grad_shape_func_b_x_times_det_J_b_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
grad_shape_func_b_y_times_det_J_b_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))

if dimention == 3:
    grad_shape_func_b_z_electrolyte = csc_matrix((np.array(grad_shape_func_b_z_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte))

    grad_shape_func_b_z_mechanical = csc_matrix((np.array(grad_shape_func_b_z_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))
    grad_shape_func_b_z_times_det_J_b_time_weight_mechanical = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical), (np.array(phi_b_nonzero_index_row_mechanical),np.array(phi_b_nonzero_index_column_mechanical))), shape = (num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical))


if dimention == 3:
    M_b_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrode)],dtype=np.float64)
    M_b_P_x_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrode)],dtype=np.float64)
    M_b_P_y_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrode)],dtype=np.float64)
    M_b_P_z_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_electrode)],dtype=np.float64)
    phi_b_nonzero_index_row_electrode, phi_b_nonzero_index_column_electrode, phi_b_nonzerovalue_data_electrode, phi_b_P_x_nonzerovalue_data_electrode, phi_b_P_y_nonzerovalue_data_electrode,phi_b_P_z_nonzerovalue_data_electrode, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, M_b_P_z_electrode = compute_phi_M(x_G_b_electrode, Gauss_b_grain_id_electrode, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode)
    
    M_b_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_pore)],dtype=np.float64)
    M_b_P_x_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_pore)],dtype=np.float64)
    M_b_P_y_pore= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_pore)],dtype=np.float64)
    M_b_P_z_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_boundary_pore)],dtype=np.float64)
    phi_b_nonzero_index_row_pore, phi_b_nonzero_index_column_pore, phi_b_nonzerovalue_data_pore, phi_b_P_x_nonzerovalue_data_pore, phi_b_P_y_nonzerovalue_data_pore,phi_b_P_z_nonzerovalue_data_pore, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, M_b_P_z_pore = compute_phi_M(x_G_b_pore, Gauss_b_grain_id_pore, x_nodes_pore, nodes_grain_id_pore, a_pore, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_pore)
    
    M_b_electrolyte_electrode_electrolyte= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_x_electrolyte_electrode_electrolyte= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_y_electrolyte_electrode_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_z_electrolyte_electrode_electrolyte = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    phi_b_nonzero_index_row_electrolyte_electrode_electrolyte, phi_b_nonzero_index_column_electrolyte_electrode_electrolyte, phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte,phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrolyte, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, M_b_P_z_electrolyte_electrode_electrolyte = compute_phi_M(x_G_b_interface_electrode_electrolyte, Gauss_b_grain_id_electrolyte_electrode_interace, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte_electrode_electrolyte)
    
    M_b_electrolyte_electrode_electrode= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_x_electrolyte_electrode_electrode= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_y_electrolyte_electrode_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    M_b_P_z_electrolyte_electrode_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrolyte_electrode_interface)],dtype=np.float64)
    phi_b_nonzero_index_row_electrolyte_electrode_electrode, phi_b_nonzero_index_column_electrolyte_electrode_electrode, phi_b_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode,phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrode, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, M_b_P_z_electrolyte_electrode_electrode = compute_phi_M(x_G_b_interface_electrode_electrolyte, Gauss_b_grain_id_electrolyte_electrode_interace, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte_electrode_electrode)
    
    M_b_electrode_pore_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_x_electrode_pore_electrode= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_y_electrode_pore_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_z_electrode_pore_electrode = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    phi_b_nonzero_index_row_electrode_pore_electrode, phi_b_nonzero_index_column_electrode_pore_electrode, phi_b_nonzerovalue_data_electrode_pore_electrode, phi_b_P_x_nonzerovalue_data_electrode_pore_electrode, phi_b_P_y_nonzerovalue_data_electrode_pore_electrode,phi_b_P_z_nonzerovalue_data_electrode_pore_electrode, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, M_b_P_z_electrode_pore_electrode = compute_phi_M(x_G_b_interface_electrode_pore, Gauss_b_grain_id_electrode_pore_interace, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode_pore_electrode)
    
    M_b_electrode_pore_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_x_electrode_pore_pore= np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_y_electrode_pore_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    M_b_P_z_electrode_pore_pore = np.array([np.zeros((4,4)) for _ in range(num_gauss_points_on_electrode_pore_interface)],dtype=np.float64)
    phi_b_nonzero_index_row_electrode_pore_pore, phi_b_nonzero_index_column_electrode_pore_pore, phi_b_nonzerovalue_data_electrode_pore_pore, phi_b_P_x_nonzerovalue_data_electrode_pore_pore, phi_b_P_y_nonzerovalue_data_electrode_pore_pore,phi_b_P_z_nonzerovalue_data_electrode_pore_pore, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, M_b_P_z_electrode_pore_pore = compute_phi_M(x_G_b_interface_electrode_pore, Gauss_b_grain_id_electrode_pore_interace, x_nodes_pore, nodes_grain_id_pore, a_pore, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode_pore_pore)
    
    if delta_point_source == 'False':
        print(np.shape(x_G_b_distributed_point_source_surface)[0])
        M_b_distributed_point_source_surface = np.array([np.zeros((4,4)) for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])],dtype=np.float64)
        M_b_P_x_distributed_point_source_surface= np.array([np.zeros((4,4)) for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])],dtype=np.float64)
        M_b_P_y_distributed_point_source_surface = np.array([np.zeros((4,4)) for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])],dtype=np.float64)
        M_b_P_z_distributed_point_source_surface = np.array([np.zeros((4,4)) for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])],dtype=np.float64)
        Gauss_b_grain_id_distributed_point_source_surface  = 1*np.ones(np.shape(x_G_b_distributed_point_source_surface)[0])
        phi_b_nonzero_index_row_distributed_point_source_surface, phi_b_nonzero_index_column_distributed_point_source_surface, phi_b_nonzerovalue_data_distributed_point_source_surface, phi_b_P_x_nonzerovalue_data_distributed_point_source_surface, phi_b_P_y_nonzerovalue_data_distributed_point_source_surface,phi_b_P_z_nonzerovalue_data_distributed_point_source_surface, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, M_b_P_z_distributed_point_source_surface = compute_phi_M(np.array(x_G_b_distributed_point_source_surface), Gauss_b_grain_id_distributed_point_source_surface, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_distributed_point_source_surface)

        num_non_zero_phi_a_b_distributed_point_source_surface = np.shape(np.array(phi_b_nonzero_index_row_distributed_point_source_surface))[0]

num_non_zero_phi_a_b_electrode = np.shape(np.array(phi_b_nonzero_index_row_electrode))[0]
num_non_zero_phi_a_b_pore = np.shape(np.array(phi_b_nonzero_index_row_pore))[0]
num_non_zero_phi_a_b_electrolyte_electrode_electrolyte = np.shape(np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte))[0]
num_non_zero_phi_a_b_electrolyte_electrode_electrode = np.shape(np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode))[0]
num_non_zero_phi_a_b_electrode_pore_electrode = np.shape(np.array(phi_b_nonzero_index_row_electrode_pore_electrode))[0]
num_non_zero_phi_a_b_electrode_pore_pore = np.shape(np.array(phi_b_nonzero_index_row_electrode_pore_pore))[0]


if dimention == 3:
    shape_func_b_value_electrode, shape_func_b_times_det_J_b_time_weight_value_electrode, grad_shape_func_b_x_value_electrode, grad_shape_func_b_y_value_electrode,grad_shape_func_b_z_value_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode = shape_grad_shape_func(x_G_b_electrode,x_nodes_electrode, num_non_zero_phi_a_b_electrode,HT0, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode, phi_b_P_x_nonzerovalue_data_electrode, phi_b_P_y_nonzerovalue_data_electrode, phi_b_nonzero_index_row_electrode, phi_b_nonzero_index_column_electrode, det_J_b_time_weight_electrode, IM_RKPM, M_b_P_z_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrode)
    shape_func_b_value_pore, shape_func_b_times_det_J_b_time_weight_value_pore, grad_shape_func_b_x_value_pore, grad_shape_func_b_y_value_pore,grad_shape_func_b_z_value_pore, grad_shape_func_b_x_times_det_J_b_time_weight_value_pore, grad_shape_func_b_y_times_det_J_b_time_weight_value_pore,grad_shape_func_b_z_times_det_J_b_time_weight_value_pore = shape_grad_shape_func(x_G_b_pore,x_nodes_pore, num_non_zero_phi_a_b_pore,HT0, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, differential_method, HT1, HT2, phi_b_nonzerovalue_data_pore, phi_b_P_x_nonzerovalue_data_pore, phi_b_P_y_nonzerovalue_data_pore, phi_b_nonzero_index_row_pore, phi_b_nonzero_index_column_pore, det_J_b_time_weight_pore, IM_RKPM, M_b_P_z_pore, HT3, phi_b_P_z_nonzerovalue_data_pore)
            
    shape_func_b_value_electrolyte_electrode_electrolyte, shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte, grad_shape_func_b_x_value_electrolyte_electrode_electrolyte, grad_shape_func_b_y_value_electrolyte_electrode_electrolyte,grad_shape_func_b_z_value_electrolyte_electrode_electrolyte, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = shape_grad_shape_func(x_G_b_interface_electrode_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_b_electrolyte_electrode_electrolyte,HT0, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_nonzero_index_row_electrolyte_electrode_electrolyte, phi_b_nonzero_index_column_electrolyte_electrode_electrolyte, det_J_b_time_weight_interface_electrode_electrolyte, IM_RKPM, M_b_P_z_electrolyte_electrode_electrolyte, HT3, phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrolyte)

    shape_func_b_value_electrolyte_electrode_electrode, shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode, grad_shape_func_b_x_value_electrolyte_electrode_electrode, grad_shape_func_b_y_value_electrolyte_electrode_electrode,grad_shape_func_b_z_value_electrolyte_electrode_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = shape_grad_shape_func(x_G_b_interface_electrode_electrolyte,x_nodes_electrode, num_non_zero_phi_a_b_electrolyte_electrode_electrode,HT0, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_nonzero_index_row_electrolyte_electrode_electrode, phi_b_nonzero_index_column_electrolyte_electrode_electrode, det_J_b_time_weight_interface_electrode_electrolyte, IM_RKPM, M_b_P_z_electrolyte_electrode_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrode)
    
    shape_func_b_value_electrode_pore_electrode, shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode, grad_shape_func_b_x_value_electrode_pore_electrode, grad_shape_func_b_y_value_electrode_pore_electrode,grad_shape_func_b_z_value_electrode_pore_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode = shape_grad_shape_func(x_G_b_interface_electrode_pore,x_nodes_electrode, num_non_zero_phi_a_b_electrode_pore_electrode,HT0, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode_pore_electrode, phi_b_P_x_nonzerovalue_data_electrode_pore_electrode, phi_b_P_y_nonzerovalue_data_electrode_pore_electrode, phi_b_nonzero_index_row_electrode_pore_electrode, phi_b_nonzero_index_column_electrode_pore_electrode, det_J_b_time_weight_interface_electrode_pore, IM_RKPM, M_b_P_z_electrode_pore_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrode_pore_electrode)

    shape_func_b_value_electrode_pore_pore, shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore, grad_shape_func_b_x_value_electrode_pore_pore, grad_shape_func_b_y_value_electrode_pore_pore,grad_shape_func_b_z_value_electrode_pore_pore, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore = shape_grad_shape_func(x_G_b_interface_electrode_pore,x_nodes_pore, num_non_zero_phi_a_b_electrode_pore_pore,HT0, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode_pore_pore, phi_b_P_x_nonzerovalue_data_electrode_pore_pore, phi_b_P_y_nonzerovalue_data_electrode_pore_pore, phi_b_nonzero_index_row_electrode_pore_pore, phi_b_nonzero_index_column_electrode_pore_pore, det_J_b_time_weight_interface_electrode_pore, IM_RKPM, M_b_P_z_electrode_pore_pore, HT3, phi_b_P_z_nonzerovalue_data_electrode_pore_pore)
    
    if delta_point_source == 'False':
        num_gauss_points_distributed_point_source_surface = np.shape(x_G_b_distributed_point_source_surface)[0]

        shape_func_b_value_distributed_point_source_surface, shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface, grad_shape_func_b_x_value_distributed_point_source_surface, grad_shape_func_b_y_value_distributed_point_source_surface,grad_shape_func_b_z_value_distributed_point_source_surface, grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface, grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface,grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface = shape_grad_shape_func(x_G_b_distributed_point_source_surface,x_nodes_electrolyte, num_non_zero_phi_a_b_distributed_point_source_surface,HT0, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, differential_method, HT1, HT2, phi_b_nonzerovalue_data_distributed_point_source_surface, phi_b_P_x_nonzerovalue_data_distributed_point_source_surface, phi_b_P_y_nonzerovalue_data_distributed_point_source_surface, phi_b_nonzero_index_row_distributed_point_source_surface, phi_b_nonzero_index_column_distributed_point_source_surface, det_J_b_time_weight_distributed_point_source_surface, IM_RKPM, M_b_P_z_distributed_point_source_surface, HT3, phi_b_P_z_nonzerovalue_data_distributed_point_source_surface)
        
        shape_func_b_distributed_point_source_surface = csc_matrix((np.array(shape_func_b_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        shape_func_b_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_x_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_x_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_y_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_y_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_z_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_z_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_x_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_y_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))
        grad_shape_func_b_z_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface), (np.array(phi_b_nonzero_index_row_distributed_point_source_surface),np.array(phi_b_nonzero_index_column_distributed_point_source_surface))), shape = (num_gauss_points_distributed_point_source_surface, num_nodes_electrolyte))

shape_func_b_electrolyte_electrode_electrolyte = csc_matrix((np.array(shape_func_b_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
grad_shape_func_b_x_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_x_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
grad_shape_func_b_y_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_y_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))

shape_func_b_electrolyte_electrode_electrode = csc_matrix((np.array(shape_func_b_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
grad_shape_func_b_x_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_x_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
grad_shape_func_b_y_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_y_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))

shape_func_b_electrode_pore_electrode = csc_matrix((np.array(shape_func_b_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
grad_shape_func_b_x_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_x_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
grad_shape_func_b_y_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_y_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
grad_shape_func_b_x_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
grad_shape_func_b_y_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))

shape_func_b_electrode_pore_pore = csc_matrix((np.array(shape_func_b_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
shape_func_b_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
grad_shape_func_b_x_electrode_pore_pore = csc_matrix((np.array(grad_shape_func_b_x_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
grad_shape_func_b_y_electrode_pore_pore = csc_matrix((np.array(grad_shape_func_b_y_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
grad_shape_func_b_x_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
grad_shape_func_b_y_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))


shape_func_b_electrode = csc_matrix((np.array(shape_func_b_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
shape_func_b_times_det_J_b_time_weight_electrode = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
grad_shape_func_b_x_electrode = csc_matrix((np.array(grad_shape_func_b_x_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
grad_shape_func_b_y_electrode = csc_matrix((np.array(grad_shape_func_b_y_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
grad_shape_func_b_x_times_det_J_b_time_weight_electrode = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
grad_shape_func_b_y_times_det_J_b_time_weight_electrode = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))

shape_func_b_pore = csc_matrix((np.array(shape_func_b_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
shape_func_b_times_det_J_b_time_weight_pore = csc_matrix((np.array(shape_func_b_times_det_J_b_time_weight_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
grad_shape_func_b_x_pore = csc_matrix((np.array(grad_shape_func_b_x_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
grad_shape_func_b_y_pore = csc_matrix((np.array(grad_shape_func_b_y_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
grad_shape_func_b_x_times_det_J_b_time_weight_pore = csc_matrix((np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
grad_shape_func_b_y_times_det_J_b_time_weight_pore = csc_matrix((np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))

if dimention == 3:
    grad_shape_func_b_z_electrode = csc_matrix((np.array(grad_shape_func_b_z_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrode = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode), (np.array(phi_b_nonzero_index_row_electrode),np.array(phi_b_nonzero_index_column_electrode))), shape = (num_gauss_points_on_boundary_electrode, num_nodes_electrode))
    
    grad_shape_func_b_z_pore = csc_matrix((np.array(grad_shape_func_b_z_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
    grad_shape_func_b_z_times_det_J_b_time_weight_pore = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_pore), (np.array(phi_b_nonzero_index_row_pore),np.array(phi_b_nonzero_index_column_pore))), shape = (num_gauss_points_on_boundary_pore, num_nodes_pore))
    
    grad_shape_func_b_z_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_z_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
    grad_shape_func_b_z_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_z_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
    grad_shape_func_b_z_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_z_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
    grad_shape_func_b_z_electrode_pore_pore = csc_matrix((np.array(grad_shape_func_b_z_value_electrode_pore_pore), (np.array(phi_b_nonzero_index_row_electrode_pore_pore),np.array(phi_b_nonzero_index_column_electrode_pore_pore))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_pore))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrolyte))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode), (np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode))), shape = (num_gauss_points_on_electrolyte_electrode_interface, num_nodes_electrode))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))
    grad_shape_func_b_z_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix((np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode), (np.array(phi_b_nonzero_index_row_electrode_pore_electrode),np.array(phi_b_nonzero_index_column_electrode_pore_electrode))), shape = (num_gauss_points_on_electrode_pore_interface, num_nodes_electrode))


"""shape function with size n_nodes times n_nodes, this is used to predict the potential on all nodes"""


if dimention == 3:
    M_electrolyte_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrolyte)])
    M_P_x_electrolyte_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrolyte)]) # partial M partial x
    M_P_y_electrolyte_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrolyte)]) # partial M partial y
    M_P_z_electrolyte_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrolyte)]) # partial M partial y
    phi_nonzero_index_row_electrolyte_nn, phi_nonzero_index_column_electrolyte_nn, phi_nonzerovalue_data_electrolyte_nn, phi_P_x_nonzerovalue_data_electrolyte_nn, phi_P_y_nonzerovalue_data_electrolyte_nn, phi_P_z_nonzerovalue_data_electrolyte_nn, M_electrolyte_nn, M_P_x_electrolyte_nn, M_P_y_electrolyte_nn, M_P_z_electrolyte_nn = compute_phi_M(x_nodes_electrolyte, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte_nn, M_P_x_electrolyte_nn, M_P_y_electrolyte_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte_nn)
    
    M_electrode_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrode)])
    M_P_x_electrode_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrode)]) # partial M partial x
    M_P_y_electrode_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrode)]) # partial M partial y
    M_P_z_electrode_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_electrode)]) # partial M partial y
    phi_nonzero_index_row_electrode_nn, phi_nonzero_index_column_electrode_nn, phi_nonzerovalue_data_electrode_nn, phi_P_x_nonzerovalue_data_electrode_nn, phi_P_y_nonzerovalue_data_electrode_nn, phi_P_z_nonzerovalue_data_electrode_nn, M_electrode_nn, M_P_x_electrode_nn, M_P_y_electrode_nn, M_P_z_electrode_nn = compute_phi_M(x_nodes_electrode, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode_nn, M_P_x_electrode_nn, M_P_y_electrode_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode_nn)

    M_pore_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_pore)])
    M_P_x_pore_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_pore)]) # partial M partial x
    M_P_y_pore_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_pore)]) # partial M partial y
    M_P_z_pore_nn = np.array([np.zeros((4,4)) for _ in range(num_nodes_pore)]) # partial M partial y
    phi_nonzero_index_row_pore_nn, phi_nonzero_index_column_pore_nn, phi_nonzerovalue_data_pore_nn, phi_P_x_nonzerovalue_data_pore_nn, phi_P_y_nonzerovalue_data_pore_nn, phi_P_z_nonzerovalue_data_pore_nn, M_pore_nn, M_P_x_pore_nn, M_P_y_pore_nn, M_P_z_pore_nn = compute_phi_M(x_nodes_pore, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore_nn, M_P_x_pore_nn, M_P_y_pore_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore_nn)

    num_non_zero_phi_a_electrolyte_nn = np.shape(np.array(phi_nonzero_index_row_electrolyte_nn))[0]
    shape_func_value_electrolyte_nn = shape_func_n_nodes_by_n_nodes(x_nodes_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte_nn,HT0, M_electrolyte_nn, phi_nonzerovalue_data_electrolyte_nn,phi_nonzero_index_row_electrolyte_nn, phi_nonzero_index_column_electrolyte_nn)
    shape_func_n_nodes_n_nodes_electrolyte = csc_matrix((np.array(shape_func_value_electrolyte_nn), (np.array(phi_nonzero_index_row_electrolyte_nn),np.array(phi_nonzero_index_column_electrolyte_nn))), shape = (num_nodes_electrolyte, num_nodes_electrolyte))

    num_non_zero_phi_a_electrode_nn = np.shape(np.array(phi_nonzero_index_row_electrode_nn))[0]
    shape_func_value_electrode_nn = shape_func_n_nodes_by_n_nodes(x_nodes_electrode,x_nodes_electrode, num_non_zero_phi_a_electrode_nn,HT0, M_electrode_nn, phi_nonzerovalue_data_electrode_nn,phi_nonzero_index_row_electrode_nn, phi_nonzero_index_column_electrode_nn)
    shape_func_n_nodes_n_nodes_electrode = csc_matrix((np.array(shape_func_value_electrode_nn), (np.array(phi_nonzero_index_row_electrode_nn),np.array(phi_nonzero_index_column_electrode_nn))), shape = (num_nodes_electrode, num_nodes_electrode))
    
    num_non_zero_phi_a_pore_nn = np.shape(np.array(phi_nonzero_index_row_pore_nn))[0]
    shape_func_value_pore_nn = shape_func_n_nodes_by_n_nodes(x_nodes_pore,x_nodes_pore, num_non_zero_phi_a_pore_nn,HT0, M_pore_nn, phi_nonzerovalue_data_pore_nn,phi_nonzero_index_row_pore_nn, phi_nonzero_index_column_pore_nn)
    shape_func_n_nodes_n_nodes_pore = csc_matrix((np.array(shape_func_value_pore_nn), (np.array(phi_nonzero_index_row_pore_nn),np.array(phi_nonzero_index_column_pore_nn))), shape = (num_nodes_pore, num_nodes_pore))


"""
1. shape function used to interpolate the phi and phie at the interface line (3d), shape: number of gauss points on source line times number of nodes
2. shape function used to interpolate the displacement at the fixed line (3d), shape: number of gauss points on fixed line times number of nodes
"""
if dimention == 3:
    M_electrolyte_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)])
    M_P_x_electrolyte_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial x
    M_P_y_electrolyte_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y
    M_P_z_electrolyte_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y

    phi_nonzero_index_row_electrolyte_line_nodes, phi_nonzero_index_column_electrolyte_line_nodes, phi_nonzerovalue_data_electrolyte_line_nodes, phi_P_x_nonzerovalue_data_electrolyte_line_nodes, phi_P_y_nonzerovalue_data_electrolyte_line_nodes, phi_P_z_nonzerovalue_data_electrolyte_line_nodes, M_electrolyte_line_nodes, M_P_x_electrolyte_line_nodes, M_P_y_electrolyte_line_nodes, M_P_z_electrolyte_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte_line_nodes, M_P_x_electrolyte_line_nodes, M_P_y_electrolyte_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte_line_nodes)
    
    num_non_zero_phi_a_electrolyte_line_nodes = np.shape(np.array(phi_nonzero_index_row_electrolyte_line_nodes))[0]
    
    shape_func_value_electrolyte_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte_line_nodes,HT0, M_electrolyte_line_nodes, phi_nonzerovalue_data_electrolyte_line_nodes,phi_nonzero_index_row_electrolyte_line_nodes, phi_nonzero_index_column_electrolyte_line_nodes)

    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func_line_n_nodes_electrolyte = csc_matrix((np.array(shape_func_value_electrolyte_line_nodes), (np.array(phi_nonzero_index_row_electrolyte_line_nodes),np.array(phi_nonzero_index_column_electrolyte_line_nodes))), shape = (num_source_line_gauss_points, num_nodes_electrolyte))
    
    shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight = shape_func_line_n_nodes_electrolyte.copy()
    shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight.data *= det_J_b_time_weight_line[shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight.indices]
    
    M_electrode_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)])
    M_P_x_electrode_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial x
    M_P_y_electrode_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y
    M_P_z_electrode_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y
    
    phi_nonzero_index_row_electrode_line_nodes, phi_nonzero_index_column_electrode_line_nodes, phi_nonzerovalue_data_electrode_line_nodes, phi_P_x_nonzerovalue_data_electrode_line_nodes, phi_P_y_nonzerovalue_data_electrode_line_nodes, phi_P_z_nonzerovalue_data_electrode_line_nodes, M_electrode_line_nodes, M_P_x_electrode_line_nodes, M_P_y_electrode_line_nodes, M_P_z_electrode_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode_line_nodes, M_P_x_electrode_line_nodes, M_P_y_electrode_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode_line_nodes)
    
    num_non_zero_phi_a_electrode_line_nodes = np.shape(np.array(phi_nonzero_index_row_electrode_line_nodes))[0]
    
    shape_func_value_electrode_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_electrode, num_non_zero_phi_a_electrode_line_nodes,HT0, M_electrode_line_nodes, phi_nonzerovalue_data_electrode_line_nodes,phi_nonzero_index_row_electrode_line_nodes, phi_nonzero_index_column_electrode_line_nodes)
    
    
    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func_line_n_nodes_electrode = csc_matrix((np.array(shape_func_value_electrode_line_nodes), (np.array(phi_nonzero_index_row_electrode_line_nodes),np.array(phi_nonzero_index_column_electrode_line_nodes))), shape = (num_source_line_gauss_points, num_nodes_electrode))
    shape_func_line_n_nodes_electrode_times_det_J_b_time_weight = shape_func_line_n_nodes_electrode.copy()
    shape_func_line_n_nodes_electrode_times_det_J_b_time_weight.data *= det_J_b_time_weight_line[shape_func_line_n_nodes_electrode_times_det_J_b_time_weight.indices]
    
    M_pore_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)])
    M_P_x_pore_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial x
    M_P_y_pore_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y
    M_P_z_pore_line_nodes = np.array([np.zeros((4,4)) for _ in range(num_source_line_gauss_points)]) # partial M partial y
    
    phi_nonzero_index_row_pore_line_nodes, phi_nonzero_index_column_pore_line_nodes, phi_nonzerovalue_data_pore_line_nodes, phi_P_x_nonzerovalue_data_pore_line_nodes, phi_P_y_nonzerovalue_data_pore_line_nodes, phi_P_z_nonzerovalue_data_pore_line_nodes, M_pore_line_nodes, M_P_x_pore_line_nodes, M_P_y_pore_line_nodes, M_P_z_pore_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore_line_nodes, M_P_x_pore_line_nodes, M_P_y_pore_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore_line_nodes)
    
    num_non_zero_phi_a_pore_line_nodes = np.shape(np.array(phi_nonzero_index_row_pore_line_nodes))[0]
    
    shape_func_value_pore_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_pore, num_non_zero_phi_a_pore_line_nodes,HT0, M_pore_line_nodes, phi_nonzerovalue_data_pore_line_nodes,phi_nonzero_index_row_pore_line_nodes, phi_nonzero_index_column_pore_line_nodes)
    
    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func_line_n_nodes_pore = csc_matrix((np.array(shape_func_value_pore_line_nodes), (np.array(phi_nonzero_index_row_pore_line_nodes),np.array(phi_nonzero_index_column_pore_line_nodes))), shape = (num_source_line_gauss_points, num_nodes_pore))
    shape_func_line_n_nodes_pore_times_det_J_b_time_weight = shape_func_line_n_nodes_pore.copy()
    shape_func_line_n_nodes_pore_times_det_J_b_time_weight.data *= det_J_b_time_weight_line[shape_func_line_n_nodes_pore_times_det_J_b_time_weight.indices]
    

    M_fixed_nodes = np.array([np.zeros((4,4)) for _ in range(num_fixed_gauss_points)])
    M_P_x_fixed_nodes = np.array([np.zeros((4,4)) for _ in range(num_fixed_gauss_points)]) # partial M partial x
    M_P_y_fixed_nodes = np.array([np.zeros((4,4)) for _ in range(num_fixed_gauss_points)]) # partial M partial y
    M_P_z_fixed_nodes = np.array([np.zeros((4,4)) for _ in range(num_fixed_gauss_points)]) # partial M partial y

    phi_nonzero_index_row_fixed_nodes, phi_nonzero_index_column_fixed_nodes, phi_nonzerovalue_data_fixed_nodes, phi_P_x_nonzerovalue_data_fixed_nodes, phi_P_y_nonzerovalue_data_fixed_nodes, phi_P_z_nonzerovalue_data_fixed_nodes, M_fixed_nodes, M_P_x_fixed_nodes, M_P_y_fixed_nodes, M_P_z_fixed_nodes = compute_phi_M(x_G_b_fixed, Gauss_grain_id_mechanical, x_nodes_mechanical,nodes_grain_id_mechanical, a_mechanical, M_fixed_nodes, M_P_x_fixed_nodes, M_P_y_fixed_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_fixed_nodes)

    num_non_zero_phi_a_fixed_nodes = np.shape(np.array(phi_nonzero_index_row_fixed_nodes))[0]

    shape_func_value_fixed_point, shape_func_times_det_J_time_weight_value_fixed_point, grad_shape_func_x_value_fixed_point,grad_shape_func_y_value_fixed_point, grad_shape_func_z_value_fixed_point, grad_shape_func_x_times_det_J_time_weight_value_fixed_point, grad_shape_func_y_times_det_J_time_weight_value_fixed_point, grad_shape_func_z_times_det_J_time_weight_value_fixed_point = shape_grad_shape_func(x_G_b_fixed,x_nodes_mechanical, num_non_zero_phi_a_fixed_nodes,HT0, M_fixed_nodes, M_P_x_fixed_nodes, M_P_y_fixed_nodes, differential_method, HT1, HT2, phi_nonzerovalue_data_fixed_nodes,phi_P_x_nonzerovalue_data_fixed_nodes,phi_P_y_nonzerovalue_data_fixed_nodes, phi_nonzero_index_row_fixed_nodes, phi_nonzero_index_column_fixed_nodes, det_J_b_time_weight_fixed, IM_RKPM, M_P_z_fixed_nodes, HT3, phi_P_z_nonzerovalue_data_fixed_nodes)

    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func_fixed_point = csc_matrix((np.array(shape_func_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    shape_func_times_det_J_time_weight_fixed_point = csc_matrix((np.array(shape_func_times_det_J_time_weight_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_x_fixed_point = csc_matrix((np.array(grad_shape_func_x_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_y_fixed_point = csc_matrix((np.array(grad_shape_func_y_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_x_times_det_J_time_weight_fixed_point = csc_matrix((np.array(grad_shape_func_x_times_det_J_time_weight_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_y_times_det_J_time_weight_fixed_point = csc_matrix((np.array(grad_shape_func_y_times_det_J_time_weight_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_z_fixed_point = csc_matrix((np.array(grad_shape_func_z_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))
    grad_shape_func_z_times_det_J_time_weight_fixed_point = csc_matrix((np.array(grad_shape_func_z_times_det_J_time_weight_value_fixed_point), (np.array(phi_nonzero_index_row_fixed_nodes),np.array(phi_nonzero_index_column_fixed_nodes))), shape = (num_fixed_gauss_points, num_nodes_mechanical))


comp_shape_func_grad_shape_func_on_boundaries = time.time()



print('time to compute the shape function and grad of shape function on baoundaries = ' + "%s seconds" % (comp_shape_func_grad_shape_func_on_boundaries-comp_shape_func_grad_shape_func_in_domain))


######################
# fuel cell solver
######################
    
if studied_physics == "fuel cell":
    
    h_E = (y_max-y_min)/num_pixels_xyz[1] # maximum element size

    C_old_electrode= np.array(np.ones((num_nodes_electrode)))*960 # initial phi is 0.03
    phi_old_electrolyte = 0.8*np.array(np.ones((num_nodes_electrolyte)))
    C_old_pore = 9.60*np.array(np.ones((num_nodes_pore)))

    results_old = np.concatenate((phi_old_electrolyte, C_old_electrode, C_old_pore))

    beta_Nitsche_electrode = np.ones((num_nodes_electrode))*100/h_E
    beta_Nitsche_electrolyte = np.ones((num_nodes_electrolyte))*100/h_E
    beta_Nitsche_pore = np.ones((num_nodes_pore))*100/h_E
    beta_Nitsche_mechanical = np.ones((num_nodes_mechanical))*1e6/h_E

    # when interpolate function g using shape function, g(x)_interpolated = sum over nodes for shape function at x at i_th node times g_i,
    # if g is constant, gi = g = constant, if g is not constant, gi is not g value at the i_th nodes,
    # instead, we need use g(gauss point) = shape_func*gi, gi = shape_func_inverse dot g(gauss point)

    # as we only integral g on left or right boundary where the dirichlet bc are defined, and the value of g on these 
    # boundaries are constant, so gi on this boundary nodes can be the correcponding constant, g = shapefunc_b dot gi,
    # g=A is constant on this bondary, so gi=A can satisfy A=shape_func_b dot A.

    # g is zero in electrolyte domain, but not constant in electrode domain
    g_diretchlet_electrolyte = np.zeros((num_gauss_points_on_boundary_electrolyte))  # phi_electolyte(x=0)=g, g=0, at x=0, 
    g_diretchlet_electrode = np.ones((num_gauss_points_on_boundary_electrode))*c_boundary
    g_diretchlet_pore = np.ones((num_gauss_points_on_boundary_pore))*c_boundary_pore

    # the normal vector (x component) is not constant
    if dimention == 3:
        normal_vector_x_electrolyte = 0
        normal_vector_x_electrode = 0
        normal_vector_x_pore = 0

        normal_vector_y_electrode = 1
        normal_vector_y_electrolyte = -1
        normal_vector_y_pore = 1

        normal_vector_z_electrode = 0
        normal_vector_z_electrolyte = 0
        normal_vector_z_pore = 0

        
    if dimention == 3:

        diff = 3000.0   # initial difference: 10, if initial_diff<threshold, stop newton interation

        iteration_num = 0

        """
        if the point source is treated as delta function
        
        """
        if delta_point_source == 'False':

            phi_old_line_gauss_electrolyte = shape_func_line_n_nodes_electrolyte*phi_old_electrolyte

            i_HOR = i_0*np.exp(0.5*Fday/R/T*(-phi_old_line_gauss_electrolyte+V_app-E_0)) 

            distributed_line_source_electrolyte = -0.0125*np.ones(np.shape(x_G_b_distributed_point_source_surface)[0])/(z_max-z_min)*2*5#i_HOR/2###
            line_source_electrode = np.zeros(np.shape(i_HOR)[0])
            line_source_pore = np.zeros(np.shape(i_HOR)[0])
        else:
            phi_old_line_gauss_electrolyte = shape_func_line_n_nodes_electrolyte*phi_old_electrolyte
            
            i_HOR = i_0*np.exp(0.5*Fday/R/T*(-phi_old_line_gauss_electrolyte+V_app-E_0))        
            line_source_electrolyte_old = i_HOR/2#-0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
            line_source_pore_old = np.zeros(np.shape(i_HOR)[0])#-i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
            line_source_electrode_old = np.zeros(np.shape(i_HOR)[0])
            line_source_electrolyte = i_HOR/2#-0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
            line_source_pore = np.zeros(np.shape(i_HOR)[0])#-i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
            line_source_electrode = np.zeros(np.shape(i_HOR)[0])
            
        C_old_electrolyte_electrode = shape_func_b_electrolyte_electrode_electrode*C_old_electrode
        phi_old_electrolyte_electrode = shape_func_b_electrolyte_electrode_electrolyte*phi_old_electrolyte
        i_solid = i_0_solid*np.exp(0.5*Fday/R/T*(-phi_old_electrolyte_electrode+V_app))* C_old_electrolyte_electrode/c_boundary
        interface_source_electrolyte_electrode_electrolyte_old = -i_solid/2
        interface_source_electrolyte_electrode_electrode_old = i_solid/2/96485
        interface_source_electrolyte_electrode_electrolyte = -i_solid/2
        interface_source_electrolyte_electrode_electrode = i_solid/2/96485
        
        C_old_electrode_pore = shape_func_b_electrode_pore_electrode*C_old_electrode
        J_interface = k_gas*(c_boundary-C_old_electrode_pore)
        interface_source_electrode_pore_electrode_old = -J_interface
        interface_source_electrode_pore_pore_old = J_interface
        interface_source_electrode_pore_electrode = -J_interface
        interface_source_electrode_pore_pore = J_interface


        while diff >6.0e-2:
            print('iteration number:', iteration_num)
            
            
            if delta_point_source == 'True':
               
                K_electrolyte, f_electrolyte = diffusion_matrix_fuel_cell(dimention, line_source_electrolyte, shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight, g_diretchlet_electrolyte, beta_Nitsche_electrolyte, normal_vector_x_electrolyte, normal_vector_y_electrolyte, diffusion_electrolyte,grad_shape_func_x_electrolyte,grad_shape_func_y_electrolyte,grad_shape_func_x_times_det_J_time_weight_electrolyte,grad_shape_func_y_times_det_J_time_weight_electrolyte,\
                        shape_func_b_electrolyte,shape_func_b_times_det_J_b_time_weight_electrolyte,grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte, shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte, interface_source_electrolyte_electrode_electrolyte, grad_shape_func_z_electrolyte, grad_shape_func_z_times_det_J_time_weight_electrolyte, grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte, normal_vector_z_electrolyte)
            else:
                K_electrolyte, f_electrolyte = diffusion_matrix_fuel_cell_distributed_point_source(dimention, distributed_line_source_electrolyte, shape_func_b_times_det_J_b_time_weight_distributed_point_source_surface, g_diretchlet_electrolyte, beta_Nitsche_electrolyte, normal_vector_x_electrolyte, normal_vector_y_electrolyte, diffusion_electrolyte,grad_shape_func_x_electrolyte,grad_shape_func_y_electrolyte,grad_shape_func_x_times_det_J_time_weight_electrolyte,grad_shape_func_y_times_det_J_time_weight_electrolyte,\
                        shape_func_b_electrolyte,shape_func_b_times_det_J_b_time_weight_electrolyte,grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte, shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte, interface_source_electrolyte_electrode_electrolyte, grad_shape_func_z_electrolyte, grad_shape_func_z_times_det_J_time_weight_electrolyte, grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte, normal_vector_z_electrolyte)
            
            interface_source_electrode = np.concatenate((interface_source_electrolyte_electrode_electrode, interface_source_electrode_pore_electrode))
            # shape_func_interface_electrode = vstack((shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode, shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode), format='csc')
            shape_func_interface_electrode = csc_matrix(np.vstack([ shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode.toarray(), shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode.toarray() ]))
            K_electrode, f_electrode = diffusion_matrix_fuel_cell(dimention, line_source_electrode, shape_func_line_n_nodes_electrode_times_det_J_b_time_weight, g_diretchlet_electrode, beta_Nitsche_electrode, normal_vector_x_electrode, normal_vector_y_electrode, diffusion_electrode,grad_shape_func_x_electrode,grad_shape_func_y_electrode,grad_shape_func_x_times_det_J_time_weight_electrode,grad_shape_func_y_times_det_J_time_weight_electrode,\
                     shape_func_b_electrode,shape_func_b_times_det_J_b_time_weight_electrode,grad_shape_func_b_x_times_det_J_b_time_weight_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_electrode, shape_func_interface_electrode, interface_source_electrode, grad_shape_func_z_electrode, grad_shape_func_z_times_det_J_time_weight_electrode, grad_shape_func_b_z_times_det_J_b_time_weight_electrode, normal_vector_z_electrode)
            
            
            K_pore, f_pore = diffusion_matrix_fuel_cell(dimention, line_source_pore, shape_func_line_n_nodes_pore_times_det_J_b_time_weight, g_diretchlet_pore, beta_Nitsche_pore, normal_vector_x_pore, normal_vector_y_pore, diffusion_pore,grad_shape_func_x_pore,grad_shape_func_y_pore,grad_shape_func_x_times_det_J_time_weight_pore,grad_shape_func_y_times_det_J_time_weight_pore,\
                     shape_func_b_pore,shape_func_b_times_det_J_b_time_weight_pore,grad_shape_func_b_x_times_det_J_b_time_weight_pore, grad_shape_func_b_y_times_det_J_b_time_weight_pore, shape_func_b_times_det_J_b_time_weight_electrode_pore_pore, interface_source_electrode_pore_pore, grad_shape_func_z_pore, grad_shape_func_z_times_det_J_time_weight_pore, grad_shape_func_b_z_times_det_J_b_time_weight_pore, normal_vector_z_pore)


            # K = block_diag((K_electrolyte, K_electrode, K_pore), format='csc')
            # K = np.block([
            #             [K_electrolyte.toarray(),        np.zeros_like(K_electrode.toarray()), np.zeros_like(K_pore.toarray())],
            #             [np.zeros_like(K_electrolyte.toarray()), K_electrode.toarray(),        np.zeros_like(K_pore.toarray())],
            #             [np.zeros_like(K_electrolyte.toarray()), np.zeros_like(K_electrode.toarray()), K_pore.toarray()]
            #             ])
            Ke = K_electrolyte.toarray()
            Kc = K_electrode.toarray()
            Kp = K_pore.toarray()

            # build zero blocks with *correct shapes*
            Z_ec = np.zeros((Ke.shape[0], Kc.shape[1]))
            Z_ep = np.zeros((Ke.shape[0], Kp.shape[1]))
            Z_ce = np.zeros((Kc.shape[0], Ke.shape[1]))
            Z_cp = np.zeros((Kc.shape[0], Kp.shape[1]))
            Z_pe = np.zeros((Kp.shape[0], Ke.shape[1]))
            Z_pc = np.zeros((Kp.shape[0], Kc.shape[1]))

            K = csc_matrix(np.block([
                [Ke,   Z_ec, Z_ep],
                [Z_ce, Kc,   Z_cp],
                [Z_pe, Z_pc, Kp  ]
            ]))
            f = np.concatenate((f_electrolyte, f_electrode, f_pore))

            results_new = spsolve(K, f)

            diff = np.linalg.norm(results_new-results_old, 2)
            results_old[:] = results_new.copy()
        
            iteration_num +=1

            phi_new_electrolyte = results_new[:num_nodes_electrolyte]
            C_new_electrode = results_new[num_nodes_electrolyte:num_nodes_electrode+num_nodes_electrolyte]
            C_new_pore = results_new[num_nodes_electrode+num_nodes_electrolyte:num_nodes_electrode+num_nodes_electrolyte+num_nodes_pore]

            phi_old_electrolyte = results_new[:num_nodes_electrolyte]
            C_old_electrode = results_new[num_nodes_electrolyte:num_nodes_electrolyte+num_nodes_electrode]
            C_old_pore = results_new[num_nodes_electrolyte+num_nodes_electrode:num_nodes_electrolyte+num_nodes_electrode+num_nodes_pore]
            print('change of solution',diff)

            # update the source term
            if delta_point_source == 'True':
                phi_new_line_gauss_electrolyte = shape_func_line_n_nodes_electrolyte*phi_old_electrolyte
                i_HOR = i_0*np.exp(0.5*Fday/R/T*(-phi_new_line_gauss_electrolyte+V_app-E_0))        
                line_source_electrolyte_new = i_HOR/2#-0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
                line_source_pore_new = np.zeros(np.shape(i_HOR)[0])#-i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
                line_source_electrode_new = np.zeros(np.shape(i_HOR)[0])
                   
                # flux will be applied
                line_source_electrolyte = line_source_electrolyte_new *0.2 + line_source_electrolyte_old*0.8
                line_source_pore = line_source_pore_new *0.2 + line_source_pore_old*0.8
                line_source_electrode = line_source_electrode_new *0.2 + line_source_electrode_old*0.8
                line_source_electrolyte_old = line_source_electrolyte.copy()
                line_source_pore_old = line_source_pore.copy()
                line_source_electrode_old = line_source_electrode.copy()
            else:
                # distributed point source do not need update the source term
                pass

            # flux across electrolyte/electrode interface
            C_new_electrolyte_electrode = shape_func_b_electrolyte_electrode_electrode*C_old_electrode
            phi_new_electrolyte_electrode = shape_func_b_electrolyte_electrode_electrolyte*phi_old_electrolyte
            i_solid = i_0_solid*np.exp(0.5*Fday/R/T*(-phi_new_electrolyte_electrode+V_app))*C_new_electrolyte_electrode/c_boundary
            interface_source_electrolyte_electrode_electrolyte_new = -i_solid/2
            interface_source_electrolyte_electrode_electrode_new = i_solid/2/96485
            interface_source_electrolyte_electrode_electrolyte = interface_source_electrolyte_electrode_electrolyte_new*0.2 + interface_source_electrolyte_electrode_electrolyte_old*0.8
            interface_source_electrolyte_electrode_electrode = interface_source_electrolyte_electrode_electrode_new*0.2 + interface_source_electrolyte_electrode_electrode_old*0.8
            interface_source_electrolyte_electrode_electrolyte_old = interface_source_electrolyte_electrode_electrolyte.copy()
            interface_source_electrolyte_electrode_electrode_old = interface_source_electrolyte_electrode_electrode.copy()

            # flux across the electrode/pore interface
            C_new_electrode_pore = shape_func_b_electrode_pore_electrode*C_old_electrode
            J_interface = k_gas*(c_boundary-C_new_electrode_pore)
            interface_source_electrode_pore_electrode_new = -J_interface
            interface_source_electrode_pore_pore_new = J_interface
            interface_source_electrode_pore_electrode = interface_source_electrode_pore_electrode_new*0.2 + interface_source_electrode_pore_electrode_old*0.8
            interface_source_electrode_pore_pore = interface_source_electrode_pore_pore_new*0.2 + interface_source_electrode_pore_pore_old*0.8
            interface_source_electrode_pore_electrode_old = interface_source_electrode_pore_electrode.copy()
            interface_source_electrode_pore_pore_old = interface_source_electrode_pore_pore.copy()

        
        # mechanical solver:
        ####################################################################
        # assemble matrix for mechanical simulation and solve
        ####################################################################
        # if ii==0:
        start_mechanical_time = time.time()

        D_damage = np.zeros((num_gauss_points_in_domain_mechanical, 1))
        lambda_mechanical_electrode_array = lambda_mechanical_electrode*np.ones(num_gauss_points_in_domain_electrode)
        lambda_mechanical_electrolyte_array = lambda_mechanical_electrolyte*np.ones(num_gauss_points_in_domain_electrolyte)
        lambda_mechanical = np.concatenate((lambda_mechanical_electrolyte_array, lambda_mechanical_electrode_array))
        
        mu_electrode_array = mu_electrode*np.ones(num_gauss_points_in_domain_electrode)
        mu_electrolyte_array = mu_electrolyte*np.ones(num_gauss_points_in_domain_electrolyte)
        mu_mechanical = np.concatenate((mu_electrolyte_array, mu_electrode_array))

        print('define mechanical stiffness matrix')

        C, T_c = mechanical_C_tensor_3d(num_gauss_points_in_domain_mechanical, D_damage, lambda_mechanical, mu_mechanical, Gauss_angle_mechanical, gauss_rotation_axis)
        
        

        K_mechanical = mechanical_stiffness_matrix_3d_fuel_cell(C,num_gauss_points_in_domain_mechanical, grad_shape_func_x_times_det_J_time_weight_mechanical, grad_shape_func_x_mechanical, grad_shape_func_y_times_det_J_time_weight_mechanical, grad_shape_func_y_mechanical, grad_shape_func_z_times_det_J_time_weight_mechanical, grad_shape_func_z_mechanical,\
                                    beta_Nitsche_mechanical, \
                                    shape_func_fixed_point, shape_func_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_x_fixed_point, grad_shape_func_x_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_y_fixed_point, grad_shape_func_y_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_z_fixed_point, grad_shape_func_z_times_det_J_time_weight_fixed_point,\
                                    normal_vector_x_electrolyte, normal_vector_y_electrolyte,normal_vector_z_electrolyte,\
                                    shape_func_b_mechanical, shape_func_b_times_det_J_b_time_weight_mechanical,\
                                    grad_shape_func_b_x_mechanical, grad_shape_func_b_x_times_det_J_b_time_weight_mechanical,\
                                    grad_shape_func_b_y_mechanical, grad_shape_func_b_y_times_det_J_b_time_weight_mechanical,\
                                    grad_shape_func_b_z_mechanical, grad_shape_func_b_z_times_det_J_b_time_weight_mechanical)
        comp_mechanical_stiffness_matrix = time.time()

        print('time to compute the mechanical stiffness matrix = ' + "%s seconds" % (comp_mechanical_stiffness_matrix-start_mechanical_time))
        
        c_G_domain_electrolyte = np.ones(num_gauss_points_in_domain_electrolyte)*c_boundary
        c_G_domain = np.concatenate((c_G_domain_electrolyte, shape_func_electrode*C_new_electrode))
        

        # compute Beta (expansion coeffieicent)
        beta_fuelcell_expansion_coefficient_electrolyte = np.zeros(num_gauss_points_in_domain_electrolyte)
        beta_fuelcell_expansion_coefficient_electrode = beta_fuelcell_expansion_coefficient*np.ones(num_gauss_points_in_domain_electrode)
        beta_1 = np.concatenate((beta_fuelcell_expansion_coefficient_electrolyte, beta_fuelcell_expansion_coefficient_electrode)).reshape(num_gauss_points_in_domain_mechanical,1)*(c_G_domain.reshape(num_gauss_points_in_domain_mechanical,1)-c_boundary)
        beta_2 = np.concatenate((beta_fuelcell_expansion_coefficient_electrolyte, beta_fuelcell_expansion_coefficient_electrode)).reshape(num_gauss_points_in_domain_mechanical,1)*(c_G_domain.reshape(num_gauss_points_in_domain_mechanical,1)-c_boundary)
        beta_3 = np.concatenate((beta_fuelcell_expansion_coefficient_electrolyte, beta_fuelcell_expansion_coefficient_electrode)).reshape(num_gauss_points_in_domain_mechanical,1)*(c_G_domain.reshape(num_gauss_points_in_domain_mechanical,1)-c_boundary)
        
        

        epsilon_D1 = T_c[0,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[0,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[0,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3
        epsilon_D2 = T_c[1,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[1,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[1,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3
        epsilon_D3 = T_c[2,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[2,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[2,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3
        epsilon_D4 = 2*(T_c[3,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[3,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[3,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3)
        epsilon_D5 = 2*(T_c[4,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[4,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[4,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3)
        epsilon_D6 = 2*(T_c[5,0].reshape(num_gauss_points_in_domain_mechanical,1)*beta_1+T_c[5,1].reshape(num_gauss_points_in_domain_mechanical,1)*beta_2+T_c[5,2].reshape(num_gauss_points_in_domain_mechanical,1)*beta_3)
        


        # solve the mechenical part without damage
        f_mechanical = mechanical_force_matrix_3d(x_G_mechanical, C, epsilon_D1, epsilon_D2, epsilon_D3,epsilon_D4, epsilon_D5, epsilon_D6, grad_shape_func_x_times_det_J_time_weight_mechanical, grad_shape_func_y_times_det_J_time_weight_mechanical,grad_shape_func_z_times_det_J_time_weight_mechanical)
        
        # if ii==0:
        comp_mechanical_force_matrix = time.time()

        print('time to compute mechanical force matrix= ' + "%s seconds" % (comp_mechanical_force_matrix-comp_mechanical_stiffness_matrix))

        
        # solve displacement field
        u_disp = spsolve(K_mechanical, f_mechanical)    #1d array

        solve_mechanical_matrix = time.time()

        print('time to solve mechanical matrix= ' + "%s seconds" % (solve_mechanical_matrix-comp_mechanical_force_matrix))

        ux = u_disp[0:num_nodes_mechanical]          # disp at nodes along x
        uy = u_disp[num_nodes_mechanical:2*num_nodes_mechanical]           # disp at nodes along y
        uz = u_disp[num_nodes_mechanical*2:]           # disp at nodes along y

        ux_gauss = shape_func_mechanical*ux
        uy_gauss = shape_func_mechanical*uy
        uz_gauss = shape_func_mechanical*uz

        # predict the damage factor:
        """
        (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
        epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
        !!!!! reshape (grad_shape_func_x*ux)
        """

        epsilon_x = (grad_shape_func_x_mechanical*ux).reshape(num_gauss_points_in_domain_mechanical,1)       # normal strain along x at all gauss points
        epsilon_y = (grad_shape_func_y_mechanical*uy).reshape(num_gauss_points_in_domain_mechanical,1)       # normal strain along y at all gauss points
        epsilon_z = (grad_shape_func_z_mechanical*uz).reshape(num_gauss_points_in_domain_mechanical,1)       # normal strain along x at all gauss points
        
        gamma_xy = ((grad_shape_func_x_mechanical*uy+ grad_shape_func_y_mechanical*ux)*0.5).reshape(num_gauss_points_in_domain_mechanical,1)      # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array
        gamma_xz = ((grad_shape_func_x_mechanical*uz+ grad_shape_func_z_mechanical*ux)*0.5).reshape(num_gauss_points_in_domain_mechanical,1)      # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array
        gamma_yz = ((grad_shape_func_z_mechanical*uy+ grad_shape_func_y_mechanical*uz)*0.5).reshape(num_gauss_points_in_domain_mechanical,1)      # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

        epsilon_x_mechanical = epsilon_x - epsilon_D1
        epsilon_y_mechanical = epsilon_y - epsilon_D2
        epsilon_z_mechanical = epsilon_z - epsilon_D3
        gamma_xy_mechanical = gamma_xy-epsilon_D4/2
        gamma_xz_mechanical = gamma_xz-epsilon_D5/2
        gamma_yz_mechanical = gamma_yz-epsilon_D6/2


        # calculate the principle strain:
        epsilon_e_eq = (2.0/3.0*((epsilon_x_mechanical-epsilon_y_mechanical)**2+(epsilon_x_mechanical-epsilon_z_mechanical)**2+(epsilon_y_mechanical-epsilon_z_mechanical)**2+6*(gamma_xy_mechanical**2+gamma_xz_mechanical**2+gamma_yz_mechanical**2)))**0.5

    print(np.shape(epsilon_e_eq))
    k = epsilon_e_eq

    D_damage[np.logical_and(k>k_i, k<=k_f)] = (k[np.logical_and(k>k_i, k<=k_f)]-k_i)/(k_f-k_i)*k_f/k[np.logical_and(k>k_i, k<=k_f)]
    D_damage[k>k_f] = 1.0
    D_damage[k<=k_i] = 0.0
    
    print(np.shape(x_G_mechanical))
    print(np.shape(D_damage))

    phi_on_nodes_electrolyte=shape_func_n_nodes_n_nodes_electrolyte*phi_new_electrolyte
    C_on_nodes_electrode=shape_func_n_nodes_n_nodes_electrode*C_new_electrode
    C_on_nodes_pore=shape_func_n_nodes_n_nodes_pore*C_new_pore

    phi_on_GP_electrolyte = shape_func_electrolyte*phi_new_electrolyte
    C_on_GP_electrode = shape_func_electrode*C_new_electrode
    C_on_GP_pore = shape_func_pore*C_new_pore

    print('on nodes:', np.max(phi_on_nodes_electrolyte))
    print('on nodes:', np.max(C_on_nodes_electrode))
    print('on nodes:', np.max(C_on_nodes_pore))

    print('on GP:', np.max(phi_on_GP_electrolyte))
    print('on GP:', np.max(C_on_GP_electrode))
    print('on GP:', np.max(C_on_GP_pore))

    
    if dimention == 3:
    
        potential_on_nodes_save_electrolyte = np.zeros((num_nodes_electrolyte, 4))
        potential_on_nodes_save_electrolyte[:,:3] = x_nodes_electrolyte
        potential_on_nodes_save_electrolyte[:,3] = phi_on_nodes_electrolyte

        potential_on_GP_save_electrolyte = np.zeros((num_gauss_points_in_domain_electrolyte, 4))
        potential_on_GP_save_electrolyte[:,:3] = x_G_electrolyte
        potential_on_GP_save_electrolyte[:,3] = shape_func_electrolyte*phi_new_electrolyte

        C_on_nodes_save_electrode = np.zeros((num_nodes_electrode, 4))
        C_on_nodes_save_electrode[:,:3] = x_nodes_electrode
        C_on_nodes_save_electrode[:,3] = C_on_nodes_electrode

        C_on_GP_save_electrode = np.zeros((num_gauss_points_in_domain_electrode, 4))
        C_on_GP_save_electrode[:,:3] = x_G_electrode
        C_on_GP_save_electrode[:,3] = shape_func_electrode*C_new_electrode

        C_on_nodes_save_pore = np.zeros((num_nodes_pore, 4))
        C_on_nodes_save_pore[:,:3] = x_nodes_pore
        C_on_nodes_save_pore[:,3] = C_new_pore

        C_on_GP_save_pore = np.zeros((num_gauss_points_in_domain_pore, 4))
        C_on_GP_save_pore[:,:3] = x_G_pore
        C_on_GP_save_pore[:,3] = shape_func_pore*C_new_pore


        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')
        sc = ax.scatter(potential_on_nodes_save_electrolyte[:, 0], potential_on_nodes_save_electrolyte[:, 1],potential_on_nodes_save_electrolyte[:, 2], c=potential_on_nodes_save_electrolyte[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Potential on Nodes - Electrolyte')

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        sc = ax.scatter(C_on_nodes_save_electrode[:, 0], C_on_nodes_save_electrode[:, 1], C_on_nodes_save_electrode[:, 2], c=C_on_nodes_save_electrode[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Concentration on Nodes - Electrode')

        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')
        sc = ax.scatter(C_on_nodes_save_pore[:, 0], C_on_nodes_save_pore[:, 1], C_on_nodes_save_pore[:, 2], c=C_on_nodes_save_pore[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Concentration on Nodes - Pore')

        fig4 = plt.figure()
        ax = fig4.add_subplot(111, projection='3d')
        sc = ax.scatter(potential_on_GP_save_electrolyte[:, 0], potential_on_GP_save_electrolyte[:, 1],potential_on_GP_save_electrolyte[:, 2], c=potential_on_GP_save_electrolyte[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Potential on GP - Electrolyte')

        fig5 = plt.figure()
        ax = fig5.add_subplot(111, projection='3d')
        sc = ax.scatter(C_on_GP_save_electrode[:, 0], C_on_GP_save_electrode[:, 1], C_on_GP_save_electrode[:, 2], c=C_on_GP_save_electrode[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Concentration on GP - Electrode')

        fig6 = plt.figure()
        ax = fig6.add_subplot(111, projection='3d')
        sc = ax.scatter(C_on_GP_save_pore[:, 0], C_on_GP_save_pore[:, 1], C_on_GP_save_pore[:, 2], c=C_on_GP_save_pore[:, 3])
        plt.colorbar(sc, ax=ax)
        plt.title('Concentration on GP - Pore')

        fig7 = plt.figure()
        ax = fig7.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=ux_gauss)
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement ux')

        fig8 = plt.figure()
        ax = fig8.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=uy_gauss)
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement uy')

        fig9 = plt.figure()
        ax = fig9.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=uz_gauss)
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement uz')

        fig10 = plt.figure()
        ax = fig10.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_electrode[:, 0], x_G_electrode[:, 1], x_G_electrode[:, 2], c=ux_gauss[num_gauss_points_in_domain_electrolyte:])
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement ux')

        fig11 = plt.figure()
        ax = fig11.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_electrode[:, 0], x_G_electrode[:, 1], x_G_electrode[:, 2], c=uy_gauss[num_gauss_points_in_domain_electrolyte:])
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement uy')

        fig12 = plt.figure()
        ax = fig12.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_electrode[:, 0], x_G_electrode[:, 1], x_G_electrode[:, 2], c=uz_gauss[num_gauss_points_in_domain_electrolyte:])
        plt.colorbar(sc, ax=ax)
        plt.title('Displacement uz')

        fig13 = plt.figure()
        ax = fig13.add_subplot(111, projection='3d')
        sc = ax.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=D_damage)
        plt.colorbar(sc, ax=ax)
        plt.title('Damage Factor')

        plt.show()


    

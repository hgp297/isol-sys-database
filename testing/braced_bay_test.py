############################################################################
#               Testing for a single braced bay

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: January 2023

# Description:  Various test for single HSS brace

# Open issues:  

############################################################################
# UTILITIES
############################################################################
def get_shape(shape_name, member, csv_dir='../resource/'):
    import pandas as pd
    
    if member == 'beam':
        shape_db = pd.read_csv(csv_dir+'beamShapes.csv',
                               index_col=None, header=0)
    elif member == 'column':
        shape_db = pd.read_csv(csv_dir+'colShapes.csv',
                               index_col=None, header=0)
    elif member == 'brace':
        shape_db = pd.read_csv(csv_dir+'braceShapes.csv',
                               index_col=None, header=0)  
    shape = shape_db.loc[shape_db['AISC_Manual_Label'] == shape_name]
    return(shape)

def get_properties(shape):
    Ag      = float(shape.iloc[0]['A'])
    Ix      = float(shape.iloc[0]['Ix'])
    Iy      = float(shape.iloc[0]['Iy'])
    Zx      = float(shape.iloc[0]['Zx'])
    Sx      = float(shape.iloc[0]['Sx'])
    d       = float(shape.iloc[0]['d'])
    bf      = float(shape.iloc[0]['bf'])
    tf      = float(shape.iloc[0]['tf'])
    tw      = float(shape.iloc[0]['tw'])
    return(Ag, Ix, Iy, Zx, Sx, d, bf, tf, tw)

def bot_gp_coord(nd, L_bay, h_story, offset=0.25):
    # from node number, get the parent node it's attached to
    bot_nd = nd//100
    
    # get the bottom node's coordinates
    bot_x_coord = (bot_nd%10)*L_bay
    bot_y_coord = (bot_nd//10 - 1)*h_story
    
    # if last number is 1 or 2, brace connects nw
    # if last number is 3 or 4, brace connects ne
    goes_ne = [3, 4]
    if (nd%10 in goes_ne):
        x_offset = offset/2*L_bay/2
    else:
        x_offset = -offset/2*L_bay/2
    
    y_offset = offset/2 * h_story
    gp_x_coord = bot_x_coord + x_offset
    gp_y_coord = bot_y_coord + y_offset
    
    return(gp_x_coord, gp_y_coord)


def top_gp_coord(nd, L_bay, h_story, offset=0.25):
    # from node number, get the parent node it's attached to
    top_node = nd//100
    
    # extract their corresponding coordinates from the node numbers
    top_x_coord = (top_node%10 + 0.5)*L_bay
    top_y_coord = (top_node//10 - 1)*h_story
    
    # if last number is 1 or 5, brace connects se
    # if last number is 2 or 6, brace connects sw
    if (nd % 10)%2 == 0:
        x_offset = -offset/2*L_bay/2
    else:
        x_offset = offset/2*L_bay/2
    
    y_offset = -offset/2 * h_story
    gp_x_coord = top_x_coord + x_offset
    gp_y_coord = top_y_coord + y_offset
    
    return(gp_x_coord, gp_y_coord)
    

def mid_brace_coord(nd, L_bay, h_story, camber=0.001, offset=0.25):
    # from mid brace number, get the corresponding top and bottom node numbers
    top_node = nd//100
    
    # extract their corresponding coordinates from the node numbers
    top_x_coord = (top_node%10 + 0.5)*L_bay
    top_y_coord = (top_node//10 - 1)*h_story
    
    # if the last number is 8, the brace connects sw
    # if the last number is 7, the brace connects se
    
    if (nd % 10)%2 == 0:
        bot_node = top_node - 10
        x_offset = offset/2 * L_bay/2
    else:
        bot_node = top_node - 9
        x_offset = - offset/2 * L_bay/2
    
    # get the bottom node's coordinates
    bot_x_coord = (bot_node%10)*L_bay
    bot_y_coord = (bot_node//10 - 1)*h_story
    
    # effective length is 90% of the diagonal (gusset plate offset)
    br_x = abs(top_x_coord - bot_x_coord)
    br_y = abs(top_y_coord - bot_y_coord)
    L_eff = (1-offset)*(br_x**2 + br_y**2)**0.5
    
    # angle from horizontal up to brace vector
    from math import atan, asin, sin, cos
    theta = atan(h_story/(L_bay/2))
    
    # angle from the brace vector up to camber
    beta = asin(2*camber)
    
    # angle from horizontal up to camber
    gamma  = theta + beta
    
    # origin is bottom node, adjusted for gusset plate
    # offset is the shift (+/-) of the bottom gusset plate
    # terminus is top node, adjusted for gusset plate (gusset placed opposite direction)
    x_origin = bot_x_coord + x_offset
    # x_terminus = top_x_coord - x_offset
    
    y_offset = offset/2 * h_story
    y_origin = bot_y_coord + y_offset
    # y_terminus = top_y_coord - y_offset
    
    # if the last number is 8, the brace connects sw
    # if the last number is 7, the brace connects se
    if (nd % 10)%2 == 0:
        mid_x_coord = x_origin + L_eff/2 * cos(gamma)
    else:
        mid_x_coord = x_origin - L_eff/2 * cos(gamma)
    mid_y_coord = y_origin + L_eff/2 * sin(gamma)
    
    return(mid_x_coord, mid_y_coord)

############################################################################
# Construct braced bay
############################################################################
import pandas as pd
from building import Building
test_d = {'superstructure_system':'CBF', 'num_bays':1, 'num_stories':1}
test_design = pd.Series(data=test_d, 
                          index=['superstructure_system', 'num_bays', 'num_stories'])
bldg = Building(test_design)
bldg.number_nodes()
# import OpenSees and libraries
import openseespy.opensees as ops

# remove existing model
ops.wipe()

# units: in, kip, s
# dimensions
inch    = 1.0
ft      = 12.0*inch
sec     = 1.0
g       = 386.4*inch/(sec**2)
kip     = 1.0
ksi     = kip/(inch**2)

L_bay = 30.0 * ft     # ft to in
h_story = 13.0 * ft

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# model gravity masses corresponding to the frame placed on building edge
import numpy as np

# nominal change
L_beam = L_bay
L_col = h_story

col_list = ['W12X79']
beam_list = ['W40X149']
brace_list = ['HSS9X9X1/4']
# brace_list = ['HSS4X4X1/2']
selected_col = get_shape('W12X79', 'column')
selected_beam = get_shape('W40X149', 'beam')
selected_brace = get_shape('HSS9X9X1/4', 'brace')
# selected_brace = get_shape('HSS4X4X1/2', 'brace')

(Ag_col, Iz_col, Iy_col,
 Zx_col, Sx_col, d_col,
 bf_col, tf_col, tw_col) = get_properties(selected_col)

(Ag_beam, Iz_beam, Iy_beam,
 Zx_beam, Sx_beam, d_beam,
 bf_beam, tf_beam, tw_beam) = get_properties(selected_beam)

# base node
base_nodes = [10, 11]
for idx, nd in enumerate(base_nodes):
    ops.node(nd, (idx)*L_beam, 0.0*ft, 0.0*ft)
    ops.fix(nd, 1, 1, 1, 1, 1, 1)

# end node
w_floor = 2.087/12
m_grav_outer = w_floor * L_bay / 2 /g
m_grav_brace = w_floor * L_bay / 2 /g
floor_nodes = [20, 21]
for nd in floor_nodes:
    
    # get multiplier for location from node number
    bay = nd%10
    fl = (nd//10)%10 - 1
    ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
    
    # assign masses, in direction of motion and stiffness
    # DOF list: X, Y, Z, rotX, rotY, rotZ
    m_nd = m_grav_outer
    negligible = 1e-15
    ops.mass(nd, m_nd, m_nd, m_nd,
             negligible, negligible, negligible)
    
    # restrain out of plane motion
    ops.fix(nd, 0, 1, 0, 1, 0, 1)


# midbay brace top node


brace_top_nodes = bldg.node_tags['brace_top']
for nd in brace_top_nodes:
    parent_node = nd // 10
    
    # extract their corresponding coordinates from the node numbers
    fl = parent_node//10 - 1
    x_coord = (parent_node%10 + 0.5)*L_beam
    z_coord = fl*L_col
    
    m_nd = m_grav_brace
    ops.node(nd, x_coord, 0.0*ft, z_coord)
    ops.mass(nd, m_nd, m_nd, m_nd,
             negligible, negligible, negligible)

# brace nodes
ofs = 0.25
L_diag = ((L_bay/2)**2 + L_col**2)**(0.5)
# L_eff = (1-ofs) * L_diag
L_gp = ofs/2 * L_diag

# brace nodes
brace_mid_nodes = bldg.node_tags['brace_mid']
for nd in brace_mid_nodes:
    
    # values returned are already in inches
    x_coord, z_coord = mid_brace_coord(nd, L_beam, L_col, offset=ofs)
    ops.node(nd, x_coord, 0.0*ft, z_coord)
    
# spring nodes
spring_nodes = bldg.node_tags['spring']
brace_beam_ends = bldg.node_tags['brace_beam_end']
brace_beam_tab_nodes = bldg.node_tags['brace_beam_tab']
brace_bot_nodes = bldg.node_tags['brace_bottom']

col_brace_bay_node = [nd for nd in spring_nodes
                      if (nd//10 in brace_beam_ends 
                          or nd//10 in brace_bot_nodes)
                      and (nd%10 == 6 or nd%10 == 8)]

beam_brace_bay_node = [nd//10*10+9 if nd%10 == 0
                       else nd//10*10+7 for nd in brace_beam_tab_nodes]

for nd in spring_nodes:
    parent_nd = nd//10
    
    # get multiplier for location from node number
    bay = parent_nd%10
    fl = parent_nd//10 - 1
    
    # "springs" inside the brace frames should be treated differently
    # if it's a column spring, the offset should be dbeam/2 if it's below the column node
    # if it's above the column node, there is a GP node attached to it
    # roughly, we put it 1.2x L_gp, where L_gp is the diagonal offset of the gusset plate
    
    if nd in col_brace_bay_node:
        if nd%10 == 6:
            y_offset = d_beam/2
            ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col-y_offset)
        else:
            y_offset = 1.2*L_gp
            ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col+y_offset)
            
    # if it's a beam spring, place it +/- d_col to the right/left of the column node
    elif nd in beam_brace_bay_node:
        x_offset = d_col/2
        if nd%10 == 7:
            ops.node(nd, bay*L_beam-x_offset, 0.0*ft, fl*L_col) 
        else:
            ops.node(nd, bay*L_beam+x_offset, 0.0*ft, fl*L_col)
            
    # otherwise, it is a gravity frame node and can just overlap the main node
    else:
        ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)

brace_beam_spr_nodes = bldg.node_tags['brace_beam_spring']
for nd in brace_beam_spr_nodes:
    grandparent_nd = nd//100
    
    # extract their corresponding coordinates from the node numbers
    x_offset = 1.2*L_gp
    fl = grandparent_nd//10 - 1
    x_coord = (grandparent_nd%10 + 0.5)*L_beam
    z_coord = fl*L_col
    
    # place the node with the offset l/r of midpoint according to suffix
    if nd%10 == 3:
        ops.node(nd, x_coord-x_offset, 0.0*ft, z_coord)
    else:
        ops.node(nd, x_coord+x_offset, 0.0*ft, z_coord)
    
for nd in brace_beam_tab_nodes:
    parent_nd = nd//10
    
    # get multiplier for location from node number
    bay = parent_nd%10
    fl = parent_nd//10 - 1
    
    x_offset = d_col/2
    if nd%10 == 5:
        ops.node(nd, bay*L_beam-x_offset, 0.0*ft, fl*L_col) 
    else:
        ops.node(nd, bay*L_beam+x_offset, 0.0*ft, fl*L_col)

# each end has offset/2*L_diagonal assigned to gusset plate offset
brace_bot_gp_nodes = bldg.node_tags['brace_bottom_spring']
for nd in brace_bot_gp_nodes:
    
    # values returned are already in inches
    x_coord, z_coord = bot_gp_coord(nd, L_beam, L_col, offset=ofs)
    ops.node(nd, x_coord, 0.0*ft, z_coord)

brace_top_gp_nodes = bldg.node_tags['brace_top_spring']
for nd in brace_top_gp_nodes:
    # values returned are already in inches
    x_coord, z_coord = top_gp_coord(nd, L_beam, L_col, offset=ofs)
    ops.node(nd, x_coord, 0.0*ft, z_coord)

print('Nodes placed.')
############################################################################
# Materials 
############################################################################

# General elastic section (non-plastic beam columns, leaning columns)
lc_spring_mat_tag = 51
elastic_mat_tag = 52
torsion_mat_tag = 53
ghost_mat_tag = 54

# Steel material tag
steel_mat_tag = 31
gp_mat_tag = 32
steel_no_fatigue = 33

# isolator tags
friction_1_tag = 81
friction_2_tag = 82
fps_vert_tag = 84
fps_rot_tag = 85

# Impact material tags
impact_mat_tag = 91

# reserve blocks of 10 for integration and section tags
col_sec = 110
col_int = 150

beam_sec = 120
beam_int = 160

br_sec = 130
br_int = 170

# define material: steel
Es  = 29000*ksi     # initial elastic tangent
nu  = 0.2          # Poisson's ratio
Gs  = Es/(1 + nu) # Torsional stiffness modulus
J   = 1e10          # Set large torsional stiffness

# Frame link (stiff elements)
A_rigid = 1000.0         # define area of truss section
I_rigid = 1e6        # moment of inertia for p-delta columns
ops.uniaxialMaterial('Elastic', elastic_mat_tag, Es)

# minimal stiffness elements (ghosts)
A_ghost = 1.
E_ghost = 100.0
ops.uniaxialMaterial('Elastic', ghost_mat_tag, E_ghost)

# define material: Steel02
# command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
Fy  = 50*ksi        # yield strength
b   = 0.003           # hardening ratio
R0 = 15
cR1 = 0.925
cR2 = 0.15
ops.uniaxialMaterial('Elastic', torsion_mat_tag, J)
ops.uniaxialMaterial('Steel02', steel_no_fatigue, Fy, Es, b, R0, cR1, cR2)
ops.uniaxialMaterial('Fatigue', steel_mat_tag, steel_no_fatigue)

# GP section: thin plate
W_w = (L_gp**2 + L_gp**2)**0.5
L_avg = 0.75* L_gp
t_gp = 1.375*inch
Fy_gp = 50*ksi

My_GP = (W_w*t_gp**2/6)*Fy_gp
K_rot_GP = Es/L_avg * (W_w*t_gp**3/12)
b_GP = 0.01
ops.uniaxialMaterial('Steel02', gp_mat_tag, My_GP, K_rot_GP, b_GP, R0, cR1, cR2)

################################################################################
# define column
################################################################################

# geometric transformation for beam-columns
# command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
beam_transf_tag   = 1
col_transf_tag    = 2
brace_beam_transf_tag = 3
brace_transf_tag_L = 4
brace_transf_tag_R = 5

# this is different from moment frame
# beam geometry
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(11)
beam_x_axis = np.subtract(xyz_j, xyz_i)
vecxy_beam = [0, 0, 1] # Use any vector in local x-y, but not local x
vecxz = np.cross(beam_x_axis,vecxy_beam) # What OpenSees expects
vecxz_beam = vecxz / np.sqrt(np.sum(vecxz**2))


# column geometry
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(20)
col_x_axis = np.subtract(xyz_j, xyz_i)
vecxy_col = [1, 0, 0] # Use any vector in local x-y, but not local x
vecxz = np.cross(col_x_axis,vecxy_col) # What OpenSees expects
vecxz_col = vecxz / np.sqrt(np.sum(vecxz**2))

# brace geometry (we can use one because HSS is symmetric)
brace_top_nodes = bldg.node_tags['brace_top']
xyz_i = ops.nodeCoord(brace_top_nodes[0]//10 - 10)
xyz_j = ops.nodeCoord(brace_top_nodes[0])
brace_x_axis_L = np.subtract(xyz_j, xyz_i)
brace_x_axis_L = brace_x_axis_L / np.sqrt(np.sum(brace_x_axis_L**2))
vecxy_brace = [0, 1, 0] # Use any vector in local x-y, but not local x
vecxz = np.cross(brace_x_axis_L,vecxy_brace) # What OpenSees expects
vecxz_brace_L = vecxz / np.sqrt(np.sum(vecxz**2))

# brace geometry (we can use one because HSS is symmetric)
xyz_i = ops.nodeCoord(brace_top_nodes[0]//10 - 10 + 1)
xyz_j = ops.nodeCoord(brace_top_nodes[0])
brace_x_axis_R = np.subtract(xyz_j, xyz_i)
brace_x_axis_R = brace_x_axis_R / np.sqrt(np.sum(brace_x_axis_R**2))
vecxy_brace = [0, 1, 0] # Use any vector in local x-y, but not local x
vecxz = np.cross(brace_x_axis_R,vecxy_brace) # What OpenSees expects
vecxz_brace_R = vecxz / np.sqrt(np.sum(vecxz**2))

ops.geomTransf('PDelta', brace_beam_transf_tag, *vecxz_beam) # beams
ops.geomTransf('PDelta', beam_transf_tag, *vecxz_beam) # beams
ops.geomTransf('PDelta', col_transf_tag, *vecxz_col) # columns
ops.geomTransf('Corotational', brace_transf_tag_L, *vecxz_brace_L) # braces
ops.geomTransf('Corotational', brace_transf_tag_R, *vecxz_brace_R) # braces

# outside of concentrated plasticity zones, use elastic beam columns

# Fiber section parameters
nfw = 4     # number of fibers in web
nff = 4     # number of fibers in each flange

n_IP = 4
###################### columns #############################

for fl_col, col in enumerate(col_list):
    current_col = get_shape(col, 'column')
    
    (Ag_col, Iz_col, Iy_col,
     Zx_col, Sx_col, d_col,
     bf_col, tf_col, tw_col) = get_properties(current_col)
    
    # column section: fiber wide flange section
    # match the tag number with the floor's node number
    # for column, this is the bottom node (col between 10 and 20 has tag 111)
    # e.g. first col bot nodes at 3x -> tag 113 and 153
    
    current_col_sec = col_sec + fl_col + 1
    ops.section('Fiber', current_col_sec, '-GJ', Gs*J)
    ops.patch('rect', steel_mat_tag, 
        1, nff,  d_col/2-tf_col, -bf_col/2, d_col/2, bf_col/2)
    ops.patch('rect', steel_mat_tag, 
        1, nff, -d_col/2, -bf_col/2, -d_col/2+tf_col, bf_col/2)
    ops.patch('rect', steel_mat_tag,
        nfw, 1, -d_col/2+tf_col, -tw_col/2, d_col/2-tf_col, tw_col/2)
    
    
    current_col_int = col_int + fl_col + 1
    ops.beamIntegration('Lobatto', current_col_int, 
                        current_col_sec, n_IP)

# define the columns

col_id = bldg.elem_ids['col']
col_elems = bldg.elem_tags['col']

# find which columns belong to the braced bays
# (if its i-node parent is a brace_bottom_node)
col_br_elems = [col for col in col_elems
                if col-col_id in brace_bot_nodes]

for elem_tag in col_br_elems:
    i_nd = (elem_tag - col_id)*10 + 8
    j_nd = (elem_tag - col_id + 10)*10 + 6
    col_floor = i_nd // 100
    
    col_int_tag = col_floor + col_int
    ops.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
                col_transf_tag, col_int_tag)
    
###################### beams #############################

for fl_beam, beam in enumerate(beam_list):
    current_beam = get_shape(beam, 'beam')
    
    (Ag_beam, Iz_beam, Iy_beam,
     Zx_beam, Sx_beam, d_beam,
     bf_beam, tf_beam, tw_beam) = get_properties(current_beam)
    
    # beam section: fiber wide flange section
    # match the tag number with the floor's node number
    # e.g. first beams nodes at 2x -> tag 132 and 172
    current_brace_beam_sec = beam_sec + fl_beam + 2
    
    ops.section('Fiber', current_brace_beam_sec, '-GJ', Gs*J)
    ops.patch('rect', steel_mat_tag, 
        1, nff,  d_beam/2-tf_beam, -bf_beam/2, d_beam/2, bf_beam/2)
    ops.patch('rect', steel_mat_tag, 
        1, nff, -d_beam/2, -bf_beam/2, -d_beam/2+tf_beam, bf_beam/2)
    ops.patch('rect', steel_mat_tag,
        nfw, 1, -d_beam/2+tf_beam, -tw_beam/2, d_beam/2-tf_beam, tw_beam/2)
    
    
    current_brace_beam_int = beam_int + fl_beam + 2
    ops.beamIntegration('Lobatto', current_brace_beam_int, 
                        current_brace_beam_sec, n_IP)

brace_beam_id = bldg.elem_ids['brace_beam']
brace_beam_elems = bldg.elem_tags['brace_beams']

for elem_tag in brace_beam_elems:
    parent_i_nd = (elem_tag - brace_beam_id)
    
    # determine if the left node is a mid-span or a main node
    # remap to the e/w node correspondingly
    if parent_i_nd > 100:
        i_nd = parent_i_nd*10 + 4
        j_nd = (parent_i_nd//10 + 1)*10 + 5
        beam_floor = parent_i_nd // 100
    else:
        i_nd = parent_i_nd*10
        j_nd = (parent_i_nd*10 + 1)*10 + 3
        beam_floor = parent_i_nd // 10
        
    brace_beam_int_tag = beam_floor + beam_int
    ops.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
                brace_beam_transf_tag, brace_beam_int_tag)
    
###################### Brace #############################

# starting from bottom floor, define the brace shape for that floor
# floor 1's brace at 141 and 161, etc.
for fl_br, brace in enumerate(brace_list):
    current_brace = get_shape(brace, 'brace')
    d_brace = current_brace.iloc[0]['b']
    t_brace = current_brace.iloc[0]['tdes']
    
    # brace section: HSS section
    brace_sec_tag = br_sec + fl_br + 1
    
    ops.section('Fiber', brace_sec_tag, '-GJ', Gs*J)
    ops.patch('rect', steel_mat_tag, 1, nff,  
              d_brace/2-t_brace, -d_brace/2, d_brace/2, d_brace/2)
    ops.patch('rect', steel_mat_tag, 1, nff, 
              -d_brace/2, -d_brace/2, -d_brace/2+t_brace, d_brace/2)
    ops.patch('rect', steel_mat_tag, nfw, 1, 
              -d_brace/2+t_brace, -d_brace/2, d_brace/2-t_brace, -d_brace/2+t_brace)
    ops.patch('rect', steel_mat_tag, nfw, 1, 
              -d_brace/2+t_brace, d_brace/2-t_brace, d_brace/2-t_brace, d_brace/2)
    
    brace_int_tag = br_int + fl_br + 1
    ops.beamIntegration('Lobatto', brace_int_tag, 
                        brace_sec_tag, n_IP)

brace_id = bldg.elem_ids['brace']
brace_elems = bldg.elem_tags['brace']

for elem_tag in brace_elems:
    
    # if tag is 02 or 04, it extends from the bottom up
    if elem_tag%10 == 4:
        i_nd = (elem_tag - brace_id)
        parent_i_nd = i_nd // 100 
        j_nd = (parent_i_nd + 10)*100 + 18
    elif elem_tag%10 == 2:
        i_nd = (elem_tag - brace_id)
        parent_i_nd = i_nd // 100 
        j_nd = (parent_i_nd + 9)*100 + 17
    else:
        i_nd = (elem_tag - brace_id) + 2
        j_nd = elem_tag - brace_id
        
    # ending node is always numbered with parent as floor j_floor
    j_floor = j_nd//1000
    
    current_brace_int = j_floor - 1 + br_int
    if (i_nd%10==4) or (i_nd%10==8):
        brace_transf_tag = brace_transf_tag_L
    elif (i_nd%10==2) or (i_nd%10==7):
        brace_transf_tag = brace_transf_tag_R
    ops.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
                brace_transf_tag, current_brace_int)
    
# add ghost trusses to the braces to reduce convergence problems
brace_ghosts = bldg.elem_tags['brace_ghosts']
for elem_tag in brace_ghosts:
    i_nd = (elem_tag - 5) - brace_id
    
    parent_i_nd = i_nd // 100
    if elem_tag%10 == 9:
        j_nd = (parent_i_nd + 10)*100 + 16
    else:
        j_nd = (parent_i_nd + 9)*100 + 15
    ops.element('corotTruss', elem_tag, i_nd, j_nd, A_ghost, ghost_mat_tag)
    
###################### Gusset plates #############################

brace_spr_id = bldg.elem_ids['brace_spring']
brace_top_links = bldg.elem_tags['brace_top_springs']
brace_bot_links = bldg.elem_tags['brace_bot_springs']

# on bottom, the outer (GP non rigid) nodes are 2 and 4
brace_bot_gp_spring_link = [link for link in brace_bot_links
                            if link%2 == 0]

for link_tag in brace_bot_gp_spring_link:
    i_nd = (link_tag - brace_spr_id) - 1
    j_nd = (link_tag - brace_spr_id)
    
    # put the correct local x-axis
    # torsional stiffness around local-x, GP stiffness around local-y
    # since imperfection is in x-z plane, we allow GP-stiff rotation 
    # pin around y to enable buckling
    if link_tag%10 == 4:
        ops.element('zeroLength', link_tag, i_nd, j_nd,
            '-mat', torsion_mat_tag, gp_mat_tag, 
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_L, *vecxy_brace)
    else:
        ops.element('zeroLength', link_tag, i_nd, j_nd,
            '-mat', torsion_mat_tag, gp_mat_tag, 
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_R, *vecxy_brace)
        
    # global z-rotation is restrained
    ops.equalDOF(i_nd, j_nd, 1, 2, 3, 6)
    
# at top, outer (GP non rigid nodes are 5 and 6)
brace_top_gp_spring_link = [link for link in brace_top_links
                            if link%10 > 4]

for link_tag in brace_top_gp_spring_link:
    i_nd = (link_tag - brace_spr_id)
    j_nd = (link_tag - brace_spr_id) - 4
    
    # put the correct local x-axis
    # torsional stiffness around local-x, GP stiffness around local-z
    if link_tag%10 == 6:
        ops.element('zeroLength', link_tag, i_nd, j_nd,
            '-mat', torsion_mat_tag, gp_mat_tag, 
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_L, *vecxy_brace)
    else:
        ops.element('zeroLength', link_tag, i_nd, j_nd,
            '-mat', torsion_mat_tag, gp_mat_tag, 
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_R, *vecxy_brace)
        
    # global z-rotation is restrained
    ops.equalDOF(j_nd, i_nd, 1, 2, 3, 6)
    
################################################################################
# define rigid links in the braced bays
################################################################################  

spr_id = bldg.elem_ids['spring']
spr_elems = bldg.elem_tags['spring']

# extract where rigid elements are in the entire frame
# brace_beam_end_joint = [link for link in spr_elems
#                         if (link-spr_id)//10 in brace_beam_ends
#                         and (link%10 == 9 or link%10 == 7)]


brace_beam_middle_joint = bldg.elem_tags['brace_beam_springs']

col_joint = [link for link in spr_elems
             if ((link-spr_id)//10 in brace_beam_ends 
                 or (link-spr_id)//10 in brace_bot_nodes)
             and (link%10 == 6 or link%10 == 8)]

# make link for beam around where the braces connect
for link_tag in brace_beam_middle_joint:
    outer_nd = link_tag - brace_spr_id
    
    if outer_nd%2 == 0:
        i_nd = outer_nd // 10
        j_nd = outer_nd
    else:
        i_nd = outer_nd
        j_nd = outer_nd // 10
        
    ops.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                brace_beam_transf_tag)
  
# make link for all column in braced bays
for link_tag in col_joint:
    outer_nd = link_tag - spr_id
    
    if outer_nd%10 == 8:
        i_nd = outer_nd // 10
        j_nd = outer_nd
    else:
        i_nd = outer_nd
        j_nd = outer_nd // 10
    ops.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                col_transf_tag)
    
# make link for the column/beam to gusset plate connection
brace_top_rigid_links = [link for link in brace_top_links
                         if link%10 < 3]

for link_tag in brace_top_rigid_links:
    outer_nd = link_tag - brace_spr_id
    i_nd = outer_nd
    j_nd = outer_nd//10
    
    if i_nd%10==1:
        brace_transf_tag = brace_transf_tag_R
    elif i_nd%10==2:
        brace_transf_tag = brace_transf_tag_L
    ops.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                brace_transf_tag)
    
brace_bot_rigid_links = [link for link in brace_bot_links
                         if link%2 == 1]

for link_tag in brace_bot_rigid_links:
    outer_nd = link_tag - brace_spr_id
    i_nd = outer_nd//100
    j_nd = outer_nd
    
    if outer_nd%10==1:
        brace_transf_tag = brace_transf_tag_R
    elif outer_nd%10==3:
        brace_transf_tag = brace_transf_tag_L
    
    ops.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                brace_transf_tag)

# make link for beam around where shear tabs are
beam_brace_rigid_joints = [nd+spr_id for nd in beam_brace_bay_node]

for link_tag in beam_brace_rigid_joints:
    outer_nd = link_tag - spr_id
    
    if outer_nd%10 == 9:
        i_nd = outer_nd // 10
        j_nd = outer_nd
    else:
        i_nd = outer_nd
        j_nd = outer_nd // 10
        
    ops.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                brace_beam_transf_tag)

# make shear tab pin connections
# use constraint (fix translation, allow rotation) rather than spring
shear_tab_pins = bldg.node_tags['brace_beam_tab']

for nd in shear_tab_pins:
    if nd%10 == 0:
        parent_nd = (nd//10)*10 + 9
    else:
        parent_nd = (nd//10)*10 + 7
    ops.equalDOF(parent_nd, nd, 1, 2, 3, 4, 6)
    
ghost_beams = [beam_tag//10 for beam_tag in brace_beam_elems
               if (beam_tag%brace_beam_id in brace_top_nodes)]

# place ghost trusses along braced frame beams to ensure horizontal movement
beam_id = 200
for elem_tag in ghost_beams:
    i_nd = elem_tag - beam_id
    j_nd = i_nd + 1
    ops.element('Truss', elem_tag, i_nd, j_nd, A_rigid, elastic_mat_tag)
    
print('Elements placed.')
open('./output/model.out', 'w').close()
ops.printModel('-file', './output/model.out')

#%%
############################################################################
#              Loading and analysis
############################################################################

# provide damping
modes=[1]
zeta = [0.05]
nEigenJ = 2;                    # how many modes to analyze
lambdaN  = ops.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
lambda1 = lambdaN[modes[0]-1];           # eigenvalue mode i = 1
wi = lambda1**(0.5)    # w1 (1st mode circular frequency)
Tfb = 2*3.1415/wi      # 1st mode period of the structure

print("Tfb = %.3f s" % Tfb)

betaK       = 0.0
betaKInit   = 0.0
a1 = 2*zeta[0]/wi     

all_elems = ops.getEleTags()
ops.region(80, '-ele',
    *all_elems,
    '-rayleigh', 0.0, betaK, betaKInit, a1)
print('Structure damped with %0.1f%% at frequency %0.2f Hz' % 
      (zeta[0]*100, wi))

monotonic_pattern_tag  = 2
monotonic_series_tag = 1

grav_pattern_tag = 3
grav_series_tag = 4

# ------------------------------
# Loading: gravity
# ------------------------------

# create TimeSeries
ops.timeSeries("Linear", grav_series_tag)

# create plain load pattern
ops.pattern('Plain', grav_pattern_tag, grav_series_tag)

w_applied = 2.087/12
ops.eleLoad('-ele', *brace_beam_elems, '-type', '-beamUniform', 
            -w_applied, 0.0)

nStepGravity = 10  # apply gravity in 10 steps
tol = 1e-5
dGravity = 1/nStepGravity

ops.system("BandGeneral")
ops.test("NormDispIncr", tol, 15)
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("LoadControl", dGravity)
ops.algorithm("Newton")
ops.analysis("Static")
ops.analyze(nStepGravity)

print("Gravity analysis complete!")
ops.loadConst('-time', 0.0)

steps = 500


ops.recorder('Element','-ele',92016,'-file','output/fiber.out',
             'section','fiber', 0.0, -d_brace/2, 'stressStrain')
ops.recorder('Node','-node', 10, 11,'-file', 'output/end_reaction.out', '-dof', 1, 'reaction')
ops.recorder('Node','-node', 2018,'-file', 
             'output/mid_brace_node_l.out', '-dof', 1, 3, 'disp')
ops.recorder('Node','-node', 2017,'-file', 
             'output/mid_brace_node_r.out', '-dof', 1, 3, 'disp')
ops.recorder('Node','-node', 201,'-file', 
             'output/top_brace_node.out', '-dof', 1, 3, 'disp')

#%%

# ------------------------------
# Loading: axial
# ------------------------------
ops.wipeAnalysis()
# create TimeSeries
ops.timeSeries("Linear", monotonic_series_tag)
ops.pattern('Plain', monotonic_pattern_tag, monotonic_series_tag)
ops.load(20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

tol = 1e-5

# ops.system("BandGeneral")   
# ops.test("NormDispIncr", tol, 15)
# ops.numberer("RCM")
# ops.constraints("Plain")
# ops.algorithm("Newton")

ops.test('EnergyIncr', 1.0e-5, 300, 0)
ops.algorithm('KrylovNewton')
ops.system('UmfPack')
ops.numberer("RCM")
ops.constraints("Plain")



ops.analysis("Static")                      # create analysis object
peaks = np.arange(0.1, 10.0, 0.5)
steps = 500
for i, pk in enumerate(peaks):
    du = (-1.0)**i*(peaks[i] / steps)
    ops.integrator('DisplacementControl', 20, 1, du, 1, du, du)
    ops.analyze(steps)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

# ops.analyze(n_steps)
# disp = ops.nodeDisp(201, 1)
# print('Displacement: %.5f' %disp)

# d_mid = ops.nodeDisp(2018)
# print(d_mid)

# # ------------------------------
# Loading: earthquake
# ------------------------------
'''
ops.wipeAnalysis()
# Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
GMDirection = 1  # ground-motion direction
gm_name = 'RSN3905_TOTTORI_OKY002EW'
scale_factor = 30.0
print('Current ground motion: %s at scale %.2f' % (gm_name, scale_factor))

ops.constraints('Plain')
ops.numberer('RCM')
ops.system('BandGeneral')

# Convergence Test: tolerance
tolDynamic          = 1e-5

# Convergence Test: maximum number of iterations that will be performed
maxIterDynamic      = 500

# Convergence Test: flag used to print information on convergence
printFlagDynamic    = 0         

testTypeDynamic     = 'NormDispIncr'
ops.test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)

# algorithmTypeDynamic    = 'Broyden'
# ops.algorithm(algorithmTypeDynamic, 8)
algorithmTypeDynamic    = 'Newton'
ops.algorithm(algorithmTypeDynamic)

# Newmark-integrator gamma parameter (also HHT)
newmarkGamma = 0.5
newmarkBeta = 0.25
ops.integrator('Newmark', newmarkGamma, newmarkBeta)
ops.analysis('Transient')

#  ---------------------------------    perform Dynamic Ground-Motion Analysis
# the following commands are unique to the Uniform Earthquake excitation

gm_dir = '../resource/ground_motions/PEERNGARecords_Unscaled/'

# Uniform EXCITATION: acceleration input
inFile = gm_dir + gm_name + '.AT2'
outFile = gm_dir + gm_name + '.g3'

  # call procedure to convert the ground-motion file
from ReadRecord import ReadRecord
dt, nPts = ReadRecord(inFile, outFile)
g = 386.4
GMfatt = g*scale_factor

eq_series_tag = 100
eq_pattern_tag = 400
# time series information
ops.timeSeries('Path', eq_series_tag, '-dt', dt, 
                '-filePath', outFile, '-factor', GMfatt)     
# create uniform excitation
ops.pattern('UniformExcitation', eq_pattern_tag, 
            GMDirection, '-accel', eq_series_tag)          

# set up ground-motion-analysis parameters
sec = 1.0                      
T_end = 60.0*sec


dt_transient = 0.005
n_steps = int(np.floor(T_end/dt_transient))

# actually perform analysis; returns ok=0 if analysis was successful

import time
t0 = time.time()

ok = ops.analyze(n_steps, dt_transient)   
if ok != 0:
    ok = 0
    curr_time = ops.getTime()
    print("Convergence issues at time: ", curr_time)
    while (curr_time < T_end) and (ok == 0):
        curr_time     = ops.getTime()
        ok          = ops.analyze(1, dt_transient)
        if ok != 0:
            print("Trying Newton with Initial Tangent...")
            ops.algorithm('Newton', '-initial')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)
        if ok != 0:
            print("Trying Newton with line search ...")
            ops.algorithm('NewtonLineSearch')
            ok = ops.analyze(1, dt_transient)
            if ok == 0:
                print("That worked. Back to Newton")
            ops.algorithm(algorithmTypeDynamic)

t_final = ops.getTime()
tp = time.time() - t0
minutes = tp//60
seconds = tp - 60*minutes
print('Ground motion done. End time: %.4f s' % t_final)
print('Analysis time elapsed %dm %ds.' % (minutes, seconds))
'''
ops.wipe()
#%%
############################################################################
#              Plot results
############################################################################

import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

res_columns = ['stress1', 'strain1', 'stress2', 'strain2', 'stress3', 'strain3', 'stress4', 'strain4']

brace_res = pd.read_csv('output/fiber.out', sep=' ', header=None, names=res_columns)

# stress strain
fig = plt.figure()
plt.plot(-brace_res['strain1'], -brace_res['stress1'])
plt.title('Axial stress-strain brace (midpoint, top fiber)')
plt.ylabel('Stress (ksi)')
plt.xlabel('Strain (in/in)')
plt.grid(True)


# disp plot
forces = pd.read_csv('output/end_reaction.out', sep=' ', header=None, 
                     names=['force_left', 'force_right'])
forces['force'] = forces['force_left'] + forces['force_right']

x_coord_L, z_coord_L = mid_brace_coord(2018, L_beam, L_col, offset=ofs)
mid_node_l = pd.read_csv('output/mid_brace_node_l.out', 
                       sep=' ', header=None, names=['x', 'z'])

mid_node_r = pd.read_csv('output/mid_brace_node_r.out', 
                       sep=' ', header=None, names=['x', 'z'])

x_coord_R, z_coord_R = mid_brace_coord(2017, L_beam, L_col, offset=ofs)

# buckling movement (left)
fig = plt.figure()
plt.plot(mid_node_l['x']+x_coord, mid_node_l['z']+z_coord)
plt.title('Movement of mid brace node')
plt.ylabel('z')
plt.xlabel('x')
plt.grid(True)

top_node = pd.read_csv('output/top_brace_node.out', sep=' ', header=None, names=['x', 'z'])

# buckling movement
fig = plt.figure()
plt.plot(top_node['x']+L_beam/2, top_node['z']+L_col)
plt.title('Movement of top node')
plt.ylabel('z')
plt.xlabel('x')
plt.grid(True)

# cycles
fig = plt.figure()
plt.plot(np.arange(1, len(top_node['x'])+1)/steps, top_node['x'])
plt.title('Cyclic history')
plt.ylabel('Displacement history')
plt.xlabel('Cycles')
plt.grid(True)

# force disp
fig = plt.figure()
plt.plot(top_node['x'], -forces['force'])
plt.title('Force-displacement recorded at end node')
plt.ylabel('Force (kip)')
plt.xlabel('Displacement (in)')
plt.grid(True)

#%% animate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

history_len = len(top_node['x'])

dt = 0.005
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-10, 1.1*L_beam), ylim=(-10, 1.2*L_col))
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
time_template = 'time = %.1fs'
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)

def animate(i):
    thisx = [0, mid_node_l['x'][i]+x_coord_L, top_node['x'][i]+L_beam/2,
             mid_node_r['x'][i]+x_coord_R, L_beam]
    thisy = [0, mid_node_l['z'][i]+z_coord_L, top_node['z'][i]+L_col,
             mid_node_r['z'][i]+z_coord_R, 0]
    
    line.set_data(thisx, thisy)
    
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text

ani = animation.FuncAnimation(
    fig, animate, history_len, interval=1, blit=True)
plt.show()

# ani = animation.FuncAnimation(
#     fig, animate, history_len, interval=60/1000, blit=True)
# plt.show()
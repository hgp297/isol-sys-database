############################################################################
#               Testing for a single brace element

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
    
    y_offset = offset/2 * h_story
    y_origin = bot_y_coord + y_offset
    
    mid_x_coord = x_origin + L_eff/2 * cos(gamma)
    mid_y_coord = y_origin + L_eff/2 * sin(gamma)
    
    return(mid_x_coord, mid_y_coord)

############################################################################
# Construct brace
############################################################################

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
h_story = 0.0 * ft

# set modelbuilder
# x = horizontal, y = in-plane, z = vertical
# command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
ops.model('basic', '-ndm', 3, '-ndf', 6)

# model gravity masses corresponding to the frame placed on building edge
import numpy as np

# nominal change
L_beam = L_bay
L_col = h_story

selected_brace = get_shape('HSS6X6X3/16', 'brace')

d_brace = selected_brace.iloc[0]['b']
t_brace = selected_brace.iloc[0]['tdes']

# base node
ops.node(10, 0.0, 0.0, 0.0)
ops.fix(10, 1, 1, 1, 1, 1, 1)

# end node
ops.node(201, L_bay/2, 0.0, 0.0)
ops.fix(201, 0, 1, 1, 1, 0, 1)

# middle node
xc, zc = mid_brace_coord(2018, L_bay, h_story, camber=0.001, offset=0.25)
ops.node(2018, xc, 0.0, zc)

# gusset plate ends and brace ends
xc, zc = bot_gp_coord(1003, L_bay, h_story, offset=0.25)
ops.node(1003, xc, 0.0, zc)
ops.node(1004, xc, 0.0, zc)

xc, zc = top_gp_coord(2012, L_bay, h_story, offset=0.25)
ops.node(2012, xc, 0.0, zc)
ops.node(2016, xc, 0.0, zc)

############################################################################
# Materials 
############################################################################

# General elastic section (non-plastic beam columns, leaning columns)
elastic_mat_tag = 52
torsion_mat_tag = 53
ghost_mat_tag = 54

# Steel material tag
steel_mat_tag = 31
gp_mat_tag = 32

# Section tags
col_sec_tag = 41
brace_beam_sec_tag = 43
brace_sec_tag = 44
gp_sec_tag = 45

# Integration tags
col_int_tag = 61
brace_beam_int_tag = 63
brace_int_tag = 64

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
A_ghost = 0.05
E_ghost = 100.0
ops.uniaxialMaterial('Elastic', ghost_mat_tag, E_ghost)

# define material: Steel02
# command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
Fy  = 50*ksi        # yield strength
b   = 0.1           # hardening ratio
R0 = 15
cR1 = 0.925
cR2 = 0.15
ops.uniaxialMaterial('Elastic', torsion_mat_tag, J)
ops.uniaxialMaterial('Steel02', steel_mat_tag, Fy, Es, b, R0, cR1, cR2)

################################################################################
# define brace
################################################################################

# geometric transformation for beam-columns
# command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
beam_transf_tag   = 1
col_transf_tag    = 2
brace_beam_transf_tag = 3
brace_transf_tag = 4

# brace geometry (we can use one because HSS is symmetric)
xyz_i = ops.nodeCoord(10)
xyz_j = ops.nodeCoord(201)
brace_x_axis_L = np.subtract(xyz_j, xyz_i)
brace_x_axis_L = brace_x_axis_L / np.sqrt(np.sum(brace_x_axis_L**2))
vecxy_brace = [0,1,0] # Use any vector in local x-y, but not local x
vecxz = np.cross(brace_x_axis_L,vecxy_brace) # What OpenSees expects
vecxz_brace = vecxz / np.sqrt(np.sum(vecxz**2))

ops.geomTransf('Corotational', brace_transf_tag, *vecxz_brace) # braces

# Fiber section parameters
nfw = 4     # number of fibers in web
nff = 4     # number of fibers in each flange

###################### Brace #############################

# brace section: HSS section
ops.section('Fiber', brace_sec_tag, '-GJ', Gs*J)
ops.patch('rect', steel_mat_tag, 
    1, nff,  d_brace/2-t_brace, -d_brace/2, d_brace/2, d_brace/2)
ops.patch('rect', steel_mat_tag, 
    1, nff, -d_brace/2, -d_brace/2, -d_brace/2+t_brace, d_brace/2)
ops.patch('rect', steel_mat_tag, nfw, 
    1, -d_brace/2+t_brace, -d_brace/2, d_brace/2-t_brace, -d_brace/2+t_brace)
ops.patch('rect', steel_mat_tag, nfw, 
    1, -d_brace/2+t_brace, d_brace/2-t_brace, d_brace/2-t_brace, d_brace/2)

# use a distributed plasticity integration with 4 IPs
n_IP = 4
ops.beamIntegration('Lobatto', brace_int_tag, 
                    brace_sec_tag, n_IP)

# main brace
ops.element('forceBeamColumn', 91004, 1004, 2018, 
            brace_transf_tag, brace_int_tag)
ops.element('forceBeamColumn', 92016, 2018, 2016, 
            brace_transf_tag, brace_int_tag)

# ghost truss
ops.element('corotTruss', 91009, 1004, 2016, A_ghost, ghost_mat_tag)

###################### Gusset plates #############################

# springs
ops.element('zeroLength', 51004, 1003, 1004,
            '-mat', elastic_mat_tag, gp_mat_tag, 
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_L, *vecxy_brace)
ops.equalDOF(1003, 1004, 1, 2, 3, 6)

ops.element('zeroLength', 52016, 2016, 2012,
            '-mat', elastic_mat_tag, gp_mat_tag,
            '-dir', 4, 5, 
            '-orient', *brace_x_axis_L, *vecxy_brace)
ops.equalDOF(2012, 2016, 1, 2, 3, 6)

# rigid portion
ops.element('elasticBeamColumn', 51003, 10, 1003, 
            A_rigid, Es, Gs, J, I_rigid, I_rigid, 
            brace_transf_tag)
ops.element('elasticBeamColumn', 52012, 2012, 201, 
            A_rigid, Es, Gs, J, I_rigid, I_rigid, 
            brace_transf_tag)


############################################################################
#               Generalized model constructor

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2023

# Description:  Script models designed structure for a generic frame
#               Intake (system, design values), outputs a frame in Ops domain

# Open issues:  (1) 
############################################################################

###############################################################################
#              Node and element numberer for frames
###############################################################################

def numberer(frame_type, n_bays, n_stories):
    # larger than 8 bays is not supported
    assert n_bays < 9
    
    ###### Main node system ######
    # Fixed nodes are 8xx, with xx being sequential from leftmost base
    #   898 and 899 are reserved for wall
    
    # All nodes are numbered xy, with x indicating floor number 
    #   1 at ground, n_stories+1 at roof
    # and y indicating column line
    #   0 at leftmost, n_bays at rightmost
    
    # Leaning column nodes are appended to the right at the same floor (n_bay+1)
    ##############################
    
    # base nodes
    base_id = 800
    base_nodes = [node for node in range(base_id, base_id+n_bays+1)]
    
    # wall nodes
    wall_nodes = [898, 899]
    
    # floor and leaning column nodes
    floor_id = [10*fl for fl in range(1, n_stories+2)]
    nds = [[nd for nd in range (fl, fl+n_bays+1)] for fl in floor_id]
    leaning_nodes = [(fl+n_bays+1) for fl in floor_id]
    
    ###### Spring node system ######
    # Spring support nodes have the coordinates XYA, XY being the parent node
    # A is 6,7,8,9 for S,W,N,E respectively
    ################################
    
    # flatten list to get all nodes
    floor_nodes = [nd for fl in nds for nd in fl]
    
    # make south node if not on bottom floor
    s_spr = [nd*10+6 for nd in floor_nodes
             if ((nd//10)%10 != 1)]
    # make west node if not on the leftmost column and bottom floor
    w_spr = [nd*10+7 for nd in floor_nodes
             if (nd%10) != 0 and ((nd//10)%10 != 1)]
    # make north node if not on top floor
    n_spr = [nd*10+8 for nd in floor_nodes
             if ((nd//10)%10 != n_stories+1)]
    # make east node if not on rightmost column and bottom floor
    e_spr = [nd*10+9 for nd in floor_nodes
             if (nd%10) != n_bays and ((nd//10)%10 != 1)]
    
    # repeat for leaning columns, only N-S
    lc_spr_nodes = [nd*10+6 for nd in leaning_nodes
                    if ((nd//10)%10 != 1)] + [nd*10+8 for nd in leaning_nodes
                                              if ((nd//10)%10 != n_stories+1)]
    
    spring_nodes = s_spr + w_spr + n_spr + e_spr
    
    # column elements, series 100
    col_id = 100
    # make column if not the top floor
    col_elems = [nd+col_id for nd in floor_nodes 
                 if ((nd//10)%10 != n_stories+1)]
    
    # leaning column elements 
    lc_elems = [nd+col_id for nd in leaning_nodes 
                 if ((nd//10)%10 != n_stories+1)]
    
    # beam elements, series 200
    beam_id = 200
    # make beam if not the last bay and not the bottom floor
    beam_elems = [nd+beam_id for nd in floor_nodes 
                 if (nd%10 != n_bays) and ((nd//10)%10 != 1)]
    
    # truss elements, series 300
    truss_id = 300
    # make truss on the last bay for all floors
    truss_elems = [nd+truss_id for nd in floor_nodes 
                   if (nd%10 == n_bays)]
    
    # diaphragm elements, series 400
    diaph_id = 400
    # make diaphragm if not the last bay on the bottom floor
    diaph_elems = [nd+diaph_id for nd in floor_nodes 
                   if ((nd//10)%10 == 1) and (nd%10 != n_bays)]
    
    # isolator elements, series 1000
    isol_id = 1000
    # make isolators above base nodes
    isol_elems = [nd+isol_id for nd in base_nodes]
    
    
    
    # spring elements, series 5000
    spring_id = 5000
    spring_elems = [nd+spring_id for nd in spring_nodes]
    lc_spr_elems = [nd+spring_id for nd in lc_spr_nodes]
    
    # wall elements, series 8000
    wall_id = 8000
    wall_elems = [nd+wall_id for nd in wall_nodes]
    
    # TODO: clean up outputs
    return(base_nodes, wall_nodes, floor_nodes, leaning_nodes, spring_nodes,
           lc_spr_nodes, col_elems, lc_elems, beam_elems,
           truss_elems, diaph_elems, isol_elems, spring_elems, lc_spr_elems, wall_elems)

###############################################################################
#              Steel dimensions and parameters
###############################################################################

# get shape properties
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

###############################################################################
#              Bilinear deteriorating model parameters
###############################################################################

def modified_IK_params(shape, L):
    # reference Lignos & Krawinkler (2011)
    Fy = 50 # ksi
    Es = 29000 # ksi

    Sx = float(shape['Sx'])
    Iz = float(shape['Ix'])
    d = float(shape['d'])
    htw = float(shape['h/tw'])
    bftf = float(shape['bf/2tf'])
    ry = float(shape['ry'])
    c1 = 25.4
    c2 = 6.895

    My = Fy * Sx
    thy = My/(6*Es*Iz/L)
    Ke = My/thy
    # consider using Lb = 0 for beams bc of slab?
    Lb = L
    kappa = 0.4
    thu = 0.055

    if d > 21.0:
        Lam = (536*(htw)**(-1.26)*(bftf)**(-0.525)
            *(Lb/ry)**(-0.130)*(c2*Fy/355)**(-0.291))
        thp = (0.318*(htw)**(-0.550)*(bftf)**(-0.345)
            *(Lb/ry)**(-0.0230)*(L/d)**(0.090)*(c1*d/533)**(-0.330)*
            (c2*Fy/355)**(-0.130))
        thpc = (7.50*(htw)**(-0.610)*(bftf)**(-0.710)
            *(Lb/ry)**(-0.110)*(c1*d/533)**(-0.161)*
            (c2*Fy/355)**(-0.320))
    else:
        Lam = 495*(htw)**(-1.34)*(bftf)**(-0.595)*(c2*Fy/355)**(-0.360)
        thp = (0.0865*(htw)**(-0.365)*(bftf)**(-0.140)
            *(L/d)**(0.340)*(c1*d/533)**(-0.721)*
            (c2*Fy/355)**(-0.230))
        thpc = (5.63*(htw)**(-0.565)*(bftf)**(-0.800)
            *(c1*d/533)**(-0.280)*(c2*Fy/355)**(-0.430))

    # Lam = 1000
    # thp = 0.025
    # thpc = 0.3
    thu = 0.4

    return(Ke, My, Lam, thp, thpc, kappa, thu)

###############################################################################
#              Start model and make nodes
###############################################################################

# TODO: clean up inputs (aggregate into df?)

def build_model(L_bay, h_story, w_floor, p_lc,
                 base_nodes, wall_nodes, floor_nodes,
                 leaning_nodes, spring_nodes, lc_spr_nodes,
                 selected_col, selected_beam, selected_roof):
    
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
    g = 386.4
    
    # set modelbuilder
    # x = horizontal, y = in-plane, z = vertical
    # command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # model gravity masses corresponding to the frame placed on building edge
    # TODO: check if this should be 1.0D + 0.5L
    m_grav_inner = w_floor * L_bay / g
    m_grav_outer = w_floor * L_bay / 2 /g
    m_lc = p_lc / g
    
    # convert lengths from ft to inches
    L_beam = L_bay * ft
    L_col = h_story * ft
    
    # base nodes
    for idx, nd in enumerate(base_nodes):
        ops.node(nd, idx*L_beam, 0.0*ft, -1.0*ft)
        ops.fix(nd, 1, 1, 1, 1, 1, 1)
    
    # wall nodes (should only be two)
    n_bays = len(base_nodes) - 1
    n_floors = len(leaning_nodes) - 1
    ops.node(wall_nodes[0], 0.0*ft, 0.0*ft, 0.0*ft)
    ops.node(wall_nodes[1], n_bays*L_beam, 0.0*ft, 0.0*ft)
    for nd in wall_nodes:
        ops.fix(nd, 1, 1, 1, 1, 1, 1)
    
    # structure nodes
    for nd in floor_nodes:
        
        # get multiplier for location from node number
        bay = nd%10
        fl = (nd//10)%10 - 1
        ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
        
        # assign masses, in direction of motion and stiffness
        # DOF list: X, Y, Z, rotX, rotY, rotZ
        if (bay == n_bays) or (bay == 0):
            m_nd = m_grav_outer[fl]
        else:
            m_nd = m_grav_inner[fl]
        negligible = 1e-15
        ops.mass(nd, m_nd, m_nd, m_nd,
                 negligible, negligible, negligible)
        
        # restrain out of plane motion
        ops.fix(nd, 0, 1, 0, 1, 0, 1)
        
    # leaning column nodes
    for nd in leaning_nodes:
        
        # get multiplier for location from node number
        floor = (nd//10)%10 - 1
        ops.node(nd, (n_bays+1)*L_beam, 0.0*ft, floor*L_col)
        m_nd = m_lc[floor]
        ops.mass(nd, m_nd, m_nd, m_nd,
                 negligible, negligible, negligible)
        
        # bottom is roller, otherwise, restrict OOP motion
        if floor == 0:
            ops.fix(nd, 0, 1, 1, 1, 0, 1)
        else:
            ops.fix(nd, 0, 1, 0, 1, 0, 1)
    
    # spring nodes
    from math import floor
    for nd in spring_nodes:
        parent_nd = floor(nd/10)
        
        # get multiplier for location from node number
        bay = int(parent_nd%10)
        fl = int(parent_nd%100/10) - 1
        ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
        
    print('Nodes placed.')
    
################################################################################
# tags
################################################################################

    # General elastic section (non-plastic beam columns, leaning columns)
    lc_spring_mat_tag = 51
    elastic_mat_tag = 52

    # Steel material tag
    steel_col_tag = 31
    steel_beam_tag = 32
    steel_roof_tag = 33

    # Isolation layer tags
    friction_1_tag = 41
    friction_2_tag = 42
    friction_3_tag = 43
    fps_vert_tag = 44
    fps_rot_tag = 45
    
    # Impact material tags
    impact_mat_tag = 91
    
################################################################################
# define materials
################################################################################

    # define material: steel
    Es  = 29000*ksi     # initial elastic tangent
    nu  = 0.2          # Poisson's ratio
    Gs  = Es/(1 + nu) # Torsional stiffness modulus
    J   = 1e10          # Set large torsional stiffness

    # Frame link (stiff elements)
    A_rigid = 1000.0         # define area of truss section
    I_rigid = 1e6        # moment of inertia for p-delta columns
    ops.uniaxial_material('Elastic', elastic_mat_tag, Es)

################################################################################
# define spring materials
################################################################################

    # TODO: link this to shape database
        
    (Ag_col, Iz_col, Iy_col,
     Zx_col, Sx_col, d_col,
     bf_col, tf_col, tw_col) = get_properties(selected_col)
    (Ag_beam, Iz_beam, Iy_beam,
     Zx_beam, Sx_beam, d_beam,
     bf_beam, tf_beam, tw_beam) = get_properties(selected_beam)
    (Ag_roof, Iz_roof, Iy_roof,
     Zx_roof, Sx_roof, d_roof,
     bf_roof, tf_roof, tw_roof) = get_properties(selected_roof)
    
    # Modified IK steel
    cIK = 1.0
    DIK = 1.0
    (Ke_col, My_col, lam_col,
     thp_col, thpc_col,
     kappa_col, thu_col) = modified_IK_params(selected_col, L_col)
    (Ke_beam, My_beam, lam_beam,
     thp_beam, thpc_beam,
     kappa_beam, thu_beam) = modified_IK_params(selected_beam, L_beam)
    (Ke_roof, My_roof, lam_roof,
     thp_roof, thpc_roof,
     kappa_roof, thu_roof) = modified_IK_params(selected_roof, L_beam)

    # calculate modified section properties to account for spring stiffness being in series with the elastic element stiffness
    # Ibarra, L. F., and Krawinkler, H. (2005). "Global collapse of frame structures under seismic excitations,"
    n = 10 # stiffness multiplier for rotational spring

    Iz_col_mod = Iz_col*(n+1)/n
    Iz_beam_mod = Iz_beam*(n+1)/n
    Iz_roof_mod = Iz_roof*(n+1)/n

    Iy_col_mod = Iy_col*(n+1)/n
    Iy_beam_mod = Iy_beam*(n+1)/n
    Iy_roof_mod = Iy_roof*(n+1)/n

    Ke_col = n*6.0*Es*Iz_col/(0.8*L_col)
    Ke_beam = n*6.0*Es*Iz_beam/(0.8*L_beam)
    Ke_roof = n*6.0*Es*Iz_roof/(0.8*L_beam)

    McMy = 1.05 # ratio of capping moment to yield moment, Mc / My
    a_mem_col = (n+1.0)*(My_col*(McMy-1.0))/(Ke_col*thp_col)
    b_col = a_mem_col/(1.0+n*(1.0-a_mem_col))

    a_mem_beam = (n+1.0)*(My_col*(McMy-1.0))/(Ke_beam*thp_beam)
    b_beam = a_mem_beam/(1.0+n*(1.0-a_mem_beam))

    ops.uniaxial_material('Bilin', steel_col_tag, Ke_col, b_col, b_col,
                          My_col, -My_col, lam_col, 
        0, 0, 0, cIK, cIK, cIK, cIK, thp_col, thp_col, thpc_col, thpc_col, 
        kappa_col, kappa_col, thu_col, thu_col, DIK, DIK)

    ops.uniaxial_material('Bilin', steel_beam_tag, Ke_beam, b_beam, b_beam,
                          My_beam, -My_beam, lam_beam, 
        0, 0, 0, cIK, cIK, cIK, cIK, thp_beam, thp_beam, thpc_beam, thpc_beam, 
        kappa_beam, kappa_beam, thu_beam, thu_beam, DIK, DIK)

    ops.uniaxial_material('Bilin', steel_roof_tag, Ke_roof, b_beam, b_beam,
                          My_roof, -My_roof, lam_roof, 
        0, 0, 0, cIK, cIK, cIK, cIK, thp_roof, thp_roof, thpc_roof, thpc_roof, 
        kappa_roof, kappa_roof, thu_roof, thu_roof, DIK, DIK)

################################################################################
# define springs
################################################################################

    # Create springs at column and beam ends
    # Springs follow Modified Ibarra Krawinkler model
    def rot_spring_bilin(eleID, matID, nodeI, nodeJ, mem_tag):
        # columns
        if mem_tag == 1:
            ops.element('zeroLength', eleID, nodeI, nodeJ,
                '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                elastic_mat_tag, elastic_mat_tag, matID, 
                '-dir', 1, 2, 3, 4, 5, 6,
                '-orient', 0, 0, 1, 1, 0, 0,
                '-doRayleigh', 1)           # Create zero length element (spring), rotations allowed about local z axis
        # beams
        if mem_tag == 2:
            ops.element('zeroLength', eleID, nodeI, nodeJ,
                '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                elastic_mat_tag, elastic_mat_tag, matID, 
                '-dir', 1, 2, 3, 4, 5, 6, 
                '-orient', 1, 0, 0, 0, 0, 1,
                '-doRayleigh', 1)           # Create zero length element (spring), rotations allowed about local z axis
        # ops.equalDOF(nodeI, nodeJ, 1, 2, 3, 4, 6)
        
    # spring elements: #5xxx, xxx is the spring node
    spring_id = 5000
    for elem_tag in spring_elems:
        spr_nd = elem_tag - spring_id
        parent_nd = floor(spr_nd/10)
        
        # if last digit is 6 or 8, assign column transformations
        if (spr_nd%10)%2 == 0:
            mem_tag = 1
        else:
            mem_tag = 2
            
        rot_spring_bilin(elem_tag, steel_col_tag, parent_nd, spr_nd, mem_tag)
  
################################################################################
# define beams
################################################################################

    # geometric transformation for beam-columns
    # command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
    beam_transf_tag   = 1
    col_transf_tag    = 2

    ops.geomTransf('Linear', beam_transf_tag, 0, -1, 0) #beams
    ops.geomTransf('Corotational', col_transf_tag, 0, -1, 0) #columns
    
    # outside of concentrated plasticity zones, use elastic beam columns
    
    # define the columns
    col_id = 100
    for elem_tag in col_elems:
        i_nd = (elem_tag - col_id)*10 + 8
        j_nd = (elem_tag - col_id + 10)*10 + 6
        ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                    Ag_col, Es, Gs, J, Iy_col_mod, Iz_col_mod, col_transf_tag)
    
    # define the beams
    beam_id = 200
    # get the digit corresponding to top floor
    n_fl_id = (max(beam_elems)//10)%10
    for elem_tag in beam_elems:
        i_nd = (elem_tag - beam_id)*10 + 9
        j_nd = (elem_tag - beam_id + 1)*10 + 7
        
        # if beam is on top floor, use roof beam
        if (elem_tag//10)%10 == n_fl_id:
            ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_roof, Es, Gs, J, Iy_roof_mod, Iz_roof_mod, 
                        beam_transf_tag)
        else:
            ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_beam, Es, Gs, J, Iy_beam_mod, Iz_beam_mod,
                        beam_transf_tag)
            
################################################################################
# define leaning column
################################################################################

    # Rotational hinge at leaning column joints via zeroLength elements
    k_lc = 1e-9*kip/inch

    # Create the material (spring)
    ops.uniaxialMaterial('Elastic', lc_spring_mat_tag, k_lc)
    
    # define leaning column
    for elem_tag in lc_elems:
        i_nd = (elem_tag - col_id)*10 + 8
        j_nd = (elem_tag - col_id + 10)*10 + 6
        
        # create elastic members
        ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                    A_rigid, Es, Gs, J, I_rigid, I_rigid, col_transf_tag)
        
    for elem_tag in lc_spr_elems:
        
        # create zero length element (spring), rotations allowed about local Z axis
        ops.element('zeroLength', elem_tag, i_nd, j_nd,
            '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
            elastic_mat_tag, elastic_mat_tag, lc_spring_mat_tag, 
            '-dir', 1, 2, 3, 4, 5, 6, '-orient', 0, 0, 1, 1, 0, 0)         
        
    return()
    
    

if __name__ == '__main__':
    # run an example of a frame
    nb = 3
    ns = 3
    [base_nodes, wall_nodes, floor_nodes, lc_nodes, spring_nodes,
     lc_spr_nodes, col_elems, lc_elems, beam_elems,
     truss_elems, diaph_elems, isol_elems,
     spring_elems, lc_spr_elems, wall_elems] = numberer('TFP', nb, ns)
    print('base nodes:', base_nodes)
    print('floor nodes:', floor_nodes)
    print('moat nodes:', wall_nodes)
    print('leaning column nodes:', lc_nodes)
    
    L_bay = 30.0
    h_stories = 13.0
    
    from load_calc import define_gravity_loads
    W_s, w_fl, P_lc = define_gravity_loads(n_floors=ns,
                                           n_bays=nb,
                                           L_bay=L_bay, h_story=h_stories,
                                           n_frames=2)
    W_s, w_fl, P_lc = define_gravity_loads(D_load=None, L_load=None,
                                           S_s=2.282, S_1 = 1.017,
                                           n_floors=ns, n_bays=nb, 
                                           L_bay=L_bay, h_story=h_stories,
                                           n_frames=2)
    build_model(L_bay, h_stories, w_fl, P_lc,
                 base_nodes, wall_nodes, floor_nodes,
                 lc_nodes, spring_nodes, lc_spr_nodes)
    
############################################################################
#               Generalized model constructor

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2023

# Description:  Script models designed structure for a generic frame
#               Intake (system, design values), outputs a frame in Ops domain

# Open issues:  (1) 
############################################################################

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
             if (int(nd%100/10) != 1)]
    # make west node if not on the leftmost column and bottom floor
    w_spr = [nd*10+7 for nd in floor_nodes
             if (nd%10) != 0 and (int(nd%100/10) != 1)]
    # make north node if not on top floor
    n_spr = [nd*10+8 for nd in floor_nodes
             if (int(nd%100/10) != n_stories+1)]
    # make east node if not on rightmost column and bottom floor
    e_spr = [nd*10+9 for nd in floor_nodes
             if (nd%10) != n_bays and (int(nd%100/10) != 1)]
    
    # repeat for leaning columns, only N-S
    lc_spr = [nd*10+6 for nd in leaning_nodes
             if (int(nd%100/10) != 1)] + [nd*10+8 for nd in leaning_nodes
                                          if (int(nd%100/10) != n_stories+1)]
    
    spring_nodes = s_spr + w_spr + n_spr + e_spr + lc_spr
    
    # column elements, series 100
    col_id = 100
    # make column if not the top floor
    col_elems = [nd+col_id for nd in floor_nodes 
                 if (int(nd%100/10) != n_stories+1)]
    
    # leaning column elements 
    lc_elems = [nd+col_id for nd in leaning_nodes 
                 if (int(nd%100/10) != n_stories+1)]
    
    # beam elements, series 200
    beam_id = 200
    # make beam if not the last bay and not the bottom floor
    beam_elems = [nd+beam_id for nd in floor_nodes 
                 if (int(nd%10) != n_bays) and (int(nd%100/10) != 1)]
    
    # truss elements, series 300
    truss_id = 300
    # make truss on the last bay for all floors
    truss_elems = [nd+truss_id for nd in floor_nodes 
                   if (int(nd%10) == n_bays)]
    
    # diaphragm elements, series 400
    diaph_id = 400
    # make diaphragm if not the last bay on the bottom floor
    diaph_elems = [nd+diaph_id for nd in floor_nodes 
                   if (int(nd%100/10) == 1) and (int(nd%10) != n_bays)]
    
    # isolator elements, series 1000
    isol_id = 1000
    # make isolators above base nodes
    isol_elems = [nd+isol_id for nd in base_nodes]
    
    
    
    # spring elements, series 5000
    spring_id = 5000
    spring_elems = [nd+spring_id for nd in spring_nodes]
    
    # wall elements, series 8000
    wall_id = 8000
    wall_elems = [nd+wall_id for nd in wall_nodes]
    
    return(base_nodes, wall_nodes, floor_nodes, leaning_nodes, spring_nodes,
           col_elems, lc_elems, beam_elems,
           truss_elems, diaph_elems, isol_elems, spring_elems, wall_elems)

###############################################################################
#              Start model and make nodes
###############################################################################

def define_nodes(L_bay, h_story, w_floor, p_lc,
                 base_nodes, wall_nodes, floor_nodes,
                 leaning_nodes, spring_nodes):
    
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
    # CHECK if this should be 1.0D + 0.5L
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
        bay = int(nd%10)
        fl = int(nd%100/10) - 1
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
        floor = int(nd%100/10) - 1
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
        
    return()
    

if __name__ == '__main__':
    # run an example of a frame
    nb = 3
    ns = 3
    [base_nodes, wall_nodes, floor_nodes, lc_nodes, spring_nodes,
     col_elems, lc_elems, beam_elems,
     truss_elems, diaph_elems, isol_elems,
     spring_elems, wall_elems] = numberer('TFP', nb, ns)
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
    define_nodes(L_bay, h_stories, w_fl, P_lc,
                 base_nodes, wall_nodes, floor_nodes,
                 lc_nodes, spring_nodes)
    
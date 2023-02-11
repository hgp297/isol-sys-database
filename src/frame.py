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
    floor_id = [10*floor for floor in range(1, n_stories+2)]
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
    w_spr = [nd*10+9 for nd in floor_nodes
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
    
    return(base_nodes, wall_nodes, floor_nodes, leaning_nodes, spring_nodes)

if __name__ == '__main__':
    # run an example of a frame
    nb = 3
    ns = 3
    base_nodes, wall_nodes, floor_nodes, lc_nodes, spring_nodes = numberer('TFP', nb, ns)
    print('base nodes:', base_nodes)
    print('floor nodes:', floor_nodes)
    print('moat nodes:', wall_nodes)
    print('leaning column nodes:', lc_nodes)
    
    
    
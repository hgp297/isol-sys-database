############################################################################
#               Generalized building object

# Created by:   Huy Pham
#               University of California, Berkeley

# Date created: February 2023

# Description:  Object stores all information about building for Opensees

# Open issues:  (1) 

############################################################################

# class takes a pandas Series (row) and creates a Building object that holds
# design information on the building

class Building:
        
    # import attributes as building characteristics from pd.Series
    def __init__(self, design):
        for key, value in design.items():
            setattr(self, key, value)
        
    # number nodes
    def number_nodes(self):
        
        frame_type = self.superstructure_system
        n_bays = int(self.num_bays)
        n_stories = int(self.num_stories)
        
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
        
        # flatten list to get all nodes
        floor_nodes = [nd for fl in nds for nd in fl]
            
        # for braced frames, additional nodes are needed
        if frame_type == 'CBF':
            n_braced = int(round(n_bays/2.25))
            
            # roughly center braces around middle
            n_start = round(n_bays/2 - n_braced/2)
            # start from first interior bay
            # no ground floor included
            brace_mids = [nd*10+1 for nd in floor_nodes 
                          if ((nd//10)%10 != 1) and
                          (nd%10 >= n_start) and (nd%10 < n_start+n_braced)]
            
            # bottom brace supports, none on top floor
            brace_bottoms = [nd for nd in floor_nodes
                             if ((nd//10)%10 != n_stories+1) and
                             (nd%10 >= n_start) and (nd%10 <= n_start+n_braced)]
            
        
        ###### Spring node system ######
        # Spring support nodes have the coordinates XYA, XY being the parent node
        # A is 6,7,8,9 for S,W,N,E respectively
        ################################
        
        if frame_type == 'CBF':
            br_inner_se = [nd*10+1 for nd in brace_mids]
            br_inner_sw = [nd*10+2 for nd in brace_mids]
            br_mid_east = [nd*10+3 for nd in brace_mids]
            br_mid_west = [nd*10+4 for nd in brace_mids]
            br_outer_se = [nd*10+5 for nd in brace_mids]
            br_outer_sw = [nd*10+6 for nd in brace_mids]
            
            br_mid_spr = (br_inner_se + br_inner_sw + 
                          br_mid_east + br_mid_west + 
                          br_outer_se + br_outer_sw)
            
            br_bot_inner = [nd*100+1 for nd in brace_bottoms]
            br_bot_outer = [nd*100+2 for nd in brace_bottoms]
            
            br_bot_spr = br_bot_inner + br_bot_outer
            
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
        
        # brace springs, springs 50000, actual brace 900
        if frame_type == 'CBF':
            brace_spr_id = 50000
            brace_mid_elems = [brace_spr_id+nd for nd in br_mid_spr]
            brace_bot_elems = [brace_spr_id+nd for nd in br_bot_spr]
            brace_spr_elems = brace_bot_elems + brace_mid_elems
            
            brace_id = 900
            brace_elems = [brace_id + nd for nd in brace_bottoms]
            
            brace_beams_id = 2000
            br_east_elems = [brace_beams_id+nd for nd in brace_mids]
            br_west_elems = [brace_beams_id+(nd//10) for nd in brace_mids]
            brace_beam_elems = br_east_elems + br_west_elems
        
        self.node_tags = {
            'base': base_nodes,
            'wall': wall_nodes,
            'floor': floor_nodes,
            'leaning': leaning_nodes,
            'spring': spring_nodes,
            'lc_spring': lc_spr_nodes
            }
        
        if frame_type == 'CBF':
            self.node_tags['brace_midspan'] = brace_mids
            self.node_tags['brace_corner'] = brace_bottoms
            self.node_tags['brace_mid_spring'] = br_mid_spr
            self.node_tags['brace_corner_spring'] = br_bot_spr
        
        self.elem_tags = {
            'col': col_elems, 
            'leaning': lc_elems, 
            'beam': beam_elems,
            'truss': truss_elems, 
            'diaphragm': diaph_elems, 
            'isolator': isol_elems, 
            'spring': spring_elems, 
            'lc_spring': lc_spr_elems, 
            'wall': wall_elems
            }
        
        
        if frame_type == 'CBF':
            self.elem_tags['brace'] = brace_elems
            self.elem_tags['brace_spring'] = brace_spr_elems
            self.elem_tags['brace_beams'] = brace_beam_elems
            
            
        self.elem_ids = {
            'col': col_id, 
            'leaning': col_id, 
            'beam': beam_id,
            'truss': truss_id, 
            'diaphragm': diaph_id, 
            'isolator': isol_id, 
            'spring': spring_id, 
            'lc_spring': spring_id, 
            'wall': wall_id,
            'base': base_id
            }

        if frame_type == 'CBF':
            self.elem_ids['brace'] = brace_id
            self.elem_ids['brace_spring'] = brace_spr_id
            
      
###############################################################################
#              Start model and make nodes
###############################################################################

    #TODO: distinguish frame types in builder
    # model frame(frame_type), if CBF -> run model_CBF, else run model_MF()
    
    def model_frame(self):
        
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
        
        L_bay = self.L_bay * ft     # ft to in
        h_story = self.h_story * ft
        w_cases = self.all_w_cases
        plc_cases = self.all_Plc_cases
        
        w_floor = w_cases['1.0D+0.5L'] / ft
        p_lc = plc_cases['1.0D+0.5L'] / ft
        # w_floor = self.w_fl / ft    # kip/ft to kip/in
        # p_lc = self.P_lc
        
        # set modelbuilder
        # x = horizontal, y = in-plane, z = vertical
        # command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # model gravity masses corresponding to the frame placed on building edge
        # TODO: check if "base" level should have mass
        import numpy as np
        m_grav_inner = w_floor * L_bay / g
        m_grav_outer = w_floor * L_bay / 2 /g
        m_lc = p_lc / g
        # prepend mass onto the "ground" level
        m_grav_inner = np.insert(m_grav_inner, 0 , m_grav_inner[0])
        m_grav_outer = np.insert(m_grav_outer, 0 , m_grav_outer[0])
        m_lc = np.insert(m_lc, 0 , m_lc[0])
        
        # load for isolators vertical
        p_outer = sum(w_floor)*L_bay/2
        p_inner = sum(w_floor)*L_bay
        
        # nominal change
        L_beam = L_bay
        L_col = h_story
        
        self.number_nodes()
        
        selected_col = get_shape(self.column, 'column')
        selected_beam = get_shape(self.beam, 'beam')
        selected_roof = get_shape(self.roof, 'beam')
        
        # base nodes
        base_nodes = self.node_tags['base']
        for idx, nd in enumerate(base_nodes):
            ops.node(nd, idx*L_beam, 0.0*ft, -1.0*ft)
            ops.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # wall nodes (should only be two)
        n_bays = int(self.num_bays)
        n_floors = int(self.num_stories)
        
        wall_nodes = self.node_tags['wall']
        ops.node(wall_nodes[0], 0.0*ft, 0.0*ft, 0.0*ft)
        ops.node(wall_nodes[1], n_bays*L_beam, 0.0*ft, 0.0*ft)
        for nd in wall_nodes:
            ops.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # structure nodes
        floor_nodes = self.node_tags['floor']
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
        leaning_nodes = self.node_tags['leaning']
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
        spring_nodes = self.node_tags['spring']
        for nd in spring_nodes:
            parent_nd = floor(nd/10)
            
            # get multiplier for location from node number
            bay = int(parent_nd%10)
            fl = int(parent_nd%100/10) - 1
            ops.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            
        lc_spr_nodes = self.node_tags['lc_spring']
        for nd in lc_spr_nodes:
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
        ops.uniaxialMaterial('Elastic', elastic_mat_tag, Es)
    
################################################################################
# define spring materials
################################################################################
            
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
    
        ops.uniaxialMaterial('Bilin', steel_col_tag, Ke_col, b_col, b_col,
                              My_col, -My_col, lam_col, 
                              0, 0, 0, cIK, cIK, cIK, cIK, 
                              thp_col, thp_col, thpc_col, thpc_col, 
                              kappa_col, kappa_col, thu_col, thu_col, DIK, DIK)
    
        ops.uniaxialMaterial('Bilin', steel_beam_tag, Ke_beam, b_beam, b_beam,
                              My_beam, -My_beam, lam_beam, 
                              0, 0, 0, cIK, cIK, cIK, cIK, 
                              thp_beam, thp_beam, thpc_beam, thpc_beam, 
                              kappa_beam, kappa_beam, thu_beam, thu_beam, DIK, DIK)
    
        ops.uniaxialMaterial('Bilin', steel_roof_tag, Ke_roof, b_beam, b_beam,
                              My_roof, -My_roof, lam_roof, 
                              0, 0, 0, cIK, cIK, cIK, cIK, 
                              thp_roof, thp_roof, thpc_roof, thpc_roof, 
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
        spring_id = self.elem_ids['spring']
        spring_elems = self.elem_tags['spring']
        for elem_tag in spring_elems:
            spr_nd = elem_tag - spring_id
            parent_nd = floor(spr_nd/10)
            
            # if last digit is 6 or 8, assign column transformations
            if (spr_nd%10)%2 == 0:
                mem_tag = 1
                rot_spring_bilin(elem_tag, steel_col_tag, 
                                 parent_nd, spr_nd, mem_tag)
            else:
                mem_tag = 2
                
                # if beam is on top floor, use roof beam springs
                if (parent_nd//10) == n_floors + 1:
                    rot_spring_bilin(elem_tag, steel_roof_tag, 
                                     parent_nd, spr_nd, mem_tag)
                else:
                    rot_spring_bilin(elem_tag, steel_beam_tag, 
                                     parent_nd, spr_nd, mem_tag)
            
################################################################################
# define beams
################################################################################

        # geometric transformation for beam-columns
        # command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
        beam_transf_tag   = 1
        col_transf_tag    = 2
    
        ops.geomTransf('Linear', beam_transf_tag, 0, -1, 0) # beams
        ops.geomTransf('Corotational', col_transf_tag, 0, -1, 0) # columns
        
        # outside of concentrated plasticity zones, use elastic beam columns
        
        # define the columns
        col_id = self.elem_ids['col']
        col_elems = self.elem_tags['col']
        for elem_tag in col_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_col, Es, Gs, J, Iy_col_mod, Iz_col_mod, col_transf_tag)
        
        # define the beams
        beam_id = self.elem_ids['beam']
        beam_elems = self.elem_tags['beam']
        for elem_tag in beam_elems:
            i_nd = (elem_tag - beam_id)*10 + 9
            j_nd = (elem_tag - beam_id + 1)*10 + 7
            
            # if beam is on top floor, use roof beam
            if (elem_tag//10)%10 == n_floors + 1:
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
        lc_elems = self.elem_tags['leaning']
        for elem_tag in lc_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            # create elastic members
            ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, col_transf_tag)
        
        # TODO: check that bottom node above the roller
        lc_spr_elems = self.elem_tags['lc_spring']
        for elem_tag in lc_spr_elems:
            spr_nd = elem_tag - spring_id
            parent_nd = floor(spr_nd/10)
            
            # create zero length element (spring), rotations allowed about local Z axis
            ops.element('zeroLength', elem_tag, parent_nd, spr_nd,
                '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                elastic_mat_tag, elastic_mat_tag, lc_spring_mat_tag, 
                '-dir', 1, 2, 3, 4, 5, 6, '-orient', 0, 0, 1, 1, 0, 0)
            
################################################################################
# Trusses and diaphragms
################################################################################
        truss_id = self.elem_ids['truss']
        truss_elems = self.elem_tags['truss']
        for elem_tag in truss_elems:
            i_nd = elem_tag - truss_id
            j_nd = elem_tag - truss_id + 1
            ops.element('Truss', elem_tag, i_nd, j_nd, A_rigid, elastic_mat_tag)
            
        diaph_id = self.elem_ids['diaphragm']
        diaph_elems = self.elem_tags['diaphragm']
        for elem_tag in diaph_elems:
            i_nd = elem_tag - diaph_id
            j_nd = elem_tag - diaph_id + 1
            ops.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, beam_transf_tag)
            
################################################################################
# Isolators
################################################################################

        if self.isolator_system == 'TFP':
            
            # TFP system
            # TODO: check displacement capacities
            # Isolator parameters
            R1 = self.R_1
            R2 = self.R_2
            uy          = 0.00984*inch      # 0.025cm from Scheller & Constantinou
            dSlider1    = 4 *inch           # slider diameters
            dSlider2    = 11*inch
            d1      = 10*inch   - dSlider1  # displacement capacities
            d2      = 37.5*inch - dSlider2
            h1      = 1*inch                # half-height of sliders
            h2      = 4*inch
            
            # TODO: effective height of bearing
            L1      = R1 - h1
            L2      = R2 - h2
    
            # uLim    = 2*d1 + 2*d2 + L1*d2/L2 - L1*d2/L2
    
            # friction pendulum system
            # kv = EASlider/hSlider
            kv = 6*1000*kip/inch
            ops.uniaxialMaterial('Elastic', fps_vert_tag, kv)
            ops.uniaxialMaterial('Elastic', fps_rot_tag, 10.0)
    
    
            # Define friction model for FP elements
            # command: frictionModel Coulomb tag mu
            ops.frictionModel('Coulomb', friction_1_tag, self.mu_1)
            ops.frictionModel('Coulomb', friction_2_tag, self.mu_2)
    
    
            # define 2-D isolation layer 
            kvt     = 0.01*kv
            minFv   = 1.0
            
            isol_id = self.elem_ids['isolator']
            base_id = self.elem_ids['base']
            isol_elems = self.elem_tags['isolator']
            for elem_tag in isol_elems:
                i_nd = elem_tag - isol_id
                j_nd = elem_tag - isol_id - base_id + 10
                
                # if top node is furthest left or right, vertical force is outer
                if (j_nd == 0) or (j_nd%10 == n_bays):
                    p_vert = p_outer
                else:
                    p_vert = p_inner
                ops.element('TripleFrictionPendulum', elem_tag, i_nd, j_nd,
                            friction_1_tag, friction_2_tag, friction_2_tag,
                            fps_vert_tag, fps_rot_tag, fps_rot_tag, fps_rot_tag,
                            L1, L2, L2, d1, d2, d2,
                            p_vert, uy, kvt, minFv, 1e-5)
        else:
            # LRB modeling
            
            # dimensions. Material parameters should not be edited without 
            # modifying design script
            K_bulk = 290.0*ksi
            G_r = 0.060*ksi
            D_inner = self.d_lead
            D_outer = self.d_bearing
            t_shim = 0.1*inch
            t_rubber_whole = self.t_r
            n_layers = int(self.n_layers)
            t_layer = t_rubber_whole/n_layers
            
            # calculate yield strength. this assumes design was done correctly
            Q_L = self.Q * self.W
            alpha = 1.0/self.k_ratio
            Fy_LRB = Q_L/(1 - alpha)
            
            # define 2-D isolation layer 
            isol_id = self.elem_ids['isolator']
            base_id = self.elem_ids['base']
            isol_elems = self.elem_tags['isolator']
            
            for elem_tag in isol_elems:
                i_nd = elem_tag - isol_id
                j_nd = elem_tag - isol_id - base_id + 10
                
                # if top node is furthest left or right, vertical force is outer
                if (j_nd == 0) or (j_nd%10 == n_bays):
                    p_vert = p_outer
                else:
                    p_vert = p_inner
                    
                # TODO: change temp coefficients to imperial units
                ops.element('LeadRubberX', elem_tag, i_nd, j_nd, Fy_LRB, alpha,
                            G_r, K_bulk, D_inner, D_outer,
                            t_shim, t_layer, n_layers)
  
################################################################################
# Walls
################################################################################

        # define impact moat as ZeroLengthImpact3D elements
        # https://opensees.berkeley.edu/wiki/index.php/Impact_Material
        # assume slab of base layer is 6 in
        # model half of slab
        VSlab = (6*inch)*(45*ft)*(90*ft)
        pi = 3.14159
        RSlab = (3/4/pi*VSlab)**(1/3)
        
        # assume wall extends 9 ft up to isolation layer and is 12 in thick
        VWall = (12*inch)*(1*ft)*(45*ft)
        RWall = (3/4/pi*VWall)**(1/3)
        
        # concrete slab and wall properties
        Ec = 3645*ksi
        nu_c = 0.2
        h_impact = (1-nu_c**2)/(pi*Ec)
        
        # impact stiffness parameter from Muthukumar, 2006
        khWall = 4/(3*pi*(h_impact + h_impact))*((RSlab*RWall)/(RSlab + RWall))**(0.5)
        e           = 0.7                                                   # coeff of restitution (1.0 = perfectly elastic collision)
        delM        = 0.025*inch                                            # maximum penetration during pounding event, from Hughes paper
        kEffWall    = khWall*((delM)**(0.5))                                # effective stiffness
        a           = 0.1                                                   # yield coefficient
        delY        = a*delM                                                # yield displacement
        nImpact     = 3/2                                                   # Hertz power rule exponent
        EImpact     = khWall*delM**(nImpact+1)*(1 - e**2)/(nImpact+1)       # energy dissipated during impact
        K1          = kEffWall + EImpact/(a*delM**2)                        # initial impact stiffness
        K2          = kEffWall - EImpact/((1-a)*delM**2)                    # secondary impact stiffness
    
        # TODO: adjust for torsion?
        moat_gap = self.D_m * self.moat_ampli
    
        ops.uniaxialMaterial('ImpactMaterial', impact_mat_tag, 
                             K1, K2, -delY, -moat_gap)
        
        # command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, 
        #   '-dir', *dirs, <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)
        wall_elems = self.elem_tags['wall']
        
        ops.element('zeroLength', wall_elems[0], wall_nodes[0], 10,
                    '-mat', impact_mat_tag, elastic_mat_tag,
                    '-dir', 1, 3, '-orient', 1, 0, 0, 0, 1, 0)
        ops.element('zeroLength', wall_elems[1], 10+n_bays, wall_nodes[1], 
                    '-mat', impact_mat_tag, elastic_mat_tag,
                    '-dir', 1, 3, '-orient', 1, 0, 0, 0, 1, 0)
        
        print('Elements placed.')
        # ops.printModel('-file', './test.log')
###############################################################################
#              Steel dimensions and parameters
###############################################################################

def get_shape(shape_name, member, csv_dir='../resource/'):
    import pandas as pd
    
    if member == 'beam':
        shape_db = pd.read_csv(csv_dir+'beamShapes.csv',
                               index_col=None, header=0)
    elif member == 'column':
        shape_db = pd.read_csv(csv_dir+'colShapes.csv',
                               index_col=None, header=0)
    shape = shape_db.loc[shape_db['AISC_Manual_Label'] == shape_name]
    return(shape)

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
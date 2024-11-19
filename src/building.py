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
        self._model = None 

        for key, value in design.items():
            setattr(self, key, value)
            
    def floating_nodes(self):
        import opensees.openseespy as ops
        
        connected_nodes = []
        for ele in self._model.getEleTags():
            for nd in self._model.eleNodes(ele):
                connected_nodes.append(nd)
        
        defined_nodes = self._model.getNodeTags()
     
        # Use XOR operator, ^
        return list(set(connected_nodes) ^ set(defined_nodes))
        
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
        base_id = 900
        base_nodes = [node for node in range(base_id, base_id+n_bays+1)]
        
        # wall nodes
        wall_nodes = [898, 899]
        
        # floor and leaning column nodes
        floor_id = [10*fl for fl in range(1, n_stories+2)]
        nds = [[nd for nd in range (fl, fl+n_bays+1)] for fl in floor_id]
        leaning_nodes = [(fl+n_bays+1) for fl in floor_id]
        
        # flatten list to get all nodes
        floor_nodes = [nd for fl in nds for nd in fl]
        
        diaph_nodes = [nd for nd in floor_nodes
                       if nd//10 == 1]
            
        # for braced frames, additional nodes are needed
        if frame_type == 'CBF':
            # number of brace mid nodes
            num_br_nds = 7
            n_braced = int(round(n_bays/2.25))
            n_braced = max(n_braced, 1)
            
            # roughly center braces around middle of the building
            n_start = round(n_bays/2 - n_braced/2)
            n_start = max(n_start, 1)
            
            # start from first interior bay
            # no ground floor included
            # top brace nodes have numbers xy1, where xy is the main node to its left
            t_brace_id = 1
            brace_tops = [nd*10+t_brace_id for nd in floor_nodes 
                          if ((nd//10)%10 != 1) and
                          (nd%10 >= n_start) and (nd%10 < n_start+n_braced)]
            
            # bottom brace supports, none on top floor
            # bottom nodes are just a copy of the existing main nodes where braces connect
            brace_bottoms = [nd for nd in floor_nodes
                             if ((nd//10)%10 != n_stories+1) and
                             (nd%10 >= n_start) and (nd%10 <= n_start+n_braced)]
            
            brace_beam_ends = [nd+10 for nd in brace_bottoms]
            
            # numbering scheme, brace mids are xxab, where
            # xx is the parent node of the top node (i.e. 311 -> 31)
            # a is 2 for left brace, 3 for right brace
            # b is the number of subdivision (0-9)
            
            r_brace_nodes = [(nd+2)*10+el+1 for nd in brace_tops 
                             for el in range(num_br_nds)]
            l_brace_nodes = [(nd+1)*10+el+1 for nd in brace_tops 
                             for el in range(num_br_nds)]
            
            # # create two mid-brace nodes for each top brace nodes
            # # these are extensions of the support springs (below)
            # r_brace_id = 7
            # r_brace_nodes = [nd*10+r_brace_id for nd in brace_tops]
            # l_brace_id = 8
            # l_brace_nodes = [nd*10+l_brace_id for nd in brace_tops]
            
            brace_mids = l_brace_nodes + r_brace_nodes
            
        
        ###### Spring node system ######
        # Spring support nodes have the coordinates XYA, XY being the parent node
        # A is 6,7,8,9 for S,W,N,E respectively
        
        # Support nodes for braced frames have different systems
        # Starting from inner SW and going clockwise, the numbering is xy1a
        # where xy1 is the brace_top node, and a is the position indicator
        #       xy13    xy1     xy14
        #           xy12    xy11
        #       xy16            xy15
        #   (xy2z)                  (xy3z)  <- further down midspan
        ################################
        
        if frame_type == 'CBF':
            br_top_e_inner = [nd*10+1 for nd in brace_tops]
            br_top_w_inner = [nd*10+2 for nd in brace_tops]
            br_top_west = [nd*10+3 for nd in brace_tops]
            br_top_east = [nd*10+4 for nd in brace_tops]
            br_top_e_outer = [nd*10+5 for nd in brace_tops]
            br_top_w_outer = [nd*10+6 for nd in brace_tops]
            
            br_top_spr = (br_top_e_inner + br_top_w_inner + 
                          br_top_e_outer + br_top_w_outer)
            
            br_beam_spr = br_top_west + br_top_east 
            
            br_bot_w_inner = [nd*100+1 for nd in brace_bottoms
                              if (nd%10 != n_start)]
            
            br_bot_w_outer = [nd*100+2 for nd in brace_bottoms
                              if (nd%10 != n_start)]
            
            br_bot_e_inner = [nd*100+3 for nd in brace_bottoms
                              if (nd%10 != n_start+n_braced)]
            br_bot_e_outer = [nd*100+4 for nd in brace_bottoms
                              if (nd%10 != n_start+n_braced)]
            
            br_bot_spr = (br_bot_w_inner + br_bot_w_outer + 
                          br_bot_e_inner + br_bot_e_outer)
            
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
        
        # add an additional support node for shear tabs in CBFs
        if frame_type == 'CBF':
            e_tab_spr = [nd*10 for nd in brace_beam_ends
                         if nd%10 != n_start+n_braced]
            w_tab_spr = [nd*10+5 for nd in brace_beam_ends
                         if nd%10 != n_start]
            tab_spr = e_tab_spr + w_tab_spr
            
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
        
        # for LRB, we need additional bearings represented at the edges
        from math import ceil
        isol_type = self.isolator_system
        if isol_type == 'LRB':
            # check if num bearings have been reduced
            # reduction number is fixed at 8 for now
            if int(self.N_lb) < 4*n_bays:
                print('LRB reduction active.')
                # remove 4 from each of the y-dir edge frames
                n_edge_bearings = (n_bays + 1) - 4
                
                # at minimum, keep one bearing
                # this would be incorrect only for <5-bay structures that need reduced bearings
                total_addl_bearings = max(n_edge_bearings - 2, 0)
                
                left_bearings = ceil(total_addl_bearings/2)
                right_bearings = total_addl_bearings - left_bearings
                addl_isol_elems = ([isol_elems[0]+10*(j+1) 
                                   for j in range(left_bearings)] + 
                                   [isol_elems[-1]+10*(j+1) 
                                    for j in range(right_bearings)])
            else:
                n_edge_bearings = n_bays+1
                total_addl_bearings = n_edge_bearings - 2
                
                left_bearings = ceil(total_addl_bearings/2)
                right_bearings = total_addl_bearings - left_bearings
                addl_isol_elems = ([isol_elems[0]+10*(j+1) 
                                   for j in range(left_bearings)] + 
                                   [isol_elems[-1]+10*(j+1) 
                                    for j in range(right_bearings)])
            
            isol_elems = isol_elems + addl_isol_elems
        
        # spring elements, series 5000
        spring_id = 5000
        spring_elems = [nd+spring_id for nd in spring_nodes]
        lc_spr_elems = [nd+spring_id for nd in lc_spr_nodes]
        
        # wall elements, series 8000
        wall_id = 8000
        wall_elems = [nd+wall_id for nd in wall_nodes]
        
        # brace springs, springs 50000, actual brace 900
        # braces are numbered by 9xxxx, where xxxx is the support node belonging
        # to the top/bottom nodes of the brace
        if frame_type == 'CBF':
            brace_spr_id = 50000
            brace_top_elems = [brace_spr_id+nd for nd in br_top_spr]
            brace_bot_elems = [brace_spr_id+nd for nd in br_bot_spr]
            br_beam_spr_elems = [brace_spr_id+nd for nd in br_beam_spr]
            
            brace_id = 90000
            
            # brace_end_nodes = (br_top_w_outer + br_top_e_outer + 
            #                    br_bot_w_outer + br_bot_e_outer)
            
            brace_bot_end_nodes = (br_bot_w_outer + br_bot_e_outer)
            
            # brace_elems = [brace_id + nd for nd in brace_end_nodes]
            
            # new brace scheme
            last_brace = [nd+1 for nd in brace_mids if nd%10==(num_br_nds)]
            brace_elems = [brace_id + nd for nd in (brace_mids+last_brace)]
            
            # brace_ghost_top = [brace_id + nd + 5 for nd in brace_bot_end_nodes]
            # brace_ghost_bot = [brace_id + nd + 15 for nd in brace_bot_end_nodes]
            # brace_ghost_elems = (brace_ghost_top + brace_ghost_bot)
            
            brace_ghost_elems = [brace_id + nd + 5 for nd in brace_bot_end_nodes]
            
            # brace beams are 2xxx, where xxx is either 0xy for the left parent xy
            # or xy1 for the mid-bay parent xy1
            brace_beams_id = 2000
            br_east_elems = [brace_beams_id+nd for nd in brace_tops]
            br_west_elems = [brace_beams_id+(nd//10) for nd in brace_tops]
            brace_beam_elems = br_east_elems + br_west_elems
        
        self.node_tags = {
            'base': base_nodes,
            'wall': wall_nodes,
            'floor': floor_nodes,
            'leaning': leaning_nodes,
            'spring': spring_nodes,
            'lc_spring': lc_spr_nodes,
            'diaphragm': diaph_nodes
            }
        
        if frame_type == 'CBF':
            self.node_tags['brace_top'] = brace_tops
            self.node_tags['brace_mid'] = brace_mids
            self.node_tags['brace_bottom'] = brace_bottoms
            self.node_tags['brace_beam_spring'] = br_beam_spr
            self.node_tags['brace_top_spring'] = br_top_spr
            self.node_tags['brace_bottom_spring'] = br_bot_spr
            self.node_tags['brace_beam_end'] = brace_beam_ends
            self.node_tags['brace_beam_tab'] = tab_spr
        
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
            self.elem_tags['brace_top_springs'] = brace_top_elems
            self.elem_tags['brace_bot_springs'] = brace_bot_elems
            self.elem_tags['brace_beam_springs'] = br_beam_spr_elems
            self.elem_tags['brace_beams'] = brace_beam_elems
            self.elem_tags['brace_ghosts'] = brace_ghost_elems
            
            
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
            self.elem_ids['brace_beam'] = brace_beams_id
            

    def model_frame(self, convergence_mode=False):
        print('=========== Constructing model ===========')
        print('Frame type:', self.superstructure_system, '|', 
              'Isolator type:', self.isolator_system)
        print('%d bays, %d stories, D_m = %.2f' % 
              (self.num_bays, self.num_stories, self.D_m))
        print('Moat amplification = %.2f | Ry = %.2f' % 
              (self.moat_ampli, self.RI))
        print('Tm = %.2f s | zeta = %.2f | k_ratio = %.2f' %
              (self.T_m, self.zeta_e, self.k_ratio))
        
        if self.superstructure_system == 'MF':
            self.model_moment_frame()
        else:
            if convergence_mode:
                self.model_braced_frame(convergence_mode=True)
            else:
                self.model_braced_frame()
    
###############################################################################
#              MOMENT FRAME OPENSEES MODELING
###############################################################################

    def model_moment_frame(self):

        # import OpenSees and libraries
        import opensees.openseespy as ops
        
        # remove existing model
        if self._model is not None:
            self._model.wipe()
            del self._model
            self._model = None

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
        p_lc = plc_cases['1.0D+0.5L']
        # w_floor = self.w_fl / ft    # kip/ft to kip/in
        # p_lc = self.P_lc
        
        # set modelbuilder
        # x = horizontal, y = in-plane, z = vertical
        # command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
        self._model = ops.Model(ndm=3, ndf=6)
        
        # model gravity masses corresponding to the frame placed on building edge
        import numpy as np
        m_grav_inner = w_floor * L_bay / g
        m_grav_outer = w_floor * L_bay / 2 /g
        m_lc = p_lc / g
        
        # load for isolators vertical
        p_outer = sum(w_floor)*L_bay/2
        p_inner = sum(w_floor)*L_bay
        
        # nominal change
        L_beam = L_bay
        L_col = h_story
        
        self.number_nodes()
        
        # selected_col = get_shape(self.column, 'column')
        # selected_beam = get_shape(self.beam, 'beam')
        # selected_roof = get_shape(self.roof, 'beam')
        
        col_list = self.column
        beam_list = self.beam
        
        # base nodes
        base_nodes = self.node_tags['base']
        for idx, nd in enumerate(base_nodes):
            self._model.node(nd, idx*L_beam, 0.0*ft, -1.0*ft)
            self._model.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # wall nodes (should only be two)
        n_bays = int(self.num_bays)
        
        wall_nodes = self.node_tags['wall']
        self._model.node(wall_nodes[0], 0.0*ft, 0.0*ft, 0.0*ft)
        self._model.node(wall_nodes[1], n_bays*L_beam, 0.0*ft, 0.0*ft)
        for nd in wall_nodes:
            self._model.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # structure nodes
        floor_nodes = self.node_tags['floor']
        for nd in floor_nodes:
            
            # get multiplier for location from node number
            bay = nd%10
            fl = (nd//10)%10 - 1
            self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            
            # assign masses, in direction of motion and stiffness
            # DOF list: X, Y, Z, rotX, rotY, rotZ
            if (bay == n_bays) or (bay == 0):
                m_nd = m_grav_outer[fl]
            else:
                m_nd = m_grav_inner[fl]
            negligible = 1e-15
            self._model.mass(nd, m_nd, m_nd, m_nd,
                     negligible, negligible, negligible)
            
            # restrain out of plane motion
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        # leaning column nodes
        leaning_nodes = self.node_tags['leaning']
        for nd in leaning_nodes:
            
            # get multiplier for location from node number
            floor = (nd//10)%10 - 1
            self._model.node(nd, (n_bays+1)*L_beam, 0.0*ft, floor*L_col)
            m_nd = m_lc[floor]
            self._model.mass(nd, m_nd, m_nd, m_nd,
                     negligible, negligible, negligible)
            
            # bottom is roller, otherwise, restrict OOP motion
            if floor == 0:
                self._model.fix(nd, 0, 1, 1, 1, 0, 1)
            else:
                self._model.fix(nd, 0, 1, 0, 1, 0, 1)
        
        # spring nodes
        spring_nodes = self.node_tags['spring']
        for nd in spring_nodes:
            parent_nd = nd//10
            
            # get multiplier for location from node number
            bay = parent_nd%10
            fl = parent_nd//10 - 1
            self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            
        lc_spr_nodes = self.node_tags['lc_spring']
        for nd in lc_spr_nodes:
            parent_nd = nd//10
            
            # get multiplier for location from node number
            bay = parent_nd%10
            fl = parent_nd//10 - 1
            self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            
        print('Nodes placed.')
################################################################################
# tags
################################################################################

        # General elastic section (non-plastic beam columns, leaning columns)
        lc_spring_mat_tag = 51
        elastic_mat_tag = 52
        torsion_mat_tag = 53
        
        # # Steel material tag
        # steel_col_tag = 31
        # steel_beam_tag = 32
        # steel_roof_tag = 33
    
        # Isolation layer tags
        friction_1_tag = 41
        friction_2_tag = 42
        fps_vert_tag = 44
        fps_rot_tag = 45
        
        # Impact material tags
        impact_mat_tag = 91
        
        # reserve blocks of 10 for integration and section tags
        col_sec = 110
        
        beam_sec = 120
        
        
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
        self._model.uniaxialMaterial('Elastic', elastic_mat_tag, Es)
        self._model.uniaxialMaterial('Elastic', torsion_mat_tag, J)
    
################################################################################
# define spring materials
################################################################################
        def rot_spring_bilin(eleID, matID, nodeI, nodeJ, mem_tag):
            
            # Create springs at column and beam ends
            # Springs follow Modified Ibarra Krawinkler model
            # have spring local z be global y, then allow rotation around local z
            column_x = [0, 0, 1]
            column_y = [1, 0, 0]
            beam_x = [1, 0, 0]
            beam_y = [0, 0, 1]
            
            # columns
            if mem_tag == 1:
                # Create zero length element (spring), rotations allowed about local z axis
                self._model.element('zeroLength', eleID, nodeI, nodeJ,
                    '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                    torsion_mat_tag, elastic_mat_tag, matID, 
                    '-dir', 1, 2, 3, 4, 5, 6,
                    '-orient', *column_x, *column_y,
                    '-doRayleigh', 1)           
            # beams
            if mem_tag == 2:
                # Create zero length element (spring), rotations allowed about local z axis
                self._model.element('zeroLength', eleID, nodeI, nodeJ,
                    '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                    torsion_mat_tag, elastic_mat_tag, matID, 
                    '-dir', 1, 2, 3, 4, 5, 6, 
                    '-orient', *beam_x, *beam_y,
                    '-doRayleigh', 1)
        
        cIK = 1.0
        DIK = 1.0
        n_mik = 10.0
        McMy = 1.11 # ratio of capping moment to yield moment, Mc / My
        
        for fl_col, col in enumerate(col_list):
            
            current_col = get_shape(col, 'column')
            
            # column section
            # match the tag number with the floor's node number
            # for column, this is the bottom node (col between 10 and 20 has tag 111)
            # e.g. first col bot nodes at 3x -> tag 113 and 153
            
            current_col_sec = col_sec + fl_col + 1
            
            # Iz is the stronger axis
            (Ag_col, Iz_col, Iy_col,
             Zx_col, Sx_col, d_col,
             bf_col, tf_col, tw_col) = get_properties(current_col)
            
            # Ibarra formulation places PH at hinge, so all values calculated for full beam
            (Ke_col, My_col, lam_col,
             thp_col, thpc_col,
             kappa_col, thu_col) = modified_IK_params(current_col, L_col)
            

            self._model.uniaxialMaterial('IMKBilin', current_col_sec, Ke_col,
                                  thp_col, thpc_col, thu_col, My_col, McMy, kappa_col,
                                  thp_col, thpc_col, thu_col, My_col, McMy, kappa_col,
                                  lam_col, lam_col, lam_col,
                                  cIK, cIK, cIK,
                                  DIK, DIK, DIK)

            
        for fl_beam, beam in enumerate(beam_list):
            current_beam = get_shape(beam, 'beam')
            
            # beam section: fiber wide flange section
            # match the tag number with the floor's node number
            # e.g. first beams nodes at 2x -> tag 132 and 172
        
            current_beam_sec = beam_sec + fl_beam + 2
            
            # Iz is the stronger axis
            (Ag_beam, Iz_beam, Iy_beam,
             Zx_beam, Sx_beam, d_beam,
             bf_beam, tf_beam, tw_beam) = get_properties(current_beam)
            
            # Modified IK steel
            # Ibarra formulation places PH at hinge, so all values calculated for full beam
            (Ke_beam, My_beam, lam_beam,
             thp_beam, thpc_beam,
             kappa_beam, thu_beam) = modified_IK_params(current_beam, L_beam)
            

            self._model.uniaxialMaterial('IMKBilin', current_beam_sec, Ke_beam,
                                  thp_beam, thpc_beam, thu_beam, My_beam, McMy, kappa_beam,
                                  thp_beam, thpc_beam, thu_beam, My_beam, McMy, kappa_beam,
                                  lam_beam, lam_beam, lam_beam,
                                  cIK, cIK, cIK,
                                  DIK, DIK, DIK)
        
################################################################################
# define springs
################################################################################

        # spring elements: #5xxx, xxx is the spring node
        spring_id = self.elem_ids['spring']
        spring_elems = self.elem_tags['spring']
        for elem_tag in spring_elems:
            spr_nd = elem_tag - spring_id
            parent_nd = spr_nd//10
            
            # if last digit is 6 or 8, assign column transformations
            if (spr_nd%10)%2 == 0:
                mem_tag = 1
                
                # get xXxx digit (floor number), adjust to correctly identify
                # floor's column
                if (spr_nd%10) == 8:
                    fl_num = (spr_nd//100)%10
                elif (spr_nd%10) == 6:
                    fl_num = (spr_nd//100)%10 - 1
                
                # add to match previously defined column tags
                steel_tag = col_sec + fl_num
                
                # make spring with the appropriate material/section
                rot_spring_bilin(elem_tag, steel_tag, 
                                 parent_nd, spr_nd, mem_tag)
            else:
                mem_tag = 2
                
                # get xXxx digit (floor number)
                fl_num = (spr_nd//100)%10
                
                # beam tags end in 2
                steel_tag = beam_sec + fl_num
                
                # make spring with the appropriate material/section
                rot_spring_bilin(elem_tag, steel_tag, 
                                 parent_nd, spr_nd, mem_tag)
                
###############################################################################
# define beams and columns
###############################################################################
        
        # geometric transformation for beam-columns
        # command: geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ) for 3d
        beam_transf_tag   = 1
        col_transf_tag    = 2
        
        # beam geometry
        xyz_i = self._model.nodeCoord(10)
        xyz_j = self._model.nodeCoord(11)
        beam_x_axis = np.subtract(xyz_j, xyz_i)
        vecxy_beam = [0, 0, 1] # Use any vector in local x-y, but not local x
        vecxz = np.cross(beam_x_axis,vecxy_beam) # What OpenSees expects
        vecxz_beam = vecxz / np.sqrt(np.sum(vecxz**2))
        
        # column geometry
        xyz_i = self._model.nodeCoord(10)
        xyz_j = self._model.nodeCoord(20)
        col_x_axis = np.subtract(xyz_j, xyz_i)
        vecxy_col = [1, 0, 0] # Use any vector in local x-y, but not local x
        vecxz = np.cross(col_x_axis,vecxy_col) # What OpenSees expects
        vecxz_col = vecxz / np.sqrt(np.sum(vecxz**2))
    
        self._model.geomTransf('Linear', beam_transf_tag, *vecxz_beam) # beams
        self._model.geomTransf('Corotational', col_transf_tag, *vecxz_col) # columns
        
        # outside of concentrated plasticity zones, use elastic beam columns
        # define the columns
        col_id = self.elem_ids['col']
        col_elems = self.elem_tags['col']
        for elem_tag in col_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            # determine which floor's column to use
            cur_floor_idx = (elem_tag//10)%10 - 1
            col_name = col_list[cur_floor_idx]
            current_col = get_shape(col_name, 'column')
            
            # Iz is the stronger axis
            (Ag_col, Iz_col, Iy_col,
             Zx_col, Sx_col, d_col,
             bf_col, tf_col, tw_col) = get_properties(current_col)
            
            # calculate modified section properties to account for spring stiffness 
            # being in series with the elastic element stiffness
            # Ibarra, L. F., and Krawinkler, H. (2005). 
            # "Global collapse of frame structures under seismic excitations,"
            
            Iz_col_mod = Iz_col*(n_mik+1)/n_mik
            Iy_col_mod = Iy_col*(n_mik+1)/n_mik
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_col, Es, Gs, J, Iy_col_mod, Iz_col_mod, col_transf_tag)
        
        # define the beams
        beam_id = self.elem_ids['beam']
        beam_elems = self.elem_tags['beam']
        for elem_tag in beam_elems:
            i_nd = (elem_tag - beam_id)*10 + 9
            j_nd = (elem_tag - beam_id + 1)*10 + 7
            
            # determine which floor's column to use
            cur_floor_idx = (elem_tag//10)%10 - 2
            beam_name = beam_list[cur_floor_idx]
            current_beam = get_shape(beam_name, 'beam')
            
            # Iz is the stronger axis
            (Ag_beam, Iz_beam, Iy_beam,
             Zx_beam, Sx_beam, d_beam,
             bf_beam, tf_beam, tw_beam) = get_properties(current_beam)
            
            # calculate modified section properties to account for spring stiffness 
            # being in series with the elastic element stiffness
            # Ibarra, L. F., and Krawinkler, H. (2005). 
            # "Global collapse of frame structures under seismic excitations,"
            
            Iz_beam_mod = Iz_beam*(n_mik+1)/n_mik
            Iy_beam_mod = Iy_beam*(n_mik+1)/n_mik
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_beam, Es, Gs, J, Iy_beam_mod, Iz_beam_mod,
                        beam_transf_tag)
        
################################################################################
# define leaning column
################################################################################

        # Rotational hinge at leaning column joints via zeroLength elements
        k_lc = 1e-9*kip/inch
    
        # Create the material (spring)
        self._model.uniaxialMaterial('Elastic', lc_spring_mat_tag, k_lc)
        
        # define leaning column
        lc_elems = self.elem_tags['leaning']
        for elem_tag in lc_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            # create elastic members
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, col_transf_tag)
        
        lc_spr_elems = self.elem_tags['lc_spring']
        for elem_tag in lc_spr_elems:
            spr_nd = elem_tag - spring_id
            parent_nd = spr_nd//10
            
            # create zero length element (spring), rotations allowed about local Z axis
            self._model.element('zeroLength', elem_tag, parent_nd, spr_nd,
                '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                torsion_mat_tag, elastic_mat_tag, lc_spring_mat_tag, 
                '-dir', 1, 2, 3, 4, 5, 6, '-orient', *col_x_axis, *vecxy_col)
            
################################################################################
# Trusses and diaphragms
################################################################################
        truss_id = self.elem_ids['truss']
        truss_elems = self.elem_tags['truss']
        for elem_tag in truss_elems:
            i_nd = elem_tag - truss_id
            j_nd = elem_tag - truss_id + 1
            self._model.element('Truss', elem_tag, i_nd, j_nd, A_rigid, elastic_mat_tag)
            
        diaph_id = self.elem_ids['diaphragm']
        diaph_elems = self.elem_tags['diaphragm']
        for elem_tag in diaph_elems:
            i_nd = elem_tag - diaph_id
            j_nd = elem_tag - diaph_id + 1
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, beam_transf_tag)
            
################################################################################
# Isolators
################################################################################

        if self.isolator_system == 'TFP':
            
            # TFP system
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
            
            L1      = R1 - h1
            L2      = R2 - h2
    
            # uLim    = 2*d1 + 2*d2 + L1*d2/L2 - L1*d2/L2
    
            # friction pendulum system
            # kv = EASlider/hSlider
            kv = 6*1000*kip/inch
            self._model.uniaxialMaterial('Elastic', fps_vert_tag, kv)
            self._model.uniaxialMaterial('Elastic', fps_rot_tag, 10.0)
    
    
            # Define friction model for FP elements
            # command: frictionModel Coulomb tag mu
            self._model.frictionModel('Coulomb', friction_1_tag, self.mu_1)
            self._model.frictionModel('Coulomb', friction_2_tag, self.mu_2)
    
    
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
                self._model.element('TripleFrictionPendulum', elem_tag, i_nd, j_nd,
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
            t_shim = 0.13*inch
            t_rubber_whole = self.t_r
            n_layers = int(self.n_layers)
            t_layer = t_rubber_whole/n_layers
            
            # calculate yield strength. this assumes design was done correctly
            pi = 3.14159
            f_y_Pb = 1.5 # ksi, shear yield strength
            
            Fy_LRB = f_y_Pb*pi*D_inner**2/4
            
            # Q_L = f_y_Pb*pi*D_inner**2/4
            # # N_lb = 4*self.num_bays
            # # Q_L = self.Q * self.W / N_lb
            
            alpha = 1.0/self.k_ratio
            # Fy_LRB = Q_L/(1 - alpha)
            
            kc = 10.0
            phi_M = 0.5
            ac = 1.0
            
            qL_imp = 0.4046256704 # (lbs/in^3) density of lead 
            cL_imp = 0.03076 # (Btu/lb/degF) specific heat of lead at room temp
            kS_imp = 26.0*12.0 # (Btu/(hr*in*F)) thermal conductivity of steel 
            aS_imp = 0.018166036 # (in^2/s) thermal diffusivity of steel 

            sdr = 0.5
            # guesstimate 60% of the entire bearing volume's mass in lead
            # mb = qL_imp*pi*D_outer**2/4*t_rubber_whole*0.6
            mb = 0.0
            cd = 0.0
            tc = 1.0
            
            tag_1 = 0 # cavitation
            tag_2 = 1 # buckling load variation
            tag_3 = 1 # horiz stiffness variation
            tag_4 = 0 # vertical stiffness variation
            tag_5 = 0 # heat
            
            addl_params = [0, 0, 1, 1, 0, 0,
                           kc, phi_M, ac, sdr, mb, cd, tc,
                           qL_imp, cL_imp, kS_imp, aS_imp,
                           tag_1, tag_2, tag_3, tag_4, tag_5]
            
            # define 2-D isolation layer 
            isol_id = self.elem_ids['isolator']
            base_id = self.elem_ids['base']
            isol_elems = self.elem_tags['isolator']
            
            for elem_idx, elem_tag in enumerate(isol_elems):
                
                # if it is an unstacked bearing, it will have xx0x
                if (elem_tag//10)%10 == 0:
                    i_nd = elem_tag - isol_id
                    j_nd = elem_tag - isol_id - base_id + 10
                else:
                    bay_pos = elem_tag % 10
                    i_nd = base_id + bay_pos
                    j_nd = i_nd - base_id + 10
                    
                self._model.element('LeadRubberX', elem_tag, i_nd, j_nd, Fy_LRB, alpha,
                            G_r, K_bulk, D_inner, D_outer,
                            t_shim, t_layer, n_layers, *addl_params)
  
################################################################################
# Walls
################################################################################

        # define impact moat as ZeroLengthImpact3D elements
        # https://opensees.berkeley.edu/wiki/index.php/Impact_Material
        # assume slab of base layer is 6 in
        # model half of slab
        L_bldg = self.L_bldg
        VSlab = (6*inch)*(L_bldg/2*ft)*(L_bldg*ft)
        pi = 3.14159
        RSlab = (3/4/pi*VSlab)**(1/3)
        
        # assume wall extends 1 ft up to isolation layer and is 12 in thick
        VWall = (12*inch)*(1*ft)*(L_bldg/2*ft)
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
    
        moat_gap = self.D_m * self.moat_ampli
    
        self._model.uniaxialMaterial('ImpactMaterial', impact_mat_tag, 
                             K1, K2, -delY, -moat_gap)
        
        # command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, 
        #   '-dir', *dirs, <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)
        wall_elems = self.elem_tags['wall']
        
        self._model.element('zeroLength', wall_elems[0], wall_nodes[0], 10,
                    '-mat', impact_mat_tag,
                    '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
        self._model.element('zeroLength', wall_elems[1], 10+n_bays, wall_nodes[1], 
                    '-mat', impact_mat_tag,
                    '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
        
        float_nodes = self.floating_nodes()
        
        # # assert that all nodes are connected (float_nodes is empty)
        # assert (not float_nodes == True), 'Some nodes are not connected.'
        
        print('Elements placed.')

###############################################################################
#              BRACED FRAME OPENSEES MODELING
###############################################################################

    def model_braced_frame(self, convergence_mode=False):
        # import OpenSees and libraries
        import opensees.openseespy as ops
        
        # remove existing model
        if self._model is not None:
            self._model.wipe()
            del self._model 
            self._model = None

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
        p_lc = plc_cases['1.0D+0.5L']
        # w_floor = self.w_fl / ft    # kip/ft to kip/in
        # p_lc = self.P_lc
        
        # set modelbuilder
        # x = horizontal, y = in-plane, z = vertical
        # command: model('basic', '-ndm', ndm, '-ndf', ndf=ndm*(ndm+1)/2)
        self._model = ops.Model(ndm=3, ndf=6)
        
        self.number_nodes()
        
        # model gravity masses corresponding to the frame placed on building edge
        import numpy as np
        m_grav_inner = w_floor * L_bay / g
        m_grav_outer = w_floor * L_bay / 2 /g
        m_grav_brace = w_floor * L_bay / 2 /g
        m_grav_br_col = w_floor * L_bay * 3/4 /g
        m_lc = p_lc / g
        
        # load for isolators vertical
        p_outer = sum(w_floor)*L_bay/2
        p_inner = sum(w_floor)*L_bay
        
        # nominal change
        L_beam = L_bay
        L_col = h_story
        
        self.number_nodes()
        
        col_list = self.column
        sample_column = get_shape(col_list[0], 'column')
        
        beam_list = self.beam
        sample_beam = get_shape(beam_list[0], 'beam')
        (Ag_beam, Iz_beam, Iy_beam,
         Zx_beam, Sx_beam, d_beam,
         bf_beam, tf_beam, tw_beam) = get_properties(sample_beam)
        
        
        (Ag_col, Iz_col, Iy_col,
         Zx_col, Sx_col, d_col,
         bf_col, tf_col, tw_col) = get_properties(sample_column)
        
        # base nodes
        base_nodes = self.node_tags['base']
        for idx, nd in enumerate(base_nodes):
            self._model.node(nd, idx*L_beam, 0.0*ft, -1.0*ft)
            self._model.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # wall nodes (should only be two)
        n_bays = int(self.num_bays)
        n_braced = int(round(n_bays/2.25))
        # roughly center braces around middle
        n_start = round(n_bays/2 - n_braced/2)
        
        wall_nodes = self.node_tags['wall']
        self._model.node(wall_nodes[0], 0.0*ft, 0.0*ft, 0.0*ft)
        self._model.node(wall_nodes[1], n_bays*L_beam, 0.0*ft, 0.0*ft)
        for nd in wall_nodes:
            self._model.fix(nd, 1, 1, 1, 1, 1, 1)
        
        # structure nodes
        floor_nodes = self.node_tags['floor']
        brace_bot_nodes = self.node_tags['brace_bottom']
        brace_top_nodes = self.node_tags['brace_top']
        
        # assigned no mass to oop directions
        for nd in floor_nodes:
            
            # get multiplier for location from node number
            bay = nd%10
            fl = (nd//10)%10 - 1
            self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            
            # assign masses, in direction of motion and stiffness
            # DOF list: X, Y, Z, rotX, rotY, rotZ
            if (bay == n_bays) or (bay == 0):
                m_nd = m_grav_outer[fl]
            elif (bay == n_start) or (bay == n_start+n_braced):
                m_nd = m_grav_br_col[fl]
            # if column is within braces, but not edge of braced bays
            # it would take wL/4 from each side
            elif (bay > n_start) and (bay < n_start+n_braced):
                m_nd = m_grav_brace[fl]
            else:
                m_nd = m_grav_inner[fl]
            negligible = 1e-15
            self._model.mass(nd, m_nd, negligible, m_nd,
                     negligible, negligible, negligible)
            
            # restrain out of plane motion
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        # fix out-of-plane translations, we do this for every node
        # no torsion, no twisting, no oop translation
        
        # leaning column nodes
        leaning_nodes = self.node_tags['leaning']
        for nd in leaning_nodes:
            
            # get multiplier for location from node number
            floor = (nd//10)%10 - 1
            self._model.node(nd, (n_bays+1)*L_beam, 0.0*ft, floor*L_col)
            m_nd = m_lc[floor]
            self._model.mass(nd, m_nd, negligible, m_nd,
                     negligible, negligible, negligible)
            
            # bottom is roller, otherwise, restrict OOP motion
            if floor == 0:
                self._model.fix(nd, 0, 1, 1, 1, 0, 1)
            else:
                self._model.fix(nd, 0, 1, 0, 1, 0, 1)
                
        # brace nodes
        ofs = 0.25
        L_diag = ((L_bay/2)**2 + L_col**2)**(0.5)
        # L_eff = (1-ofs) * L_diag
        L_gp = ofs/2 * L_diag
        brace_top_nodes = self.node_tags['brace_top']
        for nd in brace_top_nodes:
            parent_node = nd // 10
            
            # extract their corresponding coordinates from the node numbers
            fl = parent_node//10 - 1
            x_coord = (parent_node%10 + 0.5)*L_beam
            z_coord = fl*L_col
            
            m_nd = m_grav_brace[fl]
            self._model.node(nd, x_coord, 0.0*ft, z_coord)
            self._model.mass(nd, m_nd, negligible, m_nd,
                     negligible, negligible, negligible)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        # mid brace node adjusted to have camber of 0.1% L_eff
        # L_eff is defined as L_diag - offset
        brace_mid_nodes = self.node_tags['brace_mid']
        for nd in brace_mid_nodes:
            
            # new quadratic coordinates
            x_coord, z_coord = quad_brace_coord(nd, L_beam, L_col, offset=ofs)
            
            # # values returned are already in inches
            # x_coord, z_coord = mid_brace_coord(nd, L_beam, L_col, offset=ofs)
            
            self._model.node(nd, x_coord, 0.0*ft, z_coord)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
        
        # spring nodes
        spring_nodes = self.node_tags['spring']
        brace_beam_ends = self.node_tags['brace_beam_end']    
        brace_beam_tab_nodes = self.node_tags['brace_beam_tab']
        
        col_brace_bay_node = [nd for nd in spring_nodes
                              if (nd//10 in brace_beam_ends 
                                  or nd//10 in brace_bot_nodes)
                              and (nd%10 == 6 or nd%10 == 8)]
        
        beam_brace_bay_node = [nd//10*10+9 if nd%10 == 0
                               else nd//10*10+7 for nd in brace_beam_tab_nodes]
        
        grav_spring_nodes = [nd for nd in spring_nodes
                             if (nd not in col_brace_bay_node)
                             and (nd not in beam_brace_bay_node)]
        
        grav_beam_spring_nodes = [nd for nd in grav_spring_nodes
                                  if nd%2 == 1]
        
        grav_col_spring_nodes = [nd for nd in grav_spring_nodes
                                  if nd%2 == 0]
        
        for nd in spring_nodes:
            parent_nd = nd//10
            
            # get multiplier for location from node number
            bay = parent_nd%10
            fl = parent_nd//10 - 1
            
            # "springs" inside the brace frames should be treated differently
            # if it's a column spring, the offset should be dbeam/2 if it's below the column node
            # if it's above the column node, there is a GP node attached to it
            # roughly, we put it 1.2x L_gp, where L_gp is the diagonal offset of the gusset plate
            
            # to accommodate MIK hinges, we no longer have offset
            
            # if nd in col_brace_bay_node:
            #     if nd%10 == 6:
            #         y_offset = d_beam/2
            #         self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col-y_offset)
            #     else:
            #         y_offset = 1.2*L_gp
            #         self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col+y_offset)
                    
            # if it's a beam spring, place it +/- d_col to the right/left of the column node
            if nd in beam_brace_bay_node:
                x_offset = d_col/2
                if nd%10 == 7:
                    self._model.node(nd, bay*L_beam-x_offset, 0.0*ft, fl*L_col) 
                else:
                    self._model.node(nd, bay*L_beam+x_offset, 0.0*ft, fl*L_col)
                    
            # otherwise, it is a gravity frame node and can just overlap the main node
            else:
                self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
                
            
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        lc_spr_nodes = self.node_tags['lc_spring']
        for nd in lc_spr_nodes:
            parent_nd = nd//10
            
            # get multiplier for location from node number
            bay = parent_nd%10
            fl = parent_nd//10 - 1
            self._model.node(nd, bay*L_beam, 0.0*ft, fl*L_col)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        brace_beam_spr_nodes = self.node_tags['brace_beam_spring']
        for nd in brace_beam_spr_nodes:
            grandparent_nd = nd//100
            
            # extract their corresponding coordinates from the node numbers
            # no more offset for MIK hinges
            x_offset = 0.0
            # x_offset = 1.2*L_gp
            fl = grandparent_nd//10 - 1
            x_coord = (grandparent_nd%10 + 0.5)*L_beam
            z_coord = fl*L_col
            
            # place the node with the offset l/r of midpoint according to suffix
            if nd%10 == 3:
                self._model.node(nd, x_coord-x_offset, 0.0*ft, z_coord)
            else:
                self._model.node(nd, x_coord+x_offset, 0.0*ft, z_coord)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        for nd in brace_beam_tab_nodes:
            parent_nd = nd//10
            
            # get multiplier for location from node number
            bay = parent_nd%10
            fl = parent_nd//10 - 1
            
            x_offset = d_col/2
            if nd%10 == 5:
                self._model.node(nd, bay*L_beam-x_offset, 0.0*ft, fl*L_col) 
            else:
                self._model.node(nd, bay*L_beam+x_offset, 0.0*ft, fl*L_col)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
            
        # each end has offset/2*L_diagonal assigned to gusset plate offset
        brace_bot_gp_nodes = self.node_tags['brace_bottom_spring']
        
        for nd in brace_bot_gp_nodes:
            
            # values returned are already in inches
            x_coord, z_coord = bot_gp_coord(nd, L_beam, L_col, offset=ofs)
            self._model.node(nd, x_coord, 0.0*ft, z_coord)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
        brace_top_gp_nodes = self.node_tags['brace_top_spring']
        
        for nd in brace_top_gp_nodes:
            # values returned are already in inches
            x_coord, z_coord = top_gp_coord(nd, L_beam, L_col, offset=ofs)
            self._model.node(nd, x_coord, 0.0*ft, z_coord)
            self._model.fix(nd, 0, 1, 0, 1, 0, 1)
        print('Nodes placed.')
        
################################################################################
# tags
################################################################################

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
        # col_int = 150
        
        beam_sec = 120
        # beam_int = 160
        
        br_sec = 130
        br_int = 170
        
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
        self._model.uniaxialMaterial('Elastic', elastic_mat_tag, Es)
        
        # minimal stiffness elements (ghosts)
        if convergence_mode==True:
            A_ghost = 1.0
        else:
            A_ghost = 0.05
        
        # A_ghost = 1.0
        E_ghost = 100.0
        self._model.uniaxialMaterial('Elastic', ghost_mat_tag, E_ghost)
        
        # define material: Steel02
        # command: uniaxialMaterial('Steel01', matTag, Fy, E0, b, a1, a2, a3, a4)
        Fy  = 50*ksi        # yield strength
        b   = 0.001           # hardening ratio
        R0 = 22
        cR1 = 0.925
        cR2 = 0.25
        self._model.uniaxialMaterial('Elastic', torsion_mat_tag, J)
        
        # self._model.uniaxialMaterial('Steel02', steel_mat_tag, Fy, Es, b, R0, cR1, cR2)
        
        if convergence_mode==True:
            self._model.uniaxialMaterial('Steel02', steel_mat_tag, Fy, Es, b, R0, cR1, cR2)
        else:
            self._model.uniaxialMaterial('Steel02', steel_no_fatigue, Fy, Es, b, 
                                  R0, cR1, cR2)
            self._model.uniaxialMaterial('Fatigue', steel_mat_tag, steel_no_fatigue,
                                 '-E0', 0.07, '-m', -0.3, '-min', -1e7, '-max', 1e7)
        
        # GP section: thin plate
        W_w = (L_gp**2 + L_gp**2)**0.5
        L_avg = 0.75* L_gp
        t_gp = 1.375*inch
        Fy_gp = 36*ksi
        
        My_GP = (W_w*t_gp**2/6)*Fy_gp
        K_rot_GP = Es/L_avg * (W_w*t_gp**3/12)
        b_GP = 0.01
        self._model.uniaxialMaterial('Steel02', gp_mat_tag, My_GP, K_rot_GP, b_GP, 
                             R0, cR1, cR2)

################################################################################
# geometric transformations
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
        xyz_i = self._model.nodeCoord(10)
        xyz_j = self._model.nodeCoord(11)
        beam_x_axis = np.subtract(xyz_j, xyz_i)
        vecxy_beam = [0, 0, 1] # Use any vector in local x-y, but not local x
        vecxz = np.cross(beam_x_axis,vecxy_beam) # What OpenSees expects
        vecxz_beam = vecxz / np.sqrt(np.sum(vecxz**2))
        
        
        # column geometry
        xyz_i = self._model.nodeCoord(10)
        xyz_j = self._model.nodeCoord(20)
        col_x_axis = np.subtract(xyz_j, xyz_i)
        vecxy_col = [1, 0, 0] # Use any vector in local x-y, but not local x
        vecxz = np.cross(col_x_axis,vecxy_col) # What OpenSees expects
        vecxz_col = vecxz / np.sqrt(np.sum(vecxz**2))
        
        # brace geometry (left)
        xyz_i = self._model.nodeCoord(brace_top_nodes[0]//10 - 10)
        xyz_j = self._model.nodeCoord(brace_top_nodes[0])
        brace_x_axis_L = np.subtract(xyz_j, xyz_i)
        brace_x_axis_L = brace_x_axis_L / np.sqrt(np.sum(brace_x_axis_L**2))
        vecxy_brace = [0, 1, 0] # Use any vector in local x-y, but not local x
        vecxz = np.cross(brace_x_axis_L,vecxy_brace) # What OpenSees expects
        vecxz_brace_L = vecxz / np.sqrt(np.sum(vecxz**2))
        
        # brace geometry (right)
        xyz_i = self._model.nodeCoord(brace_top_nodes[0]//10 - 10 + 1)
        xyz_j = self._model.nodeCoord(brace_top_nodes[0])
        brace_x_axis_R = np.subtract(xyz_j, xyz_i)
        brace_x_axis_R = brace_x_axis_R / np.sqrt(np.sum(brace_x_axis_R**2))
        vecxy_brace = [0, 1, 0] # Use any vector in local x-y, but not local x
        vecxz = np.cross(brace_x_axis_R,vecxy_brace) # What OpenSees expects
        vecxz_brace_R = vecxz / np.sqrt(np.sum(vecxz**2))
        
        # TODO: transform the beams and columns to Corotational
        self._model.geomTransf('PDelta', brace_beam_transf_tag, *vecxz_beam) # beams
        self._model.geomTransf('PDelta', beam_transf_tag, *vecxz_beam) # beams
        self._model.geomTransf('PDelta', col_transf_tag, *vecxz_col) # columns
        self._model.geomTransf('Corotational', brace_transf_tag_L, *vecxz_brace_L) # braces
        self._model.geomTransf('Corotational', brace_transf_tag_R, *vecxz_brace_R) # braces


################################################################################
# define spring materials
################################################################################

        # should note that the parent-spring convention works only if
        # material is symmetric
        def rot_spring_bilin(eleID, matID, nodeI, nodeJ, mem_tag):
            
            # Create springs at column and beam ends
            # Springs follow Modified Ibarra Krawinkler model
            # have spring local z be global y, then allow rotation around local z
            column_x = [0, 0, 1]
            column_y = [1, 0, 0]
            beam_x = [1, 0, 0]
            beam_y = [0, 0, 1]
            
            # columns
            if mem_tag == 1:
                # Create zero length element (spring), rotations allowed about local z axis
                self._model.element('zeroLength', eleID, nodeI, nodeJ,
                    '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                    torsion_mat_tag, elastic_mat_tag, matID, 
                    '-dir', 1, 2, 3, 4, 5, 6,
                    '-orient', *column_x, *column_y,
                    '-doRayleigh', 1)           
            # beams
            if mem_tag == 2:
                # Create zero length element (spring), rotations allowed about local z axis
                self._model.element('zeroLength', eleID, nodeI, nodeJ,
                    '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                    torsion_mat_tag, elastic_mat_tag, matID, 
                    '-dir', 1, 2, 3, 4, 5, 6, 
                    '-orient', *beam_x, *beam_y,
                    '-doRayleigh', 1)
        
        cIK = 1.0
        DIK = 1.0
        n_mik = 10.0
        McMy = 1.11 # ratio of capping moment to yield moment, Mc / My
        
        for fl_col, col in enumerate(col_list):
            
            current_col = get_shape(col, 'column')
            
            # column section
            # match the tag number with the floor's node number
            # for column, this is the bottom node (col between 10 and 20 has tag 111)
            # e.g. first col bot nodes at 3x -> tag 113 and 153
            
            current_col_sec = col_sec + fl_col + 1
            
            # Iz is the stronger axis
            (Ag_col, Iz_col, Iy_col,
             Zx_col, Sx_col, d_col,
             bf_col, tf_col, tw_col) = get_properties(current_col)
            
            # Ibarra formulation places PH at hinge, so all values calculated for full beam
            (Ke_col, My_col, lam_col,
             thp_col, thpc_col,
             kappa_col, thu_col) = modified_IK_params(current_col, L_col)
            
            
            self._model.uniaxialMaterial('IMKBilin', current_col_sec, Ke_col,
                                  thp_col, thpc_col, thu_col, My_col, McMy, kappa_col,
                                  thp_col, thpc_col, thu_col, My_col, McMy, kappa_col,
                                  lam_col, lam_col, lam_col,
                                  cIK, cIK, cIK,
                                  DIK, DIK, DIK)
            
            
        for fl_beam, beam in enumerate(beam_list):
            current_beam = get_shape(beam, 'beam')
            
            # beam section: fiber wide flange section
            # match the tag number with the floor's node number
            # e.g. first beams nodes at 2x -> tag 132 and 172
        
            current_beam_sec = beam_sec + fl_beam + 2
            
            # Iz is the stronger axis
            (Ag_beam, Iz_beam, Iy_beam,
             Zx_beam, Sx_beam, d_beam,
             bf_beam, tf_beam, tw_beam) = get_properties(current_beam)
            
            # Ibarra formulation places PH at hinge, so all values calculated for full beam
            (Ke_beam, My_beam, lam_beam,
             thp_beam, thpc_beam,
             kappa_beam, thu_beam) = modified_IK_params(current_beam, L_beam)
            

            self._model.uniaxialMaterial('IMKBilin', current_beam_sec, Ke_beam,
                                  thp_beam, thpc_beam, thu_beam, My_beam, McMy, kappa_beam,
                                  thp_beam, thpc_beam, thu_beam, My_beam, McMy, kappa_beam,
                                  lam_beam, lam_beam, lam_beam,
                                  cIK, cIK, cIK,
                                  DIK, DIK, DIK)

################################################################################
# define springs
################################################################################

        # spring elements: #5xxx, xxx is the spring node
        spr_id = self.elem_ids['spring'] 
        spr_elems = self.elem_tags['spring']
        
        brace_spr_id = self.elem_ids['brace_spring']
        brace_top_links = self.elem_tags['brace_top_springs']
        brace_bot_links = self.elem_tags['brace_bot_springs']
        
        # extract where rigid elements are in the entire frame
        # brace_beam_end_joint = [link for link in spr_elems
        #                         if (link-spr_id)//10 in brace_beam_ends
        #                         and (link%10 == 9 or link%10 == 7)]
        
        
        brace_beam_middle_joint = self.elem_tags['brace_beam_springs']
        
        col_joint = [link for link in spr_elems
                     if ((link-spr_id)//10 in brace_beam_ends 
                         or (link-spr_id)//10 in brace_bot_nodes)
                     and (link%10 == 6 or link%10 == 8)]
        
        ###################### columns #############################
        
        # make MIK hinges for column joints
        for elem_tag in col_joint:
            spr_nd = elem_tag - spr_id
            parent_nd = spr_nd//10
            
            # if last digit is 6 or 8, assign column transformations
            # should be all cases
            mem_tag = 1
            
            # get xXxx digit (floor number), adjust to correctly identify
            # floor's column
            if spr_nd%10 == 6:
                fl_num = (spr_nd//100)%10 - 1
            else:
                fl_num = (spr_nd//100)%10
            
            # add to match previously defined column tags
            steel_tag = col_sec + fl_num
            
            # make spring with the appropriate material/section
            rot_spring_bilin(elem_tag, steel_tag, 
                             parent_nd, spr_nd, mem_tag)
                  
                
        # make MIK hinges for brace beams near braces
        for link_tag in brace_beam_middle_joint:
            spr_nd = link_tag - brace_spr_id
            parent_nd = spr_nd//10
            
            mem_tag = 2
            
            # get xXxx digit (floor number)
            fl_num = (spr_nd//1000)%10
            
            # beam tags end in 2
            steel_tag = beam_sec + fl_num
            
            # make spring with the appropriate material/section
            rot_spring_bilin(link_tag, steel_tag, 
                             parent_nd, spr_nd, mem_tag)
            
################################################################################
# define beams and columns - braced bays
################################################################################
        # outside of concentrated plasticity zones, use elastic beam columns
        # define the columns
        col_id = self.elem_ids['col']
        col_elems = self.elem_tags['col']
        
        # find which columns belong to the braced bays
        # (if its i-node parent is a brace_bottom_node)
        col_br_elems = [col for col in col_elems
                        if col-col_id in brace_bot_nodes]
        
        ###################### columns #############################
        
        # elastic beam columns for braced bay columns
        for elem_tag in col_br_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            # determine which floor's column to use
            cur_floor_idx = (elem_tag//10)%10 - 1
            col_name = col_list[cur_floor_idx]
            current_col = get_shape(col_name, 'column')
            
            # Iz is the stronger axis
            (Ag_col, Iz_col, Iy_col,
             Zx_col, Sx_col, d_col,
             bf_col, tf_col, tw_col) = get_properties(current_col)
            
            # calculate modified section properties to account for spring stiffness 
            # being in series with the elastic element stiffness
            # Ibarra, L. F., and Krawinkler, H. (2005). 
            # "Global collapse of frame structures under seismic excitations,"
            
            Iz_col_mod = Iz_col*(n_mik+1)/n_mik
            Iy_col_mod = Iy_col*(n_mik+1)/n_mik
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_col, Es, Gs, J, Iy_col_mod, Iz_col_mod, col_transf_tag)
            
        ###################### beams #############################
            
        # elastic beam columns for braced bay beams 
        brace_beam_id = self.elem_ids['brace_beam']
        brace_beam_elems = self.elem_tags['brace_beams']
        
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
                
            # determine which floor's beam to use
            cur_floor_idx = beam_floor - 2
            beam_name = beam_list[cur_floor_idx]
            current_beam = get_shape(beam_name, 'beam')
            
            # Iz is the stronger axis
            (Ag_beam, Iz_beam, Iy_beam,
             Zx_beam, Sx_beam, d_beam,
             bf_beam, tf_beam, tw_beam) = get_properties(current_beam)
            
            # calculate modified section properties to account for spring stiffness 
            # being in series with the elastic element stiffness
            # Ibarra, L. F., and Krawinkler, H. (2005). 
            # "Global collapse of frame structures under seismic excitations,"
            
            Iz_beam_mod = Iz_beam*(n_mik+1)/n_mik
            Iy_beam_mod = Iy_beam*(n_mik+1)/n_mik
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        Ag_beam, Es, Gs, J, Iy_beam_mod, Iz_beam_mod,
                        beam_transf_tag)
        
        # Fiber section parameters
        nfw = 4     # number of fibers in web
        nff = 4     # number of fibers in each flange
    
        '''
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
            
            self._model.section('Fiber', current_col_sec, '-GJ', Gs*J)
            self._model.patch('rect', steel_mat_tag, 
                1, nff,  d_col/2-tf_col, -bf_col/2, d_col/2, bf_col/2)
            self._model.patch('rect', steel_mat_tag, 
                1, nff, -d_col/2, -bf_col/2, -d_col/2+tf_col, bf_col/2)
            self._model.patch('rect', steel_mat_tag,
                nfw, 1, -d_col/2+tf_col, -tw_col/2, d_col/2-tf_col, tw_col/2)
            
            
            current_col_int = col_int + fl_col + 1
            n_IP = 4
            self._model.beamIntegration('Lobatto', current_col_int, 
                                current_col_sec, n_IP)
        
        # define the columns
        
        # # column section: fiber wide flange section
        # self._model.section('Fiber', col_sec_tag, '-GJ', Gs*J)
        # self._model.patch('rect', steel_mat_tag, 
        #     1, nff,  d_col/2-tf_col, -bf_col/2, d_col/2, bf_col/2)
        # self._model.patch('rect', steel_mat_tag, 
        #     1, nff, -d_col/2, -bf_col/2, -d_col/2+tf_col, bf_col/2)
        # self._model.patch('rect', steel_mat_tag,
        #     nfw, 1, -d_col/2+tf_col, -tw_col, d_col/2-tf_col, tw_col)
        
        # # use a distributed plasticity integration with 4 IPs
        # n_IP = 4
        # self._model.beamIntegration('Lobatto', col_int_tag, col_sec_tag, n_IP)
        
        col_id = self.elem_ids['col']
        col_elems = self.elem_tags['col']
        
        # find which columns belong to the braced bays
        # (if its i-node parent is a brace_bottom_node)
        col_br_elems = [col for col in col_elems
                        if col-col_id in brace_bot_nodes]
        
        grav_cols = [col for col in col_elems
                     if col not in col_br_elems]
        
        for elem_tag in col_br_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            col_floor = i_nd // 100
            
            col_int_tag = col_floor + col_int
            self._model.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
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
            
            self._model.section('Fiber', current_brace_beam_sec, '-GJ', Gs*J)
            self._model.patch('rect', steel_mat_tag, 
                1, nff,  d_beam/2-tf_beam, -bf_beam/2, d_beam/2, bf_beam/2)
            self._model.patch('rect', steel_mat_tag, 
                1, nff, -d_beam/2, -bf_beam/2, -d_beam/2+tf_beam, bf_beam/2)
            self._model.patch('rect', steel_mat_tag,
                nfw, 1, -d_beam/2+tf_beam, -tw_beam/2, d_beam/2-tf_beam, tw_beam/2)
            
            
            current_brace_beam_int = beam_int + fl_beam + 2
            self._model.beamIntegration('Lobatto', current_brace_beam_int, 
                                current_brace_beam_sec, n_IP)
        
        brace_beam_id = self.elem_ids['brace_beam']
        brace_beam_elems = self.elem_tags['brace_beams']
        
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
            self._model.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
                        brace_beam_transf_tag, brace_beam_int_tag)
        '''
        
        ###################### Brace #############################
        brace_list = self.brace
        n_IP = 5
        # starting from bottom floor, define the brace shape for that floor
        # floor 1's brace at 141 and 161, etc.
        for fl_br, brace in enumerate(brace_list):
            current_brace = get_shape(brace, 'brace')
            d_brace = current_brace.iloc[0]['b']
            t_brace = current_brace.iloc[0]['tdes']
            
            # brace section: HSS section
            brace_sec_tag = br_sec + fl_br + 1
            
            self._model.section('Fiber', brace_sec_tag, '-GJ', Gs*J)
            # web
            self._model.patch('rect', steel_mat_tag, nfw, nff,  
                      d_brace/2-t_brace, -d_brace/2, d_brace/2, d_brace/2)
            self._model.patch('rect', steel_mat_tag, nfw, nff, 
                      -d_brace/2, -d_brace/2, -d_brace/2+t_brace, d_brace/2)
            # flange
            self._model.patch('rect', steel_mat_tag, nfw, nff, 
                      -d_brace/2+t_brace, -d_brace/2, d_brace/2-t_brace, -d_brace/2+t_brace)
            self._model.patch('rect', steel_mat_tag, nfw, nff, 
                      -d_brace/2+t_brace, d_brace/2-t_brace, d_brace/2-t_brace, d_brace/2)
            
            brace_int_tag = br_int + fl_br + 1
            self._model.beamIntegration('Lobatto', brace_int_tag, 
                                brace_sec_tag, n_IP)
        
        brace_id = self.elem_ids['brace']
        brace_elems = self.elem_tags['brace']
        num_br_elems = max([el%10 for el in brace_elems])
        
        for elem_tag in brace_elems:
            
            parent_top = (elem_tag - brace_id)//100
            
            # if 9xxXx is 2, brace is left brace
            if (elem_tag//10)%10 == 2:
                brace_transf_tag = brace_transf_tag_L
                parent_bot = parent_top - 10
                bot_mod = 4
                top_mod = 6
            else:
                brace_transf_tag = brace_transf_tag_R
                parent_bot = parent_top - 9
                bot_mod = 2
                top_mod = 5
            
            # check if element is first or last
            if elem_tag%10 == 1:
                i_nd = parent_bot*100 + bot_mod
                j_nd = elem_tag - brace_id
            elif elem_tag%10 == num_br_elems:
                i_nd = elem_tag - brace_id - 1
                j_nd = (parent_top*10 + 1) * 10 + top_mod
            else:
                i_nd = elem_tag - brace_id - 1
                j_nd = elem_tag - brace_id
                
            # ending node is always numbered with parent as floor j_floor
            j_floor = j_nd//1000
            
            current_brace_int = j_floor - 1 + br_int
            
            self._model.element('dispBeamColumn', elem_tag, i_nd, j_nd, 
                        brace_transf_tag, current_brace_int, '-iter', 100, 1e-7)
        
        
        # add ghost trusses to the braces to reduce convergence problems
        brace_ghosts = self.elem_tags['brace_ghosts']
        for elem_tag in brace_ghosts:
            i_nd = (elem_tag - 5) - brace_id
            
            parent_i_nd = i_nd // 100
            if elem_tag%10 == 9:
                j_nd = (parent_i_nd + 10)*100 + 16
            else:
                j_nd = (parent_i_nd + 9)*100 + 15
            self._model.element('corotTruss', elem_tag, i_nd, j_nd, A_ghost, ghost_mat_tag)
            
        ###################### Gusset plates #############################
        
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
                self._model.element('zeroLength', link_tag, i_nd, j_nd,
                    '-mat', torsion_mat_tag, gp_mat_tag, 
                    '-dir', 4, 5, 
                    '-orient', *brace_x_axis_L, *vecxy_brace)
            else:
                self._model.element('zeroLength', link_tag, i_nd, j_nd,
                    '-mat', torsion_mat_tag, gp_mat_tag, 
                    '-dir', 4, 5, 
                    '-orient', *brace_x_axis_R, *vecxy_brace)
                
            # global z-rotation is restrained
            # removed DOF 6 here
            self._model.equalDOF(i_nd, j_nd, 1, 3)
            
        # at top, outer (GP non rigid nodes are 5 and 6)
        brace_top_gp_spring_link = [link for link in brace_top_links
                                    if link%10 > 4]
        
        for link_tag in brace_top_gp_spring_link:
            i_nd = (link_tag - brace_spr_id)
            j_nd = (link_tag - brace_spr_id) - 4
            
            # put the correct local x-axis
            # torsional stiffness around local-x, GP stiffness around local-z
            if link_tag%10 == 6:
                self._model.element('zeroLength', link_tag, i_nd, j_nd,
                    '-mat', torsion_mat_tag, gp_mat_tag, 
                    '-dir', 4, 6, 
                    '-orient', *brace_x_axis_L, *vecxy_brace)
            else:
                self._model.element('zeroLength', link_tag, i_nd, j_nd,
                    '-mat', torsion_mat_tag, gp_mat_tag, 
                    '-dir', 4, 6, 
                    '-orient', *brace_x_axis_R, *vecxy_brace)
                
            # global z-rotation is restrained
            # removed DOF 6 here
            self._model.equalDOF(j_nd, i_nd, 1, 3)
            
################################################################################
# define rigid links in the braced bays
################################################################################  
            
        # make link for the column/beam to gusset plate connection
        brace_top_rigid_links = [link for link in brace_top_links
                                 if link%10 < 3]
        
        goes_ne = [2]
        
        for link_tag in brace_top_rigid_links:
            outer_nd = link_tag - brace_spr_id
            i_nd = outer_nd
            j_nd = outer_nd//10
            
            if (outer_nd%10 in goes_ne):
                brace_transf_tag = brace_transf_tag_L
            else:
                brace_transf_tag = brace_transf_tag_R
                
            self._model.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                        brace_transf_tag)
            
        brace_bot_rigid_links = [link for link in brace_bot_links
                                 if link%2 == 1]
        
        goes_ne = [3]
        for link_tag in brace_bot_rigid_links:
            outer_nd = link_tag - brace_spr_id
            i_nd = outer_nd//100
            j_nd = outer_nd
            
            if (outer_nd%10 in goes_ne):
                brace_transf_tag = brace_transf_tag_L
            else:
                brace_transf_tag = brace_transf_tag_R
                
            self._model.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
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
                
            self._model.element('elasticBeamColumn', link_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                        brace_beam_transf_tag)
        
        # make shear tab pin connections
        # use constraint (fix translation, allow rotation) rather than spring
        shear_tab_pins = self.node_tags['brace_beam_tab']
        
        # for nd in shear_tab_pins:
        #     if nd%10 == 0:
        #         parent_nd = (nd//10)*10 + 9
        #     else:
        #         parent_nd = (nd//10)*10 + 7
        #     self._model.equalDOF(parent_nd, nd, 1, 2, 3, 4, 6)
            
        
        # ModIMKPinching material calibrated to match closely with Elkady's Pinching4
        # assumptions: we roughly use the smallest (top) beam as a reference
        # Many values are derivative of Elkady, A. and D. G. Lignos (2015)'s study
        # Shear tab retains ~15% of the strength (0.121 in Elkady)
        # assumes that this is "bare steel shear tab" connection
        
        Mmax_st = 0.15*My_beam
        # point 1 in Elkady's Pinching4
        thy_st = 0.0045
        My_st = 0.521*Mmax_st
        K0_st = My_st/thy_st 
        # point 3 in Elkady's Pinching4
        thp_st = 0.075 - 0.0045
        Fpp_st = 0.7
        Fpn_st = 0.7

        lam_st = 0.9

        thpc_st = 0.1-0.075
        kappa_res_st = 0.901
        thu_st = 0.08

        self._model.uniaxialMaterial('IMKPinching', 333, K0_st, 
                             thp_st, thpc_st, thu_st, My_st, 1/0.521, kappa_res_st,
                             thp_st, thpc_st, thu_st, My_st, 1/0.521, kappa_res_st,
                             lam_st, lam_st, lam_st, lam_st,
                             1.0, 1.0, 1.0, 1.0,
                             DIK, DIK,
                             Fpp_st, Fpn_st)

        # # Elkady's Pinching4 example
        # # assumes that this is "bare steel shear tab" connection
        
        # My_st = 0.15*My_beam
        # M_1 = 0.521*My_st
        # M_2 = 0.967*My_st
        # M_3 = 1.0*My_st
        # M_4 = 0.901*My_st
        # th1 = 0.0045
        # th2 = 0.0465
        # th3 = 0.0750
        # th4 = 0.10
        # rd_st = 0.57
        # rf_st = 0.40
        # uf_st = 0.05

        # self._model.uniaxialMaterial('Pinching4', 332, M_1, th1, M_2, th2, M_3, th3, M_4, th4,
        #                      -M_1, -th1, -M_2, -th2,- M_3, -th3, -M_4, -th4,
        #                      rd_st, rf_st, uf_st,  rd_st, rf_st, uf_st,   
        #                      0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,   
        #                      0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 'energy')

        # self._model.uniaxialMaterial('MinMax', 333, 332, '-min', -0.08, '-max', 0.08)

        for nd in shear_tab_pins:
            elem_tag = nd + spr_id
            if nd%10 == 0:
                parent_nd = (nd//10)*10 + 9
                i_nd = parent_nd
                j_nd = nd
            else:
                parent_nd = (nd//10)*10 + 7
                i_nd = nd
                j_nd = parent_nd
                
            rot_spring_bilin(elem_tag, 333, i_nd, j_nd, 2)
            
################################################################################
# define gravity frame
################################################################################
        
        # place gravity beams: elastic elements with pinned ends
        beam_elems = self.elem_tags['beam']
        beam_id = self.elem_ids['beam']
        ghost_beams = [beam_tag//10 for beam_tag in brace_beam_elems
                       if (beam_tag%brace_beam_id in brace_top_nodes)]
        grav_beams = [beam_tag for beam_tag in beam_elems
                      if beam_tag not in ghost_beams]
        
        # check equalDOFs here
        for elem_tag in grav_beams:
            i_nd = (elem_tag - beam_id)*10 + 9
            j_nd = (elem_tag - beam_id + 1)*10 + 7
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                        beam_transf_tag)
            
            # self._model.element('forceBeamColumn', elem_tag, i_nd, j_nd, 
            #             beam_transf_tag, grav_beam_int_tag)
        
        # place pin joints for all gravity beams
        for nd in grav_beam_spring_nodes:
            parent_nd = nd // 10
            self._model.equalDOF(parent_nd, nd, 1, 3)
            
        # place ghost trusses along braced frame beams to ensure horizontal movement
        # run this truss to midway
        for elem_tag in ghost_beams:
            i_nd = elem_tag - beam_id
            # j_nd = i_nd + 1
            j_nd = i_nd*10 + 1
            self._model.element('Truss', elem_tag, i_nd, j_nd, A_rigid, elastic_mat_tag)
            
            tag_2 = elem_tag*10
            i_nd = j_nd
            j_nd = (i_nd//10) + 1
            self._model.element('Truss', tag_2, i_nd, j_nd, A_rigid, elastic_mat_tag)
            
       
        # place gravity columns:
        
        grav_cols = [col for col in col_elems
                     if col not in col_br_elems]
        
        for elem_tag in grav_cols:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, 
                        col_transf_tag)
            
        # fully fix all column spring nodes to its parent
        # make pins at the bottom of the columns to ensure no stiffness added
        for nd in grav_col_spring_nodes:
            parent_nd = nd // 10
            if (parent_nd//10 == 1):
                self._model.equalDOF(parent_nd, nd, 1, 3)
            else:
                self._model.equalDOF(parent_nd, nd, 1, 3, 5)
            
################################################################################
# define leaning column
################################################################################

        # Rotational hinge at leaning column joints via zeroLength elements
        k_lc = 1e-9*kip/inch
    
        # Create the material (spring)
        self._model.uniaxialMaterial('Elastic', lc_spring_mat_tag, k_lc)
        
        # define leaning column
        lc_elems = self.elem_tags['leaning']
        for elem_tag in lc_elems:
            i_nd = (elem_tag - col_id)*10 + 8
            j_nd = (elem_tag - col_id + 10)*10 + 6
            
            # create elastic members
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, col_transf_tag)
        
        lc_spr_elems = self.elem_tags['lc_spring']
        for elem_tag in lc_spr_elems:
            spr_nd = elem_tag - spr_id
            parent_nd = spr_nd//10
            
            # create zero length element (spring), rotations allowed about local Z axis
            self._model.element('zeroLength', elem_tag, parent_nd, spr_nd,
                '-mat', elastic_mat_tag, elastic_mat_tag, elastic_mat_tag, 
                torsion_mat_tag, elastic_mat_tag, lc_spring_mat_tag, 
                '-dir', 1, 2, 3, 4, 5, 6, '-orient', *col_x_axis, *vecxy_col)
            
################################################################################
# Trusses and diaphragms
################################################################################
        truss_id = self.elem_ids['truss']
        truss_elems = self.elem_tags['truss']
        for elem_tag in truss_elems:
            i_nd = elem_tag - truss_id
            j_nd = elem_tag - truss_id + 1
            self._model.element('Truss', elem_tag, i_nd, j_nd, A_rigid, elastic_mat_tag)
            
        diaph_id = self.elem_ids['diaphragm']
        diaph_elems = self.elem_tags['diaphragm']
        for elem_tag in diaph_elems:
            i_nd = elem_tag - diaph_id
            j_nd = elem_tag - diaph_id + 1
            self._model.element('elasticBeamColumn', elem_tag, i_nd, j_nd, 
                        A_rigid, Es, Gs, J, I_rigid, I_rigid, beam_transf_tag)
            
################################################################################
# Isolators
################################################################################

        if self.isolator_system == 'TFP':
            
            # TFP system
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
            
            L1      = R1 - h1
            L2      = R2 - h2
    
            # uLim    = 2*d1 + 2*d2 + L1*d2/L2 - L1*d2/L2
    
            # friction pendulum system
            # kv = EASlider/hSlider
            kv = 6*1000*kip/inch
            self._model.uniaxialMaterial('Elastic', fps_vert_tag, kv)
            self._model.uniaxialMaterial('Elastic', fps_rot_tag, 10.0)
    
    
            # Define friction model for FP elements
            # command: frictionModel Coulomb tag mu
            self._model.frictionModel('Coulomb', friction_1_tag, self.mu_1)
            self._model.frictionModel('Coulomb', friction_2_tag, self.mu_2)
    
    
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
                self._model.element('TripleFrictionPendulum', elem_tag, i_nd, j_nd,
                            friction_1_tag, friction_2_tag, friction_2_tag,
                            fps_vert_tag, fps_rot_tag, fps_rot_tag, fps_rot_tag,
                            L1, L2, L2, d1, d2, d2,
                            p_vert, uy, kvt, minFv, 1e-5)
        else:
            # LRB modeling
            # dimensions. Material parameters should not be edited without 
            # modifying design script
            # TODO: check if D_outer should be -1.0...
            K_bulk = 290.0*ksi
            G_r = 0.060*ksi
            D_inner = self.d_lead
            D_outer = self.d_bearing
            t_shim = 0.13*inch
            t_rubber_whole = self.t_r
            n_layers = int(self.n_layers)
            t_layer = t_rubber_whole/n_layers
            
            # calculate yield strength. this assumes design was done correctly
            pi = 3.14159
            f_y_Pb = 1.5 # ksi, shear yield strength
            
            Fy_LRB = f_y_Pb*pi*D_inner**2/4
            
            # Q_L = f_y_Pb*pi*D_inner**2/4
            # # N_lb = 4*self.num_bays
            # # Q_L = self.Q * self.W / N_lb
            
            alpha = 1.0/self.k_ratio
            # Fy_LRB = Q_L/(1 - alpha)
            
            kc = 10.0
            phi_M = 0.5
            ac = 1.0
            
            qL_imp = 0.4046256704 # (lbs/in^3) density of lead 
            cL_imp = 0.03076 # (Btu/lb/degF) specific heat of lead at room temp
            kS_imp = 26.0*12.0 # (Btu/(hr*in*F)) thermal conductivity of steel 
            aS_imp = 0.018166036 # (in^2/s) thermal diffusivity of steel 

            sdr = 0.5
            # guesstimate 60% of the entire bearing volume's mass in lead
            # mb = qL_imp*pi*D_outer**2/4*t_rubber_whole*0.6
            mb = 0.0
            cd = 0.0
            tc = 1.0
            
            tag_1 = 0 # cavitation
            tag_2 = 1 # buckling load variation
            tag_3 = 1 # horiz stiffness variation
            tag_4 = 0 # vertical stiffness variation
            tag_5 = 0 # heat
            
            addl_params = [0, 0, 1, 1, 0, 0,
                           kc, phi_M, ac, sdr, mb, cd, tc,
                           qL_imp, cL_imp, kS_imp, aS_imp,
                           tag_1, tag_2, tag_3, tag_4, tag_5]
            
            # define 2-D isolation layer 
            isol_id = self.elem_ids['isolator']
            base_id = self.elem_ids['base']
            isol_elems = self.elem_tags['isolator']
            
            for elem_idx, elem_tag in enumerate(isol_elems):
                
                # if it is an unstacked bearing, it will have xx0x
                if (elem_tag//10)%10 == 0:
                    i_nd = elem_tag - isol_id
                    j_nd = elem_tag - isol_id - base_id + 10
                else:
                    bay_pos = elem_tag % 10
                    i_nd = base_id + bay_pos
                    j_nd = i_nd - base_id + 10
                    
                self._model.element('LeadRubberX', elem_tag, i_nd, j_nd, Fy_LRB, alpha,
                            G_r, K_bulk, D_inner, D_outer,
                            t_shim, t_layer, n_layers, *addl_params)
  
################################################################################
# Walls
################################################################################
        
        # define impact moat as ZeroLengthImpact3D elements
        # https://opensees.berkeley.edu/wiki/index.php/Impact_Material
        # assume slab of base layer is 6 in
        # model half of slab
        L_bldg = self.L_bldg
        VSlab = (6*inch)*(L_bldg/2*ft)*(L_bldg*ft)
        pi = 3.14159
        RSlab = (3/4/pi*VSlab)**(1/3)
        
        # assume wall extends 1 ft up to isolation layer and is 12 in thick
        VWall = (12*inch)*(1*ft)*(L_bldg/2*ft)
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
    
        moat_gap = self.D_m * self.moat_ampli
    
        self._model.uniaxialMaterial('ImpactMaterial', impact_mat_tag, 
                             K1, K2, -delY, -moat_gap)
        
        # command: element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, 
        #   '-dir', *dirs, <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)
        wall_elems = self.elem_tags['wall']
        
        self._model.element('zeroLength', wall_elems[0], wall_nodes[0], 10,
                    '-mat', impact_mat_tag,
                    '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
        self._model.element('zeroLength', wall_elems[1], 10+n_bays, wall_nodes[1], 
                    '-mat', impact_mat_tag,
                    '-dir', 1, '-orient', 1, 0, 0, 0, 1, 0)
        
        float_nodes = self.floating_nodes()
        
        # # assert that all nodes are connected (float_nodes is empty)
        # assert ((not float_nodes) == True), 'Some nodes are not connected.'
        
        print('Elements placed.')
        
    def apply_grav_load(self):
        import opensees.openseespy as ops
        
        superstructure_system = self.superstructure_system
        
        grav_series   = 1
        grav_pattern  = 1
        
        # get loads
        w_cases = self.all_w_cases
        plc_cases = self.all_Plc_cases
        
        ft = 12.0
        w_floor = w_cases['1.0D+0.5L'] / ft
        p_lc = plc_cases['1.0D+0.5L']
        
        # create TimeSeries
        self._model.timeSeries("Linear", grav_series)

        # create plain load pattern
        self._model.pattern('Plain', grav_pattern, grav_series)
        
        if superstructure_system == 'CBF':
            # get elements
            brace_beams = self.elem_tags['brace_beams']
            beams = self.elem_tags['beam']
            brace_beam_id = self.elem_ids['brace_beam']
            brace_top_nodes = self.node_tags['brace_top']
            
            ghost_beams = [beam_tag//10 for beam_tag in brace_beams
                           if (beam_tag%brace_beam_id in brace_top_nodes)]
            grav_beams = [beam_tag for beam_tag in beams
                          if beam_tag not in ghost_beams]
            
            brace_beam_id = self.elem_ids['brace_beam']
            beam_id = self.elem_ids['beam']
            
            # line loads on elements
            # this loading scheme assumes that local-y is vertical direction (global-z)
            # this is currently true for both MRF and CBF beams
            for elem in brace_beams:
                attached_node = elem - brace_beam_id
                floor_idx = int(str(attached_node)[0]) - 1
                w_applied = w_floor[floor_idx]
                self._model.eleLoad('-ele', elem, '-type', '-beamUniform', 
                            -w_applied, 0.0)
                
            for elem in grav_beams:
                attached_node = elem - beam_id
                floor_idx = attached_node//10 - 1 
                w_applied = w_floor[floor_idx]
                self._model.eleLoad('-ele', elem, '-type', '-beamUniform', 
                            -w_applied, 0.0)
                
        elif superstructure_system == 'MF':
            beams = self.elem_tags['beam']
            beam_id = self.elem_ids['beam']
                
            for elem in beams:
                attached_node = elem - beam_id
                floor_idx = attached_node//10 - 1 
                w_applied = w_floor[floor_idx]
                self._model.eleLoad('-ele', elem, '-type', '-beamUniform', 
                            -w_applied, 0.0)
         
        # line loads on diaphragm
        diaph_elems = self.elem_tags['diaphragm']
        for elem in diaph_elems:
            w_applied = w_floor[0]
            self._model.eleLoad('-ele', elem, '-type', '-beamUniform', 
                        -w_applied, 0.0)
        
        # point loads on LC
        lc_nodes = self.node_tags['leaning']
        for nd in lc_nodes:
            floor_idx = nd//10 - 1
            p_applied = p_lc[floor_idx]
            self._model.load(nd, 0, 0, -p_applied, 0, 0, 0, 
                             pattern=grav_pattern)
            
        # The following assumes two lateral frames. If more, then fix
        isol_sys = self.isolator_system
        if isol_sys == 'TFP':
            # load right above isolation layer to increase stiffness to half-building for TFP
            # line load accounts for Lbay/2 of tributary, we linearly scale
            # to include the remaining portion of Lbldg/2
            ft = 12.0
            L_bay = self.L_bay
            L_bldg = self.L_bldg
            n_bays = self.num_bays
            w_total = w_floor.sum()
            pOuter = w_total*(L_bay/2)*ft* ((L_bldg - L_bay)/L_bay)
            pInner = w_total*(L_bay)*ft* ((L_bldg - L_bay)/L_bay)

            diaph_nds = self.node_tags['diaphragm']
            
            for nd in diaph_nds:
                if (nd%10 == 0) or (nd%10 == n_bays):
                    self._model.load(nd, (0, 0, -pOuter, 0, 0, 0), 
                                     pattern=grav_pattern)
                else:
                    self._model.load(nd, (0, 0, -pInner, 0, 0, 0), 
                                     pattern=grav_pattern)
        
        # ------------------------------
        # Start of analysis generation: gravity
        # ------------------------------

        nStepGravity = 10  # apply gravity in 10 steps
        tol = 1e-5
        dGravity = 1/nStepGravity

        self._model.system("BandGeneral")
        self._model.test("NormDispIncr", tol, 15)
        self._model.numberer("RCM")
        self._model.constraints("Plain")
        self._model.integrator("LoadControl", dGravity)
        self._model.algorithm("Newton")
        self._model.analysis("Static")
        self._model.analyze(nStepGravity)

        print("Gravity analysis complete!")

        self._model.loadConst(time=0.0)
        
    def refix(self, nodeTag, action):
        for j in range(1,7):
            self._model.remove('sp', nodeTag, j)
        if(action == "fix"):
            self._model.fix(nodeTag,  1, 1, 1, 1, 1, 1)
        elif(action == "unfix"):
            self._model.fix(nodeTag,  0, 1, 0, 1, 0, 1)
        elif(action == 'fix_lc'):
            self._model.fix(nodeTag,  1, 1, 1, 1, 0, 1)
        elif(action == 'unfix_lc'):
            self._model.fix(nodeTag,  0, 1, 1, 1, 0, 1)
         
    def run_eigen(self):
        
        nEigenJ = 1;                    # how many modes to analyze
        lambdaN  = self._model.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
        lambda1 = lambdaN[0];           # eigenvalue mode i = 1
        wi = lambda1**(0.5)    # w1 (1st mode circular frequency)
        T_1 = 2*3.1415/wi      # 1st mode period of the structure
        
        print("T_1 = %.3f s" % T_1)   
        
        return(T_1)

    
    def provide_damping(self, regTag, method='SP',
                        zeta=[0.05], modes=[1]):
        import opensees.openseespy as ops
        
        diaph_nodes = self.node_tags['diaphragm']
        # fix base for Tfb
        lc_base = self.node_tags['leaning'][0]
        for diaph_nd in diaph_nodes:
            self.refix(diaph_nd, "fix")
        self.refix(lc_base, 'fix_lc')

        nEigenJ = 2;                    # how many modes to analyze
        lambdaN  = ops.eigen(nEigenJ);       # eigenvalue analysis for nEigenJ modes
        lambda1 = lambdaN[modes[0]-1];           # eigenvalue mode i = 1
        wi = lambda1**(0.5)    # w1 (1st mode circular frequency)
        Tfb = 2*3.1415/wi      # 1st mode period of the structure
        
        if method=='Rayleigh':
            lambdaJ = lambdaN[modes[-1]-1]
            wj = lambdaJ**(0.5)
            
        print("Tfb = %.3f s" % Tfb)          

        # unfix base
        for diaph_nd in diaph_nodes:
            self.refix(diaph_nd, "unfix")
        self.refix(lc_base, 'unfix_lc')
        # provide damping to superstructure only
        import numpy as np
        
        if method == 'Rayleigh':
            A = np.array([[1/wi, wi],[1/wj, wj]])
            b = np.array([zeta[0],zeta[-1]])
            
            x = np.linalg.solve(A,2*b)
        else:
            betaK       = 0.0
            betaKInit   = 0.0
            a1 = 2*zeta[0]/wi
        
        all_elems = self._model.getEleTags()

        # elems that don't need damping
        wall_elems = self.elem_tags['wall']
        isol_elems = self.elem_tags['isolator']
        truss_elems = self.elem_tags['truss']
        lc_elems = self.elem_tags['leaning'] + self.elem_tags['lc_spring']
        diaph_elems = self.elem_tags['diaphragm']
        non_damped_elems = (wall_elems + isol_elems + truss_elems + 
                            lc_elems + diaph_elems)
        
        damped_elems = [elem for elem in all_elems 
                        if elem not in non_damped_elems]
        
        # stiffness proportional
        if method == 'SP':
            self._model.region(regTag, '-ele',
                *damped_elems,
                '-rayleigh', 0.0, betaK, betaKInit, a1)
            print('Structure damped with %0.1f%% at frequency %0.2f Hz' % 
                  (zeta[0]*100, wi))
        elif method == 'Rayleigh':
            self._model.region(regTag, '-ele',
                *damped_elems,
                '-rayleigh', x[0], betaK, betaKInit, x[1])
            
        return(Tfb)

    def run_pushover(self, max_drift_ratio=0.1, 
                     data_dir='./outputs/pushover/'):
        
        import opensees.openseespy as ops
        
        # get list of relevant nodes
        superstructure_system = self.superstructure_system
        isol_id = self.elem_ids['isolator']
        isol_system = self.isolator_system
        isols = self.elem_tags['isolator']
        base_id = self.elem_ids['base']
        
        if superstructure_system == 'CBF':
            # extract nodes that belong to the braced portion
            brace_beam_ends = self.node_tags['brace_beam_end']
            left_col_digit = min([nd%10 for nd in brace_beam_ends])
            
            # get the list of nodes in all stories for the first outer and inner column
            outer_col_nds = [nd for nd in brace_beam_ends
                             if nd%10 == left_col_digit]
            inner_col_nds = [nd+1 for nd in outer_col_nds]
            
            # insert the isolation layer
            outer_col_nds.insert(0, outer_col_nds[0]-10)
            inner_col_nds.insert(0, inner_col_nds[0]-10)
            
            # record brace nodes' displacements
            brace_tops = self.node_tags['brace_top']
            brace_bottoms = self.node_tags['brace_bottom']
            brace_mids = self.node_tags['brace_mid']
            
            # lowest left bay, left brace displacements
            top_node = min(brace_tops)
            mid_node = min(brace_mids)
            bottom_node = min(brace_bottoms)
            self._model.recorder('Node', '-file', data_dir+'brace_node_disp.csv','-time',
                '-node', bottom_node, mid_node, top_node, 
                '-dof', 1, 3, 'disp')
            
            # force at corresponding top node
            self._model.recorder('Node','-node', top_node,
                         '-file', data_dir+'brace_top_node_force.csv', 
                         '-dof', 1, 3, 'reaction')
            
            # first story, leftmost bay, left brace
            brace_ghosts = self.elem_tags['brace_ghosts']
            bottom_left_ghost = min(brace_ghosts)
            bottom_right_ghost = bottom_left_ghost + 98
            self._model.recorder('Element','-ele', bottom_left_ghost,
                         '-file',data_dir+'left_ghost_deformation.csv', '-time',
                         'deformations')
            self._model.recorder('Element','-ele', bottom_right_ghost,
                         '-file',data_dir+'right_ghost_deformation.csv', '-time',
                         'deformations')
            
            # first story, leftmost bay, left brace
            braces = self.elem_tags['brace']
            bottom_left_brace = min(braces)
            # corresponding right brace (100 to shift bay, -2 for 9-7 difference)
            bottom_right_brace = bottom_left_brace + (100 - 2)
            
            selected_brace = get_shape(self.brace[0],'brace')
            d_brace = selected_brace.iloc[0]['b']
            
            self._model.recorder('Element','-ele', bottom_left_brace,
                         '-file',data_dir+'brace_left.csv', '-time',
                         'section','fiber', 0.0, -d_brace/2, 'stressStrain')
            
            self._model.recorder('Element','-ele', bottom_right_brace,
                         '-file',data_dir+'brace_right.csv', '-time',
                         'section','fiber', 0.0, -d_brace/2, 'stressStrain')
            
        else:
            floor_nodes = self.node_tags['floor']
            
            # get the list of nodes in all stories for the first outer and inner column
            outer_col_nds = [nd for nd in floor_nodes
                             if nd%10 == 0]
            
            inner_col_nds = [nd+1 for nd in outer_col_nds]
            
        # lateral frame story displacement
        self._model.recorder('Node', '-file', data_dir+'outer_col_disp.csv','-time',
                     '-node', *outer_col_nds, '-dof', 1, 'disp')
        self._model.recorder('Node', '-file', data_dir+'inner_col_disp.csv','-time',
                     '-node', *inner_col_nds, '-dof', 1, 'disp')
        
        # vertical frame story displacement
        self._model.recorder('Node', '-file', data_dir+'outer_col_vert.csv','-time',
                     '-node', *outer_col_nds, '-dof', 3, 'disp')
        self._model.recorder('Node', '-file', data_dir+'inner_col_vert.csv','-time',
                     '-node', *inner_col_nds, '-dof', 3, 'disp')
        
        # if lead rubber bearing, take a non-edge bearing 
        # (edge bearing was "stacked")
        if isol_system == 'LRB':
            isol_elem = isols[1]
            isol_node = isol_elem - isol_id - base_id + 10
        # TFPs aren't stacked, so just take left-most 
        else:
            # get the leftmost isolator
            isol_elem = isols[0]
            isol_node = isol_elem - isol_id - base_id + 10
        
        # isolator node displacement of outer column
        self._model.recorder('Node', '-file', data_dir+'isolator_displacement.csv', 
                     '-time', '-node', isol_node, '-dof', 1, 3, 5, 'disp')
        
        # isolator response of beneath outer column
        self._model.recorder('Element', '-file', data_dir+'isolator_forces.csv',
                     '-time', '-ele', isol_elem, 'localForce')
        
        base_nodes = self.node_tags['base']
        wall_nodes = self.node_tags['wall']
        
        ground_nodes = base_nodes + wall_nodes
        self._model.recorder('Node', '-file', data_dir+'ground_rxn.csv', 
                     '-time', '-node', 
                     *ground_nodes, '-dof', 1, 'reaction')
        
        # Set lateral load pattern with a Linear TimeSeries
        pushover_pattern_tag  = 400
        pushover_series_tag   = 4
        ops.pattern('Plain', pushover_pattern_tag, "Linear")

        import time
        t0 = time.time()
        print('Running pushover...')
        
        Fx_vec = self.Fx
        
        # Create nodal loads at outer column nodes
        #    nd    FX  FY  FZ MX MY MZ
        for fl_idx, Fx in enumerate(Fx_vec):
            self._model.load(outer_col_nds[fl_idx+1], (Fx, 0.0, 0.0, 0.0, 0.0, 0.0),
                     pattern=pushover_pattern_tag)
        
        #----------------------------------------------------
        # Start of modifications to analysis for push over
        # ----------------------------------------------------
        self._model.wipeAnalysis()

        # units: in, kip, s
        # dimensions

        hsx     = self.hsx
        # Set some parameters
        dU = 0.001  # Displacement increment

        # Change the integration scheme to be displacement control
        #                             node dof init Jd min max
        self._model.integrator('DisplacementControl', outer_col_nds[-1], 1, dU, 1, dU, dU)
        
        # ------------------------------
        # Finally perform the analysis
        # ------------------------------

        # Set some parameters
        maxU = max_drift_ratio*hsx.sum()  # Max displacement
        nSteps = int(round(maxU/dU))
        ok = 0

        # Create the system of equation, a sparse solver with partial pivoting
        self._model.system('UmfPack')

        # Create the constraint handler, the transformation method
        self._model.constraints('Plain')

        # Create the DOF numberer, the reverse Cuthill-McKee algorithm
        self._model.numberer('RCM')

        self._model.test('NormUnbalance', 1.0e-3, 4000)
        self._model.algorithm('Newton')
        # Create the analysis object
        self._model.analysis('Static')

        ok = self._model.analyze(nSteps)
        # for gravity analysis, load control is fine, 0.1 is the load factor increment 
        # (http://opensees.berkeley.edu/wiki/index.php/Load_Control)

        testList = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 
                    4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 
                    6: 'NormUnbalance'}
        algoList = {1:'KrylovNewton', 2: 'SecantNewton' , 
                    4: 'RaphsonNewton',5: 'PeriodicNewton', 
                    6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}
                    
        for i in testList:
            for j in algoList:

                if ok != 0:
                    if j < 4:
                        self._model.algorithm(algoList[j], '-initial')
                        
                    else:
                        self._model.algorithm(algoList[j])
                        
                    self._model.test(testList[i], 1e-3, 1000)
                    ok = self._model.analyze(nSteps)                            
                    print(testList[i], algoList[j], ok)             
                    if ok == 0:
                        break
                else:
                    continue
        
        self._model.wipe()
        
        tp = time.time() - t0
        minutes = tp//60
        seconds = tp - 60*minutes
        print('Pushover complete. Time elapsed %dm %ds.' % (minutes, seconds))

    def run_ground_motion(self, gm_name, scale_factor, dt_transient, T_end=60.0,
                          gm_dir='../resource/ground_motions/PEERNGARecords_Unscaled/',
                          data_dir='./outputs/'):
        
        # Recorders
        import opensees.openseespy as ops
        
        # get list of relevant nodes
        superstructure_system = self.superstructure_system
        isol_system = self.isolator_system
        isols = self.elem_tags['isolator']
        walls = self.elem_tags['wall']
        isol_id = self.elem_ids['isolator']
        base_id = self.elem_ids['base']
        h_story = self.h_story
        # print all warnings to log file
        ops.logFile(data_dir+'run.log', '-noEcho')
        
        if superstructure_system == 'CBF':
            # extract nodes that belong to the braced portion
            brace_beam_ends = self.node_tags['brace_beam_end']
            left_col_digit = min([nd%10 for nd in brace_beam_ends])
            
            # get the list of nodes in all stories for the first outer and inner column
            outer_col_nds = [nd for nd in brace_beam_ends
                             if nd%10 == left_col_digit]
            inner_col_nds = [nd+1 for nd in outer_col_nds]
            
            # insert the isolation layer
            outer_col_nds.insert(0, outer_col_nds[0]-10)
            inner_col_nds.insert(0, inner_col_nds[0]-10)
            
            '''
            # record brace nodes' displacements
            brace_tops = self.node_tags['brace_top']
            brace_bottoms = self.node_tags['brace_bottom']
            brace_mids = self.node_tags['brace_mid']
            
            # lowest left bay, left brace displacements
            top_node = min(brace_tops)
            mid_node = min(brace_mids)
            bottom_node = min(brace_bottoms)
            self._model.recorder('Node', '-file', data_dir+'brace_node_disp.csv','-time',
                '-node', bottom_node, mid_node, top_node, 
                '-dof', 1, 3, 'disp')
            
            # force at corresponding top node
            self._model.recorder('Node','-node', top_node,
                         '-file', data_dir+'brace_top_node_force.csv', 
                         '-dof', 1, 3, 'reaction')
            '''
            
            # first story, leftmost bay, left brace
            brace_ghosts = self.elem_tags['brace_ghosts']
            bottom_left_ghost = min(brace_ghosts)
            bottom_right_ghost = bottom_left_ghost + 98
            self._model.recorder('Element','-ele', bottom_left_ghost,
                         '-file',data_dir+'left_ghost_deformation.csv', '-time',
                         'deformations')
            self._model.recorder('Element','-ele', bottom_right_ghost,
                         '-file',data_dir+'right_ghost_deformation.csv', '-time',
                         'deformations')
            
            # first story, leftmost bay, left brace
            braces = self.elem_tags['brace']
            bottom_left_brace = min(braces)
            # corresponding right brace 
            bottom_right_brace = bottom_left_brace + 10
            
            selected_brace = get_shape(self.brace[0],'brace')
            d_brace = selected_brace.iloc[0]['b']
            
            self._model.recorder('Element','-ele', bottom_left_brace,
                         '-file',data_dir+'brace_left_str.csv', '-time',
                         'section','fiber', 0.0, -d_brace/2, 'stressStrain')
            
            self._model.recorder('Element','-ele', bottom_right_brace,
                         '-file',data_dir+'brace_right_str.csv', '-time',
                         'section','fiber', 0.0, -d_brace/2, 'stressStrain')
            
            self._model.recorder('Element','-ele', bottom_left_brace, '-time',
                         '-file',data_dir+'brace_left_force.csv', 'basicForce')
            
            self._model.recorder('Element','-ele', bottom_right_brace, '-time',
                         '-file',data_dir+'brace_right_force.csv', 'basicForce')
            
        else:
            floor_nodes = self.node_tags['floor']
            
            # get the list of nodes in all stories for the first outer and inner column
            outer_col_nds = [nd for nd in floor_nodes
                             if nd%10 == 0]
            
            inner_col_nds = [nd+1 for nd in outer_col_nds]
            
        # if lead rubber bearing, take a non-edge bearing 
        # (edge bearing was "stacked")
        if isol_system == 'LRB':
            isol_elem = isols[1]
            isol_node = isol_elem - isol_id - base_id + 10
        # TFPs aren't stacked, so just take left-most 
        else:
            # get the leftmost isolator
            isol_elem = isols[0]
            isol_node = isol_elem - isol_id - base_id + 10
        
        # isol nodes are diaphragm nodes
        isol_nodes_all = self.node_tags['diaphragm']
        
        # open(data_dir+'model.out', 'w').close()
        # ops.printModel('-file', data_dir+'model.out')
        
        # lateral frame story displacement
        self._model.recorder('Node', '-file', data_dir+'outer_col_disp.csv','-time',
                     '-node', *outer_col_nds, '-dof', 1, 'disp')
        self._model.recorder('Node', '-file', data_dir+'inner_col_disp.csv','-time',
                     '-node', *inner_col_nds, '-dof', 1, 'disp')
        
        # vertical frame story displacement
        self._model.recorder('Node', '-file', data_dir+'outer_col_vert.csv','-time',
                     '-node', *outer_col_nds, '-dof', 3, 'disp')
        self._model.recorder('Node', '-file', data_dir+'inner_col_vert.csv','-time',
                     '-node', *inner_col_nds, '-dof', 3, 'disp')
        
        # lateral frame story velocity
        self._model.recorder('Node', '-file', data_dir+'outer_col_vel.csv','-time',
                     '-node', *outer_col_nds, '-dof', 1, 'vel')
        self._model.recorder('Node', '-file', data_dir+'inner_col_vel.csv','-time',
                     '-node', *inner_col_nds, '-dof', 1, 'vel')
        
        
        # isolator node displacement of outer column
        self._model.recorder('Node', '-file', data_dir+'isolator_displacement.csv', 
                     '-time', '-node', isol_node, '-dof', 1, 3, 5, 'disp')
        
        # isolator response of beneath outer column
        self._model.recorder('Element', '-file', data_dir+'isolator_forces.csv',
                     '-time', '-ele', isol_elem, 'localForce')
        
        base_nodes = self.node_tags['base']
        
        if isol_system == 'LRB':
            self._model.recorder('Node', '-file', data_dir+'lrb_disp.csv', 
                         '-time', '-node', *isol_nodes_all, '-dof', 1, 'disp')
        elif isol_system == 'TFP':
            self._model.recorder('Node', '-file', data_dir+'tfp_disp.csv', 
                         '-time', '-node', *isol_nodes_all, '-dof', 1, 'disp')
            self._model.recorder('Node', '-file', data_dir+'tfp_base_vert.csv', 
                         '-time', '-node', 
                         *base_nodes, '-dof', 3, 'reaction')
        
        self._model.recorder('Node', '-file', data_dir+'base_rxn.csv', 
                     '-time', '-node', 
                     *base_nodes, '-dof', 1, 'reaction')
        
        self._model.recorder('Node', '-file', data_dir+'diaph_rxn.csv', 
                     '-time', '-node', 
                     *isol_nodes_all, '-dof', 1, 'reaction')
        
        story_1_nodes = [x+10 for x in isol_nodes_all]
        
        self._model.recorder('Node', '-file', data_dir+'story_1_rxn.csv', 
                     '-time', '-node', 
                     *story_1_nodes, '-dof', 1, 'reaction')
        
        # gusset plate?
        # beam force?
        # column force?
        
        self._model.recorder('Element', '-file', data_dir+'impact_forces.csv', 
                     '-time', '-ele', *walls, 'basicForce')
        self._model.recorder('Element', '-file', data_dir+'impact_disp.csv', 
                     '-time', '-ele', *walls, 'basicDeformation')
        
        # diaphragm?
        diaph_elems = self.elem_tags['diaphragm']
        self._model.recorder('Element', '-file', data_dir+'diaphragm_forces.csv', 
                     '-time', '-ele', diaph_elems[0], 'basicForce')
        
        # leaning column?
        

        ops.wipeAnalysis()

        # Uniform Earthquake ground motion (uniform acceleration input at all support nodes)
        GMDirection = 1  # ground-motion direction
        
        print('Current ground motion: %s at scale %.2f' % (gm_name, scale_factor))

        ops.constraints('Plain')
        ops.numberer('RCM')

        ops.system('UmfPack')

        if superstructure_system == 'CBF':
            
            # Convergence Test: maximum number of iterations that will be performed
            maxIterDynamic      = 100
            
            # Convergence Test: flag used to print information on convergence
            printFlagDynamic    = 0  
            
            # algorithmTypeDynamic    = 'Broyden'
            # ops.algorithm(algorithmTypeDynamic, 8)
            algorithmTypeDynamic    = 'KrylovNewton'
            ops.algorithm(algorithmTypeDynamic)
            
            # # Convergence Test: tolerance
            # testTypeDynamic = 'EnergyIncr'
            # tolDynamic = 1e-6
            
            # # TRBDF2 integrator, best with energy
            # ops.integrator('TRBDF2')
            
            # Convergence Test: tolerance
            testTypeDynamic     = 'EnergyIncr'
            tolDynamic = 1e-8
            
            # Newmark-integrator gamma parameter (also HHT)
            newmarkGamma = 0.5
            newmarkBeta = 0.25
            ops.integrator('Newmark', newmarkGamma, newmarkBeta)
            
        else:
            # Convergence Test: maximum number of iterations that will be performed
            maxIterDynamic      = 100
            
            # Convergence Test: flag used to print information on convergence
            printFlagDynamic    = 0  
            
            # algorithmTypeDynamic    = 'Broyden'
            # ops.algorithm(algorithmTypeDynamic, 8)
            algorithmTypeDynamic    = 'Newton'
            self._model.algorithm(algorithmTypeDynamic)
            
            # Convergence Test: tolerance
            testTypeDynamic     = 'NormDispIncr'
            tolDynamic = 1e-8
            
            # Newmark-integrator gamma parameter (also HHT)
            newmarkGamma = 0.5
            newmarkBeta = 0.25
            self._model.integrator('Newmark', newmarkGamma, newmarkBeta)
            # self._model.integrator('CentralDifference')
        
        self._model.test(testTypeDynamic, tolDynamic, maxIterDynamic, printFlagDynamic)
            
        self._model.analysis('Transient')

        #  ---------------------------------    perform Dynamic Ground-Motion Analysis
        # the following commands are unique to the Uniform Earthquake excitation

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
        self._model.timeSeries('Path', eq_series_tag, '-dt', dt, 
                       '-filePath', outFile, '-factor', GMfatt)     
        # create uniform excitation
        self._model.pattern('UniformExcitation', eq_pattern_tag, 
                    GMDirection, '-accel', eq_series_tag)          

        # set recorder for absolute acceleration (requires time series defined)
        self._model.recorder('Node', '-file', data_dir+'outer_col_acc.csv',
                     '-timeSeries', eq_series_tag, '-time',
                     '-node', *outer_col_nds, '-dof', 1, 'accel')
        self._model.recorder('Node', '-file', data_dir+'inner_col_acc.csv',
                     '-timeSeries', eq_series_tag, '-time',
                     '-node', *inner_col_nds, '-dof', 1, 'accel')
        
        import numpy as np
        n_steps = int(np.floor(T_end/dt_transient))
        
        # actually perform analysis; returns ok=0 if analysis was successful
        
        import time
        t0 = time.time()
        
        # Convergence loop, careful with Broyden/BFGS with energy
        ok = self._model.analyze(n_steps, dt_transient)   
        
        # drift limits triggering halt to analysis
        cbf_drift_limit = 0.10
        mf_drift_limit = 0.20
        
        # if good collapse, halt. if non-convergent collapse, discard and retry
        if superstructure_system == 'MF':
            collapse_status = determine_collapse(self._model, self._model, outer_col_nds, h_story, mf_drift_limit)
        else:
            collapse_status = determine_collapse(self._model, outer_col_nds, h_story, cbf_drift_limit)
        
        if collapse_status == 'collapse':
            ok = 0
            print('Collapse occurred (MF drift 0.2 | CBF drift 0.1).')
        elif collapse_status == 'non-convergence':
            ok = -3
            t_final = self._model.getTime()
            tp = time.time() - t0
            minutes = tp//60
            seconds = tp - 60*minutes
            print('Drift is beyond convergence. Ending...')
            print('Ground motion done. End time: %.4f s' % t_final)
            print('Analysis time elapsed %dm %ds.' % (minutes, seconds))
            self._model.wipe()
            return(ok)
            
        # If analysis failed reasonably
        if ok != 0:
            self._model.analysis('Transient')
            curr_time = self._model.getTime()
            print("Convergence issues at time: ", curr_time)
            # The analysis will be time-controlled and is done for the remaining time
                
            if superstructure_system == 'MF':
                ok = 0
                while (curr_time < T_end) and (ok == 0):
                    curr_time     = self._model.getTime()
                    ok = self._model.analyze(1, dt_transient)
                        
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, mf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        print("Trying Newton with line search ...")
                        self._model.algorithm('NewtonLineSearch')
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to Newton")
                            self._model.algorithm('Newton')
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, mf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        print('Trying Broyden ... ')
                        algorithmTypeDynamic = 'Broyden'
                        self._model.algorithm(algorithmTypeDynamic)
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to Newton")
                            self._model.algorithm('Newton')
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, mf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        print('Trying BFGS ... ')
                        algorithmTypeDynamic = 'BFGS'
                        self._model.algorithm(algorithmTypeDynamic)
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to Newton")
                            self._model.algorithm('Newton')
            else:
                ok = 0
                while (curr_time < T_end) and (ok == 0):
                    curr_time     = self._model.getTime()
                    ok = self._model.analyze(1, dt_transient)
                    
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        
                        print("Trying Newton with line search ...")
                        self._model.algorithm('NewtonLineSearch')
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to KrylovNewton")
                            self._model.algorithm('KrylovNewton')
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        print('Trying Broyden ... ')
                        self._model.algorithm('Broyden')
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to KrylovNewton")
                            self._model.algorithm('KrylovNewton')
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        print('Trying BFGS ... ')
                        self._model.algorithm('BFGS')
                        ok = self._model.analyze(1, dt_transient)
                        if ok == 0:
                            print("That worked. Back to KrylovNewton")
                            self._model.algorithm('KrylovNewton')
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        curr_time     = self._model.getTime()
                        print("Trying KrylovNewton with 1/5 dt for 10 steps ...")
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(10, dt_transient/5.0)
                        if ok == 0:
                            print("That worked. Back to regular dt.")
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        curr_time     = self._model.getTime()
                        print("Trying KrylovNewton with 1/10 dt for 10 steps ...")
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(10, dt_transient/10.0)
                        if ok == 0:
                            print("That worked. Back to regular dt.")
                    if ok != 0:
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status == 'collapse':
                            print('Collapse triggered.')
                            ok = 0
                            break
                        elif collapse_status == 'non-convergence':
                            print('Drift is beyond convergence. Ending...')
                            ok = -3
                            break
                        curr_time     = self._model.getTime()
                        print("Trying KrylovNewton with 1/100 dt for 10 steps ...")
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(10, dt_transient/100.0)
                        if ok == 0:
                            print("That worked. Back to regular dt.")
                    if ok != 0:
                        print('CBF convergence loop exhausted. Ending run...')
            '''
            else:
                ok = 0
                curr_time     = self._model.getTime()
                
                while (curr_time < T_end) and (ok == 0):
                    
                    # check for collapse first
                    collapse_status = determine_collapse(self._model, 
                        outer_col_nds, h_story, cbf_drift_limit)
                    if collapse_status:
                        print('Collapse triggered.')
                        ok = 0
                        break
                    else:
                        ok = -1
                    
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / (dt_transient/2.0)))
                        print("Trying KrylovNewton with 1/2 dt for 10 steps ...")
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(10, dt_transient/2.0)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                       
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / dt_transient))
                        print("Going back to original ...")
                        self._model.test('EnergyIncr', 1.0e-2, 100, 0)
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(remaining_steps, dt_transient)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / 0.001))
                        print("Trying KrylovNewton with 0.001 dt for 10 steps ...")
                        self._model.test('EnergyIncr', 1.0e-2, 100, 0)
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(remaining_steps, 0.001)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / (dt_transient/2.0)))
                        print("Going back to 1/2 dt for 10 steps ...")
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(10, dt_transient/2.0)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / 0.0001))
                        print("Trying KrylovNewton with 0.0001 dt for 5 steps ...")
                        self._model.test('EnergyIncr', 1.0e-2, 100, 0)
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(5, 0.001)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / dt_transient))
                        print("Going back to original ...")
                        self._model.test('EnergyIncr', 1.0e-2, 100, 0)
                        self._model.algorithm('KrylovNewton')
                        ok = self._model.analyze(remaining_steps, dt_transient)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                    if ok != 0:
                        curr_time     = self._model.getTime()
                        remaining_time = T_end - curr_time
                        remaining_steps = int(np.floor(remaining_time / 0.0001))
                        print("Trying Newton with fixed iters ...")
                        self._model.test('FixedNumIter', 50)
                        self._model.integrator('NewmarkHSFixedNumIter', 0.5, 0.25)
                        self._model.algorithm('Newton')
                        ok = self._model.analyze(10, dt_transient)
                        collapse_status = determine_collapse(self._model, 
                            outer_col_nds, h_story, cbf_drift_limit)
                        if collapse_status:
                            ok = 0
                        
                    curr_time     = self._model.getTime()
            '''
        
            # # cutting time convergence loop
            # else:
            #     ok = 0
            #     while (curr_time < T_end) and (ok == 0):
            #         curr_time     = self._model.getTime()
            #         ok = self._model.analyze(1, dt_transient)
            #         if ok != 0:
            #             print("Cutting time step for ten step...")
            #             self._model.algorithm('NewtonLineSearch')
            #             ok = self._model.analyze(10, dt_transient/10)
            #             if ok == 0:
            #                 print("That worked. Back to Newton")
            #                 self._model.algorithm('Newton')
            #         if ok != 0:
            #             print('CBF convergence loop exhausted. Ending run...')
                
        t_final = self._model.getTime()
        tp = time.time() - t0
        minutes = tp//60
        seconds = tp - 60*minutes
        print('Ground motion done. End time: %.4f s' % t_final)
        print('Analysis time elapsed %dm %ds.' % (minutes, seconds))
        self._model.wipe()
        
        return(ok)
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
    elif member == 'brace':
        shape_db = pd.read_csv(csv_dir+'braceShapes.csv',
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
    
    Zx = float(shape.iloc[0]['Zx'])
    # Sx = float(shape['Sx'])
    Iz = float(shape.iloc[0]['Ix'])
    d = float(shape.iloc[0]['d'])
    htw = float(shape.iloc[0]['h/tw'])
    bftf = float(shape.iloc[0]['bf/2tf'])
    ry = float(shape.iloc[0]['ry'])
    c1 = 25.4
    c2 = 6.895

    # approximate adjustment for isotropic hardening
    My = Fy * Zx * 1.17
    thy = My/(6*Es*Iz/L)
    
    # n is an adjustment to equate stiffness of spring-beam-spring element to 
    # actual stiffness of spring
    n = 10
    Ke = n*My/thy
    # consider using Lb = 0 for beams bc of slab?
    Lb = L
    kappa = 0.4

    if d > 21.0:
        lam = (536*(htw)**(-1.26)*(bftf)**(-0.525)
            *(Lb/ry)**(-0.130)*(c2*Fy/355)**(-0.291))
        thp = (0.318*(htw)**(-0.550)*(bftf)**(-0.345)
            *(Lb/ry)**(-0.0230)*(L/d)**(0.090)*(c1*d/533)**(-0.330)*
            (c2*Fy/355)**(-0.130))
        thpc = (7.50*(htw)**(-0.610)*(bftf)**(-0.710)
            *(Lb/ry)**(-0.110)*(c1*d/533)**(-0.161)*
            (c2*Fy/355)**(-0.320))
    else:
        lam = 495*(htw)**(-1.34)*(bftf)**(-0.595)*(c2*Fy/355)**(-0.360)
        thp = (0.0865*(htw)**(-0.365)*(bftf)**(-0.140)
            *(L/d)**(0.340)*(c1*d/533)**(-0.721)*
            (c2*Fy/355)**(-0.230))
        thpc = (5.63*(htw)**(-0.565)*(bftf)**(-0.800)
            *(c1*d/533)**(-0.280)*(c2*Fy/355)**(-0.430))
        
    thu = 0.2

    return(Ke, My, lam, thp, thpc, kappa, thu)

###############################################################################
#              Brace geometry
###############################################################################
def determine_collapse(model, nds, h_story, drift_limit):
    import numpy as np
    disp_array = np.array([model.nodeDisp(node, 1)
                               for node in nds])
    drift_array = np.abs(np.diff(disp_array)/(h_story*12.0))
    if np.any(drift_array > drift_limit) and np.all(drift_array < 1.5):
        return 'collapse'
    elif np.any(drift_array > 1.5):
        return 'non-convergence'
    else:
        return 'okay'


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

# quadratic brace coordinates
def quad_brace_coord(nd, L_bay, h_story, camber=0.001, offset=0.25):
    # from mid brace number, get the corresponding top and bottom node numbers
    top_node = nd//100
    
    # extract their corresponding coordinates from the node numbers
    top_x_coord = (top_node%10 + 0.5)*L_bay
    top_y_coord = (top_node//10 - 1)*h_story
    
    # if the xxXx number is 2, the brace connects sw
    # if the xxXx number is 3, the brace connects se
    
    if (nd//10)%2 == 0:
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
    
    num_br_elem = 8
    ele_L = L_eff/num_br_elem
    
    # local coords
    nd_sub = nd%10
    xm = ele_L*nd_sub
    # p = camber*L_eff
    # zm = 4*p/L_eff*xm*(1- xm/L_eff)
    zm = 4*camber/L_eff*xm*(L_eff - xm)
    
    # origin is bottom node, adjusted for gusset plate
    # offset is the shift (+/-) of the bottom gusset plate
    # terminus is top node, adjusted for gusset plate (gusset placed opposite direction)
    x_origin = bot_x_coord + x_offset
    y_offset = offset/2 * h_story
    y_origin = bot_y_coord + y_offset
    
    # angle from the brace vector up to camber
    beta = asin(zm/xm)
    
    # angle from horizontal up to camber
    gamma  = theta + beta
    
    # if the xxXx number is 2, the brace connects sw
    # if the xxXx number is 3, the brace connects se
    if (nd//10)%2 == 0:
        mid_x_coord = x_origin + xm * cos(gamma)
    else:
        mid_x_coord = x_origin - xm * cos(gamma)
    mid_y_coord = y_origin + xm * sin(gamma)
    
    return(mid_x_coord, mid_y_coord)
    
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
    



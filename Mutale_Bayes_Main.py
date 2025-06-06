### This code is used to create a Bayesian Network for the design of a beam in a building under uncertainty. The beam is modelled as a simply supported beam
### It is composed of a number of nodes that represent the properties of the beam, the building and the loads on the beam. The nodes are connected by edges that represent the relationships between the nodes.
### The network is then used to calculate the load resistance ratio of the beam under different conditions. The network is created using the PySMILE library.

import matplotlib.pyplot as plt
import numpy as np
import pysmile_license
import pysmile
import networkx as nx
from scipy.stats import lognorm
import pandas as pd
import csv

class beam_BN:
    
    def __init__(self, beam_num, nint, building_len, building_wid, roof_ang, dec, char_bending, beam_space):
        """
        Initiating the class
        """
        self.initiation()
        #self.read_csv_return_values_priors()
        
        self.define_network()

        self.beam_num = beam_num
        self.nint = nint
        
        self.building_wid = building_wid
        self.building_len = building_len
        self.roof_ang = roof_ang
        self.dec = dec
        self.char_bending = char_bending
        self.beam_space = beam_space

        #self.inference_beam_evidence()
        self.u_output_x, self.u_output_y, self.u_output_z  = self.inference_beam_evidence_u_output(self.load_resist_ratio, beam_num, nint, building_len, building_wid, roof_ang, dec, char_bending, beam_space) # outputs load resistance ratio values and probabilities
        self.vol_output_x, self.vol_output_y, self.vol_output_z = self.inference_beam_evidence_u_output(self.beam_volume_in_roof, beam_num, 36, building_len, building_wid, roof_ang, dec, char_bending, beam_space) # outputs vol and probabilities
        self.beam_resist_x, self.beam_resist_y, self.beam_resist_z  = self.inference_beam_evidence_u_output(self.beam_resist, beam_num, 15, building_len, building_wid, roof_ang, dec, char_bending, beam_space) # outputs beam resistance and probabilities
        self.beam_load_x, self.beam_load_y, self.beam_load_z  = self.inference_beam_evidence_u_output(self.beam_load, beam_num, 9, building_len, building_wid, roof_ang, dec, char_bending, beam_space) # outputs beam load and probabilities
        #self.inference_beam_strength_spacing_evidence()
        
        
        #self.plot_DAG()
        #self.net.write_file("StructuralBN31.xdsl")
        #print("Beam BN complete: Network written to StructuralBN17.xdsl")


        
    def initiation(self):
        """
        Defining the settings.
        """
        self.net = pysmile.Network()
        self.net.set_outlier_rejection_enabled(True)
        self.net.set_sample_count(int(1e5))  # Trade-off between calculation time and precision


    def inference_beam_evidence_u_output(self, node_name, beam_num, nint, building_len, building_wid, roof_ang, dec, char_bending, beam_space):
        #self.net.set_cont_evidence(self.char_bending_str, 24)

        if beam_num > 0:
            self.net.set_cont_evidence(self.beam_section, beam_num)
        if building_len > 0:
            self.net.set_cont_evidence(self.building_length, building_len)
        if building_wid > 0:
            self.net.set_cont_evidence(self.building_width, building_wid)
        if roof_ang > 0:
            self.net.set_cont_evidence(self.roof_angle, roof_ang)
        if dec > 0:
            self.net.set_cont_evidence(self.decade, dec)
        if char_bending > 0:
            self.net.set_cont_evidence(self.char_bending_str, char_bending)
        if beam_space > 0:
            self.net.set_cont_evidence(self.beam_spacing, beam_space)
        
        
        self.update_and_show_stats(self.net)

        y_output, x_input = self.create_node_output(self.net, node_name, nint)
        z_output = self.show_node_stats(self.net, node_name)
        #self.create_node_sub_figure(self.net, axes[2,1], 6, self.beam_volume_in_roof,11, '$V_{purlins}$ [m$^{3}$]',color=color, alpha=0.5)
        return x_input, y_output, z_output
        
    def inference_beam_evidence(self, beam):
             
        self.net.set_cont_evidence(self.beam_section, beam)
        self.update_and_show_stats(self.net)

        self.fig_prior_post_props, self.axes_prior_post_props = self.create_node_figure(f'Building and Beam {beam} properties', 6)
        self.fill_node_figureBeamProps(self.net, self.axes_prior_post_props, color = 'tab:red')
        self.fig_prior_post_props.savefig(f"Beam_num_{beam}_props.png",  bbox_inches='tight')

        self.fig_prior_post_out, self.axes_prior_post_out = self.create_node_figure(f'Mechanical properties, beam length and spacing for Beam {beam}', 6)
        self.fill_node_figureOutput(self.net, self.axes_prior_post_out, color = 'tab:pink')
        self.fig_prior_post_out.savefig(f"Beam_num_{beam}_mech.png",  bbox_inches='tight')

        self.fig_prior_post_resist, self.axes_prior_post_resist = self.create_node_figure(f'Moment, Shear, Deflection UDL resistance, load and utilization ratio for Beam {beam}', 6)
        self.fill_node_figureResist(self.net, self.axes_prior_post_resist, color = 'tab:orange')
        self.fig_prior_post_resist.savefig(f"Beam_num_{beam}_resist.png",  bbox_inches='tight')
    
    def convert_csv_to_string(self, file_path, x):
        x_values = []
        y_values = []
        
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x_values.append(row[x])
                y_values.append(row["priors"])
            # Combine x_values and y_values into a single list
        combined_values = x_values + y_values
        x_values=[float(i) for i in x_values]

        return ",".join(combined_values), min(x_values), max(x_values)
        
    def convert_csv_to_strings(self, x, y, val):
        x_values = []
        y_values = []
        file_path = f"{x}_given_{y}.csv"
        y = f"{y}_{val}"
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x_values.append(row[x])
                y_values.append(row[y])
            # Combine x_values and y_values into a single list
        combined_values = x_values + y_values
        x_values=[float(i) for i in x_values]

        return ",".join(combined_values)#, min(x_values), max(x_values)
    
    def define_network(self):
        #define the decade
        self.decade = self.create_equation_node(self.net, "decade", "Decade",
                                "decade = CustomPDF(1970, 1980, 1990, 2000, 2010, 0.2326, 0.3099, 0.1542, 0.1479, 0.1553)", 1970, 2010, 0, 0)

        self.building_length = self.create_equation_node(self.net, "building_length", "Building Length",
                                #"building_length = CustomPDF(13,14,15,16,17,18,19,20, 0.1478, 0.1492, 0.1535, 0.149, 0.127, 0.109, 0.091, 0.0730)", 12.9, 20.1, 0, 0)
                                "building_length = CustomPDF(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 0.0001, 0.0001, 0.0014, 0.0102, 0.0868, 0.1554, 0.1588, 0.1617, 0.1263, 0.0876, 0.0629, 0.0431, 0.0285, 0.0179, 0.0126, 0.0111, 0.0088, 0.0079, 0.0076, 0.0052, 0.0061)", 1.9, 42.1, 0, 0)
        

        
        
        self.building_width = self.create_equation_node(self.net, "building_width", "Building Width",
                                #"building_width = CustomPDF(6, 6.5, 7,7.5, 8, 8.5, 9, 9.5, 10,10.5,0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1)", 5.9, 10.6, 0, 0)
                                #"building_width = CustomPDF(6,7,8,9,0.223, 0.294, 0.3, 0.184)", 5.9, 9.1, 0, 0)
                                "building_width = CustomPDF(4, 5, 6, 7, 8, 9, 10, 11,0.04, 0.122, 0.165, 0.218, 0.222, 0.136, 0.063, 0.033)", 3.9, 11.1, 0, 0)
              
        self.roof_angle = self.create_equation_node(self.net, "roof_angle", "Roof Angle",
                                "roof_angle = CustomPDF(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 0.2, 0.08, 0.12, 0.2, 0.16, 0.08, 0.06, 0.04, 0.04, 0.02)", 0, 45.1, 0, 0)
        
        # Define the beam properties
        self.beam_section = self.create_equation_node(self.net, "beam_section", "Beam Section",
                                "beam_section = CustomPDF(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04)", 0.9, 25.1, 0, 0)
       
        self.beam_area = self.create_equation_node(self.net, "beam_area", "Beam Area", 
                                "beam_area = if(beam_section < 25, if(beam_section < 24, if(beam_section < 23, if(beam_section < 22, if(beam_section < 21, if(beam_section < 20, if(beam_section < 19, if(beam_section < 18, if(beam_section < 17, if(beam_section < 16, if(beam_section < 15, if(beam_section < 14, if(beam_section < 13, if(beam_section < 12, if(beam_section < 11, if(beam_section < 10, if(beam_section < 9, if(beam_section < 8, if(beam_section < 7, if(beam_section < 6, if(beam_section < 5, if(beam_section < 4, if(beam_section < 3, if(beam_section < 2, 3.528, 4.428), 5.328), 7.128), 4.704), 5.904), 7.104), 8.304), 9.504), 10.704), 5.978), 7.503), 9.028), 10.553), 12.078), 13.603), 7.154), 8.979), 10.804), 12.629), 14.454), 16.279), 9.604), 19.404), 21.854)", 3.528, 21.855, 0, 0)
                                
        self.section_mod = self.create_equation_node(self.net, "section_mod", "Section Modulus",
                                "section_mod = if(beam_section < 25, if(beam_section < 24, if(beam_section < 23, if(beam_section < 22, if(beam_section < 21, if(beam_section < 20, if(beam_section < 19, if(beam_section < 18, if(beam_section < 17, if(beam_section < 16, if(beam_section < 15, if(beam_section < 14, if(beam_section < 13, if(beam_section < 12, if(beam_section < 11, if(beam_section < 10, if(beam_section < 9, if(beam_section < 8, if(beam_section < 7, if(beam_section < 6, if(beam_section < 5, if(beam_section < 4, if(beam_section < 3, if(beam_section < 2, 57.624, 90.774), 131.424), 235.224), 76.832), 121.032), 175.232), 239.432), 313.632), 397.832), 97.641), 153.812), 222.691), 304.278), 398.574), 505.578), 116.849), 184.070), 266.499), 364.136), 476.982), 605.036), 156.865), 640.332), 812.240)", 57.624, 812.241, 0, 0)
    
        self.second_moment_area = self.create_equation_node(self.net, "second_moment_area", "Second Moment Area",
                                "second_moment_area = if(beam_section < 25, if(beam_section < 24, if(beam_section < 23, if(beam_section < 22, if(beam_section < 21, if(beam_section < 20, if(beam_section < 19, if(beam_section < 18, if(beam_section < 17, if(beam_section < 16, if(beam_section < 15, if(beam_section < 14, if(beam_section < 13, if(beam_section < 12, if(beam_section < 11, if(beam_section < 10, if(beam_section < 9, if(beam_section < 8, if(beam_section < 7, if(beam_section < 6, if(beam_section < 5, if(beam_section < 4, if(beam_section < 3, if(beam_section < 2, 2.823576, 5.582601), 9.725376), 23.287176), 3.764768), 7.443468), 12.967168), 20.710868), 31.049568), 44.358268), 4.784392667), 9.45940725), 16.47910933), 26.32006142), 39.458826), 56.37196558), 5.725584667), 11.32027425), 19.72090133), 31.49777842), 47.221218), 67.46153258), 7.686401333), 63.392868), 90.56479717)", 2.823576, 90.56479718, 0, 0)
        

        
        
        # Creating the nodes for the beam section properties

        self.length = self.create_equation_node(self.net, "length", "Beam length",
                                "length = if(building_width < 7.1, building_width/2,  building_width/3)", 1, 5, 0, 0)
                                #"length = if(building_width < 7.1, building_width/2,  building_width/4)", 1, 6, 0, 0)
        
        self.beam_spacing = self.create_equation_node(self.net, "beam_spacing", "Beam spacing",
                                #"beam_spacing = CustomPDF(300, 350, 400, 450, 500, 550, 600, 0.05, 0.05, 0.05, 0.1, 0.2, 0.25, 0.3)", 299, 601, 0, 0)
                                "beam_spacing = CustomPDF(300, 350, 400, 450, 500, 550, 600, 0.05, 0.05, 0.05, 0.1, 0.2, 0.25, 0.3)", 299, 601, 0, 0)
        
     
        # Calculate the number of beams in the roof
        self.num_beams_x = self.create_equation_node(self.net, "num_beams_x", "Number of Beams in X direction",
                                "num_beams_x = building_length/length", 0, 1500, 0, 0)
        self.roof_slope_length = self.create_equation_node(self.net, "roof_slope_length", "Roof Slope Length",
                                "roof_slope_length = building_width/(2*cos(pi()/180*roof_angle))", 0, 1500, 0, 0)
        self.num_beams_y = self.create_equation_node(self.net, "num_beams_y", "Number of Beams in Y direction",
                                "num_beams_y = roof_slope_length/(beam_spacing/1000)", 0, 3000, 0, 0)
        self.num_beams = self.create_equation_node(self.net, "num_beams", "Number of Beams",
                                "num_beams = if(building_width < 7.1, num_beams_x*num_beams_y*2, num_beams_x*num_beams_y*3)", 0, 3000, 0, 0)
        
        # Define kmod values
        self.k_uls = self.create_equation_node(self.net, "k_uls", "kmod ULS",
                                "k_uls = if(decade < 2004, if(decade < 1990, if( decade < 1980, 1.0, 1.0/1.2), 0.9), 0.8)", 0.79, 1.1, 0, 0)
        self.k_sls = self.create_equation_node(self.net, "k_sls", "kmod SLS",
                                "k_sls = if(decade < 2004, if(decade < 1990, if( decade < 1980, 1.0, 1.3), 1.3), 1.0)", 0.9, 2.6, 0, 0)
        self.k_b = self.create_equation_node(self.net, "k_b", "kmod B",
                                "k_b = if(decade < 2004, 1,0.67)", 0.66, 1.01, 0, 0)  
        
        # Define the beam properties
        self.char_bending_str = self.create_equation_node(self.net, "char_bending_str", "Charactristic bending strength",
                                #"char_bending_str = CustomPDF(18, 24, 30, 40,  0.1,  0.7,  0.15,  0.05)",17 , 41, 0, 0)
                                "char_bending_str = CustomPDF(18, 24, 30, 40,  0.1,  0.7,  0.15,  0.05)",17 , 41, 0, 0)
        self.young_mod = self.create_equation_node(self.net, "young_mod", "Young's modulus",
                                "young_mod = if(char_bending_str < 40, if(char_bending_str < 31, if(char_bending_str < 25, if(char_bending_str < 19, 9, 11), 12), 14), 14)", 8.9, 14.1, 0, 0)
        self.density = self.create_equation_node(self.net, "density", "Density",
                                "density = if(char_bending_str < 40, if(char_bending_str < 31, if(char_bending_str < 25, if(char_bending_str < 19, 380, 420), 460), 500), 500)", 379, 501, 0, 0)
        self.char_shear_str = self.create_equation_node(self.net, "char_shear_str", "Characteristic shear strength",         
                                "char_shear_str = if(decade > 1970, if(decade > 1980, if(decade > 1990, min(3.8, 0.2 * (char_bending_str)^0.8), 0.2 * (char_bending_str)^0.8), 0.2 * (char_bending_str)^0.8), 2)",1, 4, 0, 0)

        self.material_psf = self.create_equation_node(self.net, "material_psf", "Material partial safety factor",
                                "material_psf = if(decade < 2004, if(decade < 1990, 1.213, 1.2075), 1.25)", 1.0, 1.5, 0, 0)
                                
        self.des_bending_str = self.create_equation_node(self.net, "des_bending_str", "Design bending strength",
                                "des_bending_str = char_bending_str/material_psf*k_uls", 10, 40, 0, 0)
        
        self.des_shear_str = self.create_equation_node(self.net, "des_shear_str", "Design shear strength",
                                "des_shear_str = char_shear_str/material_psf*k_uls*k_b", 0.95, 4.2, 0, 0)
        
        #calculate the resistance of the beam
        self.mom_resist = self.create_equation_node(self.net, "mom_resist", "Moment Resistance udl",
                                "mom_resist = 8*section_mod*1000*des_bending_str/(length*length*1000000)", 0, 70, 0, 0)
        
        self.shear_resist = self.create_equation_node(self.net, "shear_resist", "Shear Resistance udl",
                                "shear_resist = 2*beam_area*1000*des_shear_str/(1.5*length*1000)", 0, 55, 0, 0)
        
        self.defl_resist = self.create_equation_node(self.net, "defl_resist", "Deflection Resistance udl",
                                "defl_resist = 384*young_mod*1000*second_moment_area*1000000/(1500*length*length*length*1000000000)*k_sls", 0, 25, 0, 0)
        self.beam_resist = self.create_equation_node(self.net, "beam_resist", "Beam Resistance",
                                "beam_resist = min(mom_resist, shear_resist, defl_resist)", 0, 30, 0, 0)
        
        #calculate the load on the beam
        self.perm_psf = self.create_equation_node(self.net, "perm_psf", "Permanent partial safety factor",
                                "perm_psf = 1.2", 1.2, 1.2, 0, 0)
        self.live_psf = self.create_equation_node(self.net, "live_psf", "Live partial safety factor",
                                "live_psf = if(decade >= 2004, 1.5, 1.6)", 1.49, 1.61, 0, 0)
        self.snow_load = self.create_equation_node(self.net, "snow_load", "Snow Load",
                                "snow_load = if(decade < 2004, if(decade < 1990, if( decade < 1980, 2.5, 2.5*(60 - roof_angle)/30), 3.5 * 1.2 * (60 - roof_angle) / 30), 3.5 * 0.8 * (60 - roof_angle) / 30)", 0, 10, 0, 0)
        self.perm_load = self.create_equation_node(self.net, "perm_load", "Permanent Load",
                                "perm_load = 0.5", 0.5, 0.5, 0, 0)
        self.des_live_load = self.create_equation_node(self.net, "des_live_load", "Design Live Load",
                                "des_live_load = snow_load*live_psf", 0, 14, 0, 0)   
        self.des_perm_load = self.create_equation_node(self.net, "des_perm_load", "Design Permanent Load",
                                "des_perm_load = perm_load*perm_psf", 0, 10, 0, 0)
        
        self.beam_load = self.create_equation_node(self.net, "beam_load", "Beam Load",
                                "beam_load = beam_spacing/1000*(des_live_load+des_perm_load)", 0, 9, 0, 0)
        
        #Calculate the load resistance ratio
        self.load_resist_ratio = self.create_equation_node(self.net, "load_resist_ratio", "Load Resistance Ratio",
                                "load_resist_ratio = beam_load/beam_resist", 0, 5.0, 0, 0) # the second number is the upper bound of the range and must equal ten when divided by nint

        # Calculate the volume of the beam only if the load resistance ratio is less than 1
        self.beam_volume = self.create_equation_node(self.net, "beam_volume", "Beam Volume",   
                                "beam_volume = if(load_resist_ratio  < 1, beam_area*1000*length/1000000, 0)", 0, 20, 0, 0)

        # Calculate the volume of all beams in the roof
        self.beam_volume_in_roof = self.create_equation_node(self.net, "beam_volume_in_roof", "Beam Volume in Roof",
                                "beam_volume_in_roof = beam_volume*num_beams", 0, 18, 0, 0)    #m^3

        # Calculate the mass of all the beams in the roof
        self.beam_mass_in_roof = self.create_equation_node(self.net, "beam_mass_in_roof", "Beam Mass in Roof",
                                "beam_mass_in_roof = beam_volume_in_roof*density/1000", 0, 30, 0, 0) #tons
    
    def create_node_figure(self, title, nnodes):
        fig, axes = plt.subplots(int(nnodes/2),2, figsize = (6,nnodes+2))
        fig.tight_layout()#h_pad=4, w_pad=1)
        fig.suptitle(title, fontsize=16, y =1.05)
        fig.subplots_adjust(hspace=0.75)
        return fig, axes

    def fill_node_figureResist(self,net,axes,color):
        colors = {'critical':'tab:orange'}  
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[2,0].legend(handles, labels)
        
        colors = {'prior':'tab:grey', 'posterior':'tab:blue'}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[0,0].legend(handles, labels)
        
        self.create_node_sub_figure(net, axes[0,0], 1, self.mom_resist,11, '$w_{m,Rd}$ [kN/m]', color = color)        
        self.create_node_sub_figure(net, axes[0,1], 2, self.shear_resist, 15, '$w_{v,Rd}$ [kN/m]', color = color)  
        self.create_node_sub_figure(net, axes[1,0], 2, self.defl_resist, 15, '$w_{d,Rd}$ [kN/m]', color = color)
        self.create_node_sub_figure(net, axes[1,1], 2, self.beam_resist, 15, '$w_{Rd}$ [kN/m]', color = color) #mm$^{2}$/year
        self.create_node_sub_figure(net, axes[2,0], 1, self.beam_load,11, '$w_{Ed}$ [kN/m]', color = color)        
        self.create_node_sub_figure(net, axes[2,1], 3, self.load_resist_ratio,14, '$u$ [-]', color = color)

    def fill_node_figureBeamProps(self,net,axes,color):
        colors = {'critical':'tab:orange'}  
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[2,0].legend(handles, labels)
        
        colors = {'prior':'tab:grey', 'posterior':'tab:blue'}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[0,0].legend(handles, labels)
        
        self.create_node_sub_figure(net, axes[0,0], 1, self.beam_section,17, '$Beam section$ [-]', color = color)        
        self.create_node_sub_figure(net, axes[0,1], 2, self.beam_area, 14, '$Beam area$ [x$10^{3}$ mm$^{2}$]', color = color) #mm$^{2}$/year
        self.create_node_sub_figure(net, axes[1,0], 3, self.section_mod, 14, '$Section modulus$ [x$10^{3}$ mm$^{3}$]', color = color)
        self.create_node_sub_figure(net, axes[1,1], 4, self.second_moment_area, 14, '$Second moment of area$ [x$10^{6}$ mm$^{4}$]', color = color)
        self.create_node_sub_figure(net, axes[2,1], 5, self.building_length,14, '$Building length$ [m]', color = color)
        self.create_node_sub_figure(net, axes[2,0], 6, self.building_width,11, '$Building width$ [m]', color = color) 

    def fill_node_figureOutput(self,net,axes,color):
        colors = {'critical':'tab:orange'}  
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[2,0].legend(handles, labels)
        
        colors = {'prior':'tab:grey', 'posterior':'tab:blue'}         
        labels = list(colors.keys())
        handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha = 0.5) for label in labels]
        #axes[0,0].legend(handles, labels)

        self.create_node_sub_figure(net, axes[0,0], 1, self.char_bending_str,4, '$f_{m,k}$ [N/mm$^{2}$]', color = color)
        self.create_node_sub_figure(net, axes[0,1], 2, self.young_mod, 6, '$E$ [x$10^{3}$ N/mm$^{2}$]', color = color)
        self.create_node_sub_figure(net, axes[1,0], 3, self.density, 8, '$Density$ [kg/m$^{3}$]', color = color)              
        self.create_node_sub_figure(net, axes[1,1], 4, self.des_live_load,17, '$Q_{d}$ [kN/m$^{2}$]', color = color)
        self.create_node_sub_figure(net, axes[2,1], 5, self.beam_spacing, 7, '$Spacing_{beam} [mm]$', color = color)
        self.create_node_sub_figure(net, axes[2,0], 6, self.length,11, '$Beam length$ [m]', color = color)
        #self.create_node_sub_figure(net, axes[2,1], 6, self.beam_volume_in_roof,11, '$V_{purlins}$ [m$^{3}$]',color=color, alpha=0.5)

    def create_node_sub_figure(self,net,axis,subp,name,nint,title,labelstring=False,color='tab:grey',alpha=0.5):
        # If not already discretized, discretize them manually.
        #,ymax=0.6
        if net.is_value_discretized(name):
            bounds = net.get_node_equation_bounds(name)
            disc_beliefs = net.get_node_value(name)
            nint = len(disc_beliefs)
            boundvec = np.linspace(bounds[0], bounds[1], nint+1)
        else:
            bounds = net.get_node_equation_bounds(name)
            boundvec = np.linspace(bounds[0], bounds[1], nint+1)
            sample = np.asarray(net.get_node_value(name))
            disc_beliefs = []
            for i in range(nint):
                p = ((boundvec[i] < sample) & (sample < boundvec[i+1])).sum()/len(sample)
                disc_beliefs.append(p)    
        
        interval=[]
        
        for j in range(nint):
            s1 = '{:.1f}'.format(boundvec[j])
            s2 = '{:.1f}'.format(boundvec[j+1])
            interval.append(s1+'-'+ s2)

        labels = interval
        
        if labelstring == False:
            labels = interval
        else:
            labels = labelstring
        
        axis.bar(np.arange(nint), height = disc_beliefs, color=color,alpha=alpha)
        axis.set_xticks(np.arange(nint))
        axis.set_xticklabels(labels = labels, rotation='vertical', fontsize = 8)
        axis.set_title(title)
        # Add grid lines for better readability
        #axis.grid(True, linestyle='--', alpha=0.7)
        #axis.set_ylabel('Probability')

    

    def create_node_output(self,net,name,nint):
        # If not already discretized, discretize them manually.
        #,ymax=0.6
        if net.is_value_discretized(name):
            bounds = net.get_node_equation_bounds(name)
            disc_beliefs = net.get_node_value(name)
            nint = len(disc_beliefs)
            boundvec = np.linspace(bounds[0], bounds[1], nint+1)
        else:
            bounds = net.get_node_equation_bounds(name)
            boundvec = np.linspace(bounds[0], bounds[1], nint+1)
            sample = np.asarray(net.get_node_value(name))
            disc_beliefs = []
            for i in range(nint):
                p = ((boundvec[i] < sample) & (sample < boundvec[i+1])).sum()/len(sample)
                disc_beliefs.append(p)    
        
        interval=[]
        
        for j in range(nint):
            s1 = '{:.2f}'.format(boundvec[j])
            s2 = '{:.2f}'.format(boundvec[j+1])
            interval.append(s1+'-'+ s2)
        
        return disc_beliefs, interval
    
    def update_and_show_node_stats(self, net, node_id):
        net.update_beliefs()
        node_handle = self.net.get_node(node_id)
        self.show_node_stats(self.net, node_handle)

    def set_uniform_intervals(self, net, node_handle, count):
        bounds = net.get_node_equation_bounds(node_handle)
        lo = bounds[0]
        hi = bounds[1]

        iv = [None] * count
        for i in range(0, count):
            iv[i] = pysmile.DiscretizationInterval("", lo + (i + 1) * (hi - lo) / count)
        net.set_node_equation_discretization(node_handle, iv)  
    
    def create_equation_node(self, net, id, name, equation, lo_bound, hi_bound, x_pos, y_pos):
        handle = net.add_node(pysmile.NodeType.EQUATION, id)
        net.set_node_name(handle, name)
        net.set_node_equation(handle, equation)
        net.set_node_equation_bounds(handle, lo_bound, hi_bound)
        net.set_node_position(handle, x_pos, y_pos, 85, 55)
        return handle 
    
    def show_stats(self, net, node_handle):
        node_id = net.get_node_id(node_handle)

        if net.is_evidence(node_handle):
            v = net.get_cont_evidence(node_handle)
            #print(f"{node_id} has evidence set: {v}")
            return
        
        if net.is_value_discretized(node_handle):
            #print(f"{node_id} is discretized.")
            iv = net.get_node_equation_discretization(node_handle)
            bounds = net.get_node_equation_bounds(node_handle)
            disc_beliefs = net.get_node_value(node_handle)
            lo = bounds[0]
            for i in range(0, len(disc_beliefs)):
                hi = iv[i].boundary
                print(f"\tP({node_id} in {lo}..{hi})={disc_beliefs[i]}")
                lo = hi
        else:
            stats = net.get_node_sample_stats(node_handle)
            #print(f"{node_id}: mean={round(stats[0], 3)} stddev={round(stats[1], 3)} min={round(stats[2], 3)} max={round(stats[3], 3)}")
    
    def show_node_stats(self, net, node_handle):
        node_id = net.get_node_id(node_handle)
        stats = net.get_node_sample_stats(node_handle)
        return stats[0], stats[1], stats[2], stats[3]


    def update_and_show_stats(self, net):
        net.update_beliefs()
        for h in net.get_all_nodes():
            self.show_stats(net, h)
        print()
    
    def create_cpt_node(self, net, id, name, outcomes, x_pos, y_pos):
        handle = net.add_node(pysmile.NodeType.CPT, id)
        net.set_node_name(handle, name)
        net.set_node_position(handle, x_pos, y_pos, 85, 55)
        initial_outcome_count = net.get_outcome_count(handle)
        for i in range(0, initial_outcome_count):
            net.set_outcome_id(handle, i, outcomes[i])
        for i in range(initial_outcome_count, len(outcomes)):
            net.add_outcome(handle, outcomes[i])
        return handle



def create_bar_chart(output_LRR, output_LRR_prob):
    """
    Creates a bar chart for one beam using output_LRR and output_LRR_prob.
    
    Parameters:
    output_LRR (list): A list of x-axis values for the beam.
    output_LRR_prob (list): A list of y-axis values for the beam.
    """
    plt.figure(figsize=(10, 6))
    truncated_x = output_LRR[:10]
    truncated_y = output_LRR_prob[:10]
    plt.bar(truncated_x, truncated_y, width=0.3, color='green', alpha=0.7)
    #plt.bar(output_LRR, output_LRR_prob, width=0.4, color='blue', alpha=0.5)
    plt.xlabel('Load resistance ratio')
    plt.ylabel('Probability')
    plt.tick_params(axis='x', rotation=90)
    plt.title('Bar Chart of load resistance ratio vs probability')
    plt.show()


def plot_lognormal_distribution():
        mu = -2
        sigma = 0.75

        # create a range of x values
        x = np.linspace(0, 1, 1000)

        # calculate the pdf
        pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))

        # plot the pdf
        plt.plot(x, pdf)
        plt.title('Lognormal Distribution for 1 - utilization')
        plt.xlabel('1 - utilization')
        plt.ylabel('pdf')
        plt.grid(True)
        #plt.show() 

def generate_lognormal_data(mu, sigma):
    import scipy.stats as stats

    """
    Generates a list with 10 values between 0 and 1 for the x-axis intervals and the corresponding y-axis (probabilities) between 0 and 1.
    
    Parameters:
    mu (float): The mean of the underlying normal distribution.
    sigma (float): The standard deviation of the underlying normal distribution.
    
    Returns:
    x_intervals (list): A list of 10 interval strings for the x-axis.
    y_values (list): A list of corresponding normalized probabilities between 0 and 1 for the y-axis.
    """
    # Define the x-axis intervals
    x_intervals = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    
    # Calculate the midpoints of the intervals
    x_midpoints = np.linspace(0.05, 0.95, 10)
    
    # Calculate the corresponding y-values (probabilities) using the PDF of the lognormal distribution
    s = sigma
    scale = np.exp(mu)
    y_values = stats.lognorm.pdf(x_midpoints, s=s, scale=scale)
    
    # Normalize the y-values by dividing each y-value by the total sum of y-values
    total_sum = np.sum(y_values)
    y_values_normalized = y_values / total_sum
    
    # Reverse the order of the y-values as specified
    y_values_reversed = [y_values_normalized[-1]] + y_values_normalized[-2:0:-1].tolist() + [y_values_normalized[0]]
    
    # Convert to lists
    x_intervals = x_intervals[:10]  # Ensure we have 10 intervals
    y_values_reversed = y_values_reversed[:10]
    
    return x_intervals, y_values_reversed

def generate_lognormal_data_interval(mu, sigma, n_intervals):
    import scipy.stats as stats

    """
    Generates a list with n_intervals values between 0 and 1 for the x-axis intervals and the corresponding y-axis (probabilities) between 0 and 1.
    
    Parameters:
    mu (float): The mean of the underlying normal distribution.
    sigma (float): The standard deviation of the underlying normal distribution.
    
    Returns:
    x_intervals (list): A list of n_intervals interval strings for the x-axis.
    y_values (list): A list of corresponding normalized probabilities between 0 and 1 for the y-axis.
    """
    # Define the x-axis intervals
    x_intervals = [f'{i/n_intervals:.2f}-{(i+1)/n_intervals:.2f}' for i in range(n_intervals)]
    
    # Calculate the midpoints of the intervals
    x_midpoints = np.linspace(0.025, 0.975, n_intervals)
    
    # Calculate the corresponding y-values (probabilities) using the PDF of the lognormal distribution
    s = sigma
    scale = np.exp(mu)
    y_values = stats.lognorm.pdf(x_midpoints, s=s, scale=scale)
    
    # Normalize the y-values by dividing each y-value by the total sum of y-values
    total_sum = np.sum(y_values)
    y_values_normalized = y_values / total_sum
    
    # Reverse the order of the y-values as specified
    y_values_reversed = [y_values_normalized[-1]] + y_values_normalized[-2:0:-1].tolist() + [y_values_normalized[0]]
    
    # Convert to lists
    x_intervals = x_intervals[:n_intervals]  # Ensure we have 20 intervals
    y_values_reversed = y_values_reversed[:n_intervals]
    
    return x_intervals, y_values_reversed

import re
def create_one_bar_chart(x_values, y_values, chart_title, x_axis_title, y_axis_title, color):
    import matplotlib.pyplot as plt
    """
    Creates a single bar chart.
    
    Parameters:
    x_values (list): A list of x-axis values.
    y_values (list): A list of y-axis values.
    chart_title (str): The title of the chart.
    x_axis_title (str): The title of the x-axis.
    y_axis_title (str): The title of the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, width=0.3, color=color, edgecolor='black',alpha=0.7)
    plt.xlabel(x_axis_title, fontsize=12)
    plt.ylabel(y_axis_title, fontsize=12)
    plt.tick_params(axis='x', rotation=90, labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    #plt.title(chart_title)

    # Save the figure based on the name of the chart title
    sanitized_title = re.sub(r'[^\w\s]', '', chart_title).replace(" ", "_")
    filename = sanitized_title + ".png"
    plt.tight_layout()
    plt.savefig(filename, dpi=720)
    
    #plt.show()

def create_one_bar_chart_with_mean(x_values, y_values, chart_title, x_axis_title, y_axis_title, color):
    import matplotlib.pyplot as plt
    """
    Creates a single bar chart.
    
    Parameters:
    x_values (list): A list of x-axis values.
    y_values (list): A list of y-axis values.
    chart_title (str): The title of the chart.
    x_axis_title (str): The title of the x-axis.
    y_axis_title (str): The title of the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, width=0.3, color=color, alpha=0.7)
    plt.xlabel(x_axis_title, fontsize=12)
    plt.ylabel(y_axis_title, fontsize=12)
    plt.tick_params(axis='x', rotation=90, labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title(chart_title)
   
     # Calculate and highlight the mean value
    mean_value = sum(x * y for x, y in zip(range(len(x_values)), y_values)) / sum(y_values)
    plt.axvline(x=mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    #plt.legend()

    # Save the figure based on the name of the chart title
    sanitized_title = re.sub(r'[^\w\s]', '', chart_title).replace(" ", "_")
    filename = sanitized_title + ".png"
    plt.tight_layout()
    plt.savefig(filename, dpi=720)
    
    #plt.show()


def create_5x5_bar_charts(output_LRR_list, output_LRR_prob_list, beam_titles, chart_title, y_max):
    """
    Creates a 5x5 plot with each subplot being a bar chart showing results of a single beam.
    
    Parameters:
    output_LRR_list (list of lists): A list containing 25 lists of x-axis values for each beam.
    output_LRR_prob_list (list of lists): A list containing 25 lists of y-axis values for each beam.
    """
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle(chart_title, fontsize=16)
    
    #y_max = 0.5
    for i in range(5):
        for j in range(5):
            index = i * 5 + j
            ax = axes[i, j]
            truncated_x = output_LRR_list[index]#[:10]
            truncated_y = output_LRR_prob_list[index]#[:10]
            ax.bar(truncated_x, truncated_y, width=0.5, color='red', alpha=0.7)
            #ax.bar(output_LRR_list[index], output_LRR_prob_list[index], width=0.4, color='blue', alpha=0.7)
            ax.set_title(f'Beam {beam_titles[index]}')
            #ax.set_title(beam_titles[index])
            ax.set_xlabel('load resistance ratio (LRR)')
            ax.set_ylabel('LRR probability')
            ax.tick_params(axis='x', rotation=90)
            # Set the y-axis to a logarithmic scale
            #ax.set_yscale('log')
            
            # Set the y-axis limit based on the maximum value in the y-values
            #y_max = max(truncated_y)
            ax.set_ylim(0, y_max)  # Adjust the lower limit as needed
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    fig.savefig(chart_title + ".png", dpi=720)
    #plt.show()

def subtract_lists(list1, list2):
    """
    Subtracts corresponding values of one list from another.
    
    Parameters:
    list1 (list): The first list of values.
    list2 (list): The second list of values.
    
    Returns:
    result (list): A list of the differences between corresponding values of list1 and list2.
    """
    return [abs(a - b) for a, b in zip(list1, list2)]

# define a normalization function
def normalize_each_value_in_list(input_list):
    """
    Normalizes each value in a list by dividing it by the sum of all values in the list.
    
    Parameters:
    input_list (list): A list of values to be normalized.
    
    Returns:
    normalized_list (list): A list of normalized values.
    """
    sum_values = sum(input_list)
    return [value / sum_values for value in input_list]


#beam_num = 21
nint = 100 # number of intervals for the LRR, this number must be divisible by 10 when divided by 
n_intervals = 20
beam_titles = [
    "36x98", "36x123", "36x148", "36x198", "48x98", "48x123", "48x148", "48x173", "48x198", "48x223",
    "61x98", "61x123", "61x148", "61x173", "61x198", "61x223", "73x98", "73x123", "73x148", "73x173",
    "73x198", "73x223", "98x98", "98x198", "98x223"]

# Create the lognormal distribution add intervals and plot the results
log_x_intervals, log_y_values = generate_lognormal_data_interval(mu=-2, sigma=0.75, n_intervals=n_intervals)
#print("X intervals:", log_x_intervals)
#print("Y values:", log_y_values)

#create_one_bar_chart(log_x_intervals, log_y_values,"Target LRR",  "LRR", "LRR Probability", "red")

def likely_beam_BN(n_intervals):

    # 1. Find the LRR for each of the beams using the Bayesian network and plot the results

    load_resist_ratios = []
    load_resist_ratios_probs = []
    load_resist_ratios_probs_cumsum = []
    max_LRR_index = -1
    max_LRR_beam_num = -1
    for beam_num in range(1, 26):
        beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
        output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first ten for each beam
        output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first ten for each beam

        load_resist_ratios.append(output_LRR)
        load_resist_ratios_probs.append(output_LRR_prob)
        load_resist_ratios_probs_cumsum.append(sum(output_LRR_prob))
        print(f"Beam {beam_num} LRR probabilities for each interval: {output_LRR_prob}")
        
        # When all evidence has been entered, the LRR_prob is equal to 1. The beam with the LRR closest to 1 is the most likely beam. Find the highest index of 1 in output_LRR_prob
        if 1 in output_LRR_prob:
            index_of_one = len(output_LRR_prob) - 1 - output_LRR_prob[::-1].index(1)
            if index_of_one > max_LRR_index:
                max_LRR_index = index_of_one
                max_LRR_beam_num = beam_num
            
            #print(f"Beam {beam_num} LRR: {output_LRR_prob}")

    #print(f"The beam with the 1 closest to the end of the list is Beam {max_LRR_beam_num} with index {max_LRR_index}")
    print(f"The list for the cumulative sum of all probabilities between 0 and 1 for all beams is {load_resist_ratios_probs_cumsum}") #should sum to 1 or less

    # 2. Create a normalised version of the LRR probabilities and print the results

    load_resist_ratios_probs_normalized = []
    for i in range(25):
        sublist = []
        for j in range(n_intervals):
            if load_resist_ratios_probs_cumsum[i] == 0:
                LRR_probs_normalized_value = 0
            else:
                LRR_probs_normalized_value = load_resist_ratios_probs[i][j] / load_resist_ratios_probs_cumsum[i]
            sublist.append(LRR_probs_normalized_value)
        load_resist_ratios_probs_normalized.append(sublist)
        print(f"Beam {i+1} LRR normalized: {load_resist_ratios_probs_normalized[i]}")


    # 3. Subtract the LRR probabilities normalized from the target LRR probabilities for each beam to get the error

    LRR_probs_minus_target = []
    LRR_probs_minus_target_cumsum = []

    for i in range(25):
        print(f"Beam {i+1} LRR normalized probabilities: {load_resist_ratios_probs_normalized[i]}")
        print(f"Beam {i+1} LRR target probabilities: {log_y_values}")
        LRR_probs_minus_target.append(subtract_lists(load_resist_ratios_probs_normalized[i], log_y_values))
        if sum(LRR_probs_minus_target[i]) == 1:
            LRR_probs_minus_target_cumsum.append(0)
        else:
            LRR_probs_minus_target_cumsum.append(sum(LRR_probs_minus_target[i]))
        print(f"Beam {i+1} LRR normalized - target probabilities: {LRR_probs_minus_target[i]}")



    # 4. Find the most likely beam by first summing LRR_minus_target for each beam

    # The most likely beam is the one with the smallest value of LRR_probs_minus_target divided by the sum of all LRR_probs_minus_target

    LRR_least_error = []
    load_resist_ratios_probs_cumsum_all_beams = sum(LRR_probs_minus_target_cumsum)
    #print(f"The sum of all probabilities is {load_resist_ratios_probs_cumsum_all_beams}")

    for i in range(25):
        LRR_least_error.append(LRR_probs_minus_target_cumsum[i] / load_resist_ratios_probs_cumsum_all_beams)
        print(f"Beam {i+1} LRR error: {LRR_least_error[i]}")

    # Find the index of the beam with the least error which is greater than zero and print the beam number
    min_LRR_error = min(i for i in LRR_least_error if i > 0)
    min_LRR_index = LRR_least_error.index(min_LRR_error)
    min_LRR_beam = beam_titles[min_LRR_index]
    print(f"The beam with the least error is Beam {min_LRR_beam} with error {round(min_LRR_error, 3)}")

    # sum the errors and print
    LRR_least_error_sum = sum(LRR_least_error)
    print(f"The sum of all LRR_probs errors is {LRR_least_error_sum}")

def alternative_likely_beam(n_intervals, evidence):
# ALTERNATIVE METHOD: Find the most likely beam by using expected value of the LRR for each beam
    # 1. Find the LRR for each of the beams using the Bayesian network and plot the results

    load_resist_ratios = []
    load_resist_ratios_probs = []
    load_resist_ratios_probs_cumsum = []

    for beam_num in range(1, 26):
        beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
        output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first ten for each beam
        output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first ten for each beam

        load_resist_ratios.append(output_LRR)
        load_resist_ratios_probs.append(output_LRR_prob)
        load_resist_ratios_probs_cumsum.append(sum(output_LRR_prob))
        print(f"Beam {beam_num} LRR probabilities for each interval: {output_LRR_prob}")
        
    #print(f"The beam with the 1 closest to the end of the list is Beam {max_LRR_beam_num} with index {max_LRR_index}")
    #print(f"The list for the cumulative sum of all probabilities between 0 and 1 for all beams is {load_resist_ratios_probs_cumsum}") #should sum to 1 or less
    #create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs, beam_titles, "25 Beam types & LRR probabilities"+evidence)
    # 2. Create a normalised version of the LRR probabilities and print the results

    load_resist_ratios_probs_normalized = []
    for i in range(25):
        sublist = []
        for j in range(n_intervals):
            if load_resist_ratios_probs_cumsum[i] == 0:
                LRR_probs_normalized_value = 0
            else:
                LRR_probs_normalized_value = load_resist_ratios_probs[i][j] / load_resist_ratios_probs_cumsum[i]
            sublist.append(LRR_probs_normalized_value)
        load_resist_ratios_probs_normalized.append(sublist)
        #print(f"Beam {i+1} LRR normalized: {load_resist_ratios_probs_normalized[i]}")
    
    # 3B. Find expected values of each beam using LRR 'expected' probabilities and print the results

    load_resist_ratios_probs_expected = []
    load_resist_ratios_probs_expected_cumsum = []
    for i in range(25):
        sublist = []
        for j in range(n_intervals):
            if load_resist_ratios_probs_cumsum[i] == 0:
                LRR_probs_expected_value = 0
            else:
                LRR_probs_expected_value = load_resist_ratios_probs[i][j]*log_y_values[j]
            sublist.append(LRR_probs_expected_value)
        load_resist_ratios_probs_expected_cumsum.append(sum(sublist))
        load_resist_ratios_probs_expected.append(sublist)
        #print(f"Beam {i+1} LRR expected: {load_resist_ratios_probs_expected[i]}")
        #print(f"Beam {i+1} LRR expected sum: {load_resist_ratios_probs_expected_cumsum[i]}")
    #create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs_expected, beam_titles, "25 Beam types & LRR probabilities expected"+evidence)


 # 4. Find the most likely beam by first summing LRR_minus_target for each beam

    # The most likely beam is the one with the smallest value of LRR_probs_minus_target divided by the sum of all LRR_probs_minus_target

    LRR_score = []
    load_resist_ratios_probs_cumsum_all_beams = sum(load_resist_ratios_probs_expected_cumsum)
    #print(f"The sum of all probabilities is {load_resist_ratios_probs_cumsum_all_beams}")

    for i in range(25):
        LRR_score.append(load_resist_ratios_probs_expected_cumsum[i] / load_resist_ratios_probs_cumsum_all_beams)
        #print(f"Beam {i+1} LRR score: {LRR_score[i]}")

    # Find the index of the beam with the least error which is greater than zero and print the beam number
    max_LRR = max(i for i in LRR_score if i > 0)
    max_LRR_index = LRR_score.index(max_LRR)
    max_LRR_beam = beam_titles[max_LRR_index]
    print(f"The beam with the highest score is Beam {max_LRR_beam} with the highest score: {round(max_LRR, 3)}")
    #create_one_bar_chart(beam_titles, LRR_score, '25 Beam types & highest LRR score'+evidence, 'Beam Type', 'Score', 'green')
    return max_LRR_index+1

def likely_beam_by_score(n_intervals, evidence, y_max_lrr_prob, y_max_lrr_prob_expected):
#Find the most likely beam by using expected value of the LRR for each beam
    # 1. Find the LRR for each of the beams using the Bayesian network and plot the results

    load_resist_ratios = []
    load_resist_ratios_probs = []
    load_resist_ratios_probs_cumsum = []
    volume_list = []

    for beam_num in range(1, 26):
        beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
        output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
        output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
        output_volume = beam_BN_instance.vol_output_z[0]

        load_resist_ratios.append(output_LRR)
        load_resist_ratios_probs.append(output_LRR_prob)
        load_resist_ratios_probs_cumsum.append(sum(output_LRR_prob))
        volume_list.append(output_volume)
        #print(f"Beam {beam_num} LRR probabilities for each interval: {output_LRR_prob}")
        #print(f"Beam {beam_num} volume: {round(output_volume, 3)}")
    #create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs, beam_titles, "25 Beam types & LRR probabilities "+evidence, y_max_lrr_prob)     
    # 2. For each beam: find expected values of each interval by multiplying LRR and target LRR to get 'expected' probabilities. Sum up the values to get the expected value for each beam

    load_resist_ratios_probs_expected = []
    load_resist_ratios_probs_expected_cumsum = []
    for i in range(25):
        sublist = []
        for j in range(n_intervals):
            if load_resist_ratios_probs_cumsum[i] == 0:
                LRR_probs_expected_value = 0
            else:
                LRR_probs_expected_value = load_resist_ratios_probs[i][j]*log_y_values[j]
            sublist.append(LRR_probs_expected_value)
        load_resist_ratios_probs_expected_cumsum.append(sum(sublist))
        load_resist_ratios_probs_expected.append(sublist)
        #print(f"Beam {i+1} LRR expected: {load_resist_ratios_probs_expected[i]}")
        #print(f"Beam {i+1} LRR expected sum: {load_resist_ratios_probs_expected_cumsum[i]}")
    #create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs_expected, beam_titles, "25 Beam types & LRR probabilities expected " + evidence, y_max_lrr_prob_expected)


    # 3. Find the most likely beam by finding the fraction of each beam's expected probabilities in the sum of all beams expected probabilities

    LRR_score = []
    LRR_score_times_volume = []
    load_resist_ratios_probs_cumsum_all_beams = sum(load_resist_ratios_probs_expected_cumsum)
    #print(f"The sum of all probabilities is {load_resist_ratios_probs_cumsum_all_beams}")

    for i in range(25):
        LRR_score_i = load_resist_ratios_probs_expected_cumsum[i] / load_resist_ratios_probs_cumsum_all_beams
        LRR_score_times_volume_i = LRR_score_i * volume_list[i]
        LRR_score.append(LRR_score_i)
        LRR_score_times_volume.append(LRR_score_times_volume_i)

        #print(f"Beam {i+1} LRR score: {round(LRR_score[i], 3)}")
        #print(f"Beam {i+1} LRR score times volume: {round(LRR_score_times_volume_i, 3)}")

    # Find the index of the beam with the least error which is greater than zero and print the beam number
    expected_volume = sum(LRR_score_times_volume)
    max_LRR = max(i for i in LRR_score if i > 0)
    max_LRR_index = LRR_score.index(max_LRR)
    max_LRR_beam = beam_titles[max_LRR_index]
    print(f"For {evidence} the beam with the highest score is Beam {max_LRR_beam} with the highest score: {round(max_LRR, 3)}")
    #create_one_bar_chart_with_horizontal_line(beam_titles, LRR_score, '25 Beam types & highest LRR score '+evidence, 'Beam Type', 'Score', 'green', max_LRR, max_LRR_beam)
    #create_one_bar_chart_with_label(beam_titles, LRR_score_times_volume, '25 Beam types & expected volume '+evidence, 'Beam Type', 'Volume * score [m$^{3}$]', 'orange', expected_volume, 'Expected combined volume =')
    print(f"For {evidence} the expected volume considering all beams and their scores is {round(expected_volume, 3)}")
    return max_LRR_index+1, expected_volume

def likely_beam_by_score_figs(n_intervals, evidence, y_max_lrr_prob, y_max_lrr_prob_expected):
#Find the most likely beam by using expected value of the LRR for each beam
    # 1. Find the LRR for each of the beams using the Bayesian network and plot the results

    load_resist_ratios = []
    load_resist_ratios_probs = []
    load_resist_ratios_probs_cumsum = []
    volume_list = []

    for beam_num in range(1, 26):
        beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
        output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
        output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
        output_volume = beam_BN_instance.vol_output_z[0]

        load_resist_ratios.append(output_LRR)
        load_resist_ratios_probs.append(output_LRR_prob)
        load_resist_ratios_probs_cumsum.append(sum(output_LRR_prob))
        volume_list.append(output_volume)
        #print(f"Beam {beam_num} LRR probabilities for each interval: {output_LRR_prob}")
        #print(f"Beam {beam_num} volume: {round(output_volume, 3)}")
    #create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs, beam_titles, "25 Beam types & LRR probabilities "+evidence, y_max_lrr_prob)     
    # 2. For each beam: find expected values of each interval by multiplying LRR and target LRR to get 'expected' probabilities. Sum up the values to get the expected value for each beam

    load_resist_ratios_probs_expected = []
    load_resist_ratios_probs_expected_cumsum = []
    for i in range(25):
        sublist = []
        for j in range(n_intervals):
            if load_resist_ratios_probs_cumsum[i] == 0:
                LRR_probs_expected_value = 0
            else:
                LRR_probs_expected_value = load_resist_ratios_probs[i][j]*log_y_values[j]
            sublist.append(LRR_probs_expected_value)
        load_resist_ratios_probs_expected_cumsum.append(sum(sublist))
        load_resist_ratios_probs_expected.append(sublist)
        #print(f"Beam {i+1} LRR expected: {load_resist_ratios_probs_expected[i]}")
        #print(f"Beam {i+1} LRR expected sum: {load_resist_ratios_probs_expected_cumsum[i]}")
    #reate_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs_expected, beam_titles, "25 Beam types & LRR probabilities expected " + evidence, y_max_lrr_prob_expected)


    # 3. Find the most likely beam by finding the fraction of each beam's expected probabilities in the sum of all beams expected probabilities

    LRR_score = []
    LRR_score_times_volume = []
    load_resist_ratios_probs_cumsum_all_beams = sum(load_resist_ratios_probs_expected_cumsum)
    #print(f"The sum of all probabilities is {load_resist_ratios_probs_cumsum_all_beams}")

    for i in range(25):
        LRR_score_i = load_resist_ratios_probs_expected_cumsum[i] / load_resist_ratios_probs_cumsum_all_beams
        LRR_score_times_volume_i = LRR_score_i * volume_list[i]
        LRR_score.append(LRR_score_i)
        LRR_score_times_volume.append(LRR_score_times_volume_i)

        #print(f"Beam {i+1} LRR score: {round(LRR_score[i], 3)}")
        #print(f"Beam {i+1} LRR score times volume: {round(LRR_score_times_volume_i, 3)}")

    # Find the index of the beam with the least error which is greater than zero and print the beam number
    expected_volume = sum(LRR_score_times_volume)
    max_LRR = max(i for i in LRR_score if i > 0)
    max_LRR_index = LRR_score.index(max_LRR)
    max_LRR_beam = beam_titles[max_LRR_index]
    #print(f"For {evidence} the beam with the highest score is Beam {max_LRR_beam} with the highest score: {round(max_LRR, 3)}")
    #create_one_bar_chart_with_horizontal_line(beam_titles, LRR_score, 'Bldg_1_Beam_prob'+evidence, 'Beam Type', 'Probability', 'blue', max_LRR, max_LRR_beam)
    #create_one_bar_chart_with_label(beam_titles, LRR_score_times_volume, '25 Beam types & expected volume '+evidence, 'Beam Type', 'Volume * score [m$^{3}$]', 'red', expected_volume, 'Expected combined volume =')
    #print(f"For {evidence} the expected volume considering all beams and their scores is {round(expected_volume, 3)}")
    return max_LRR_index+1, expected_volume

def create_one_bar_chart_with_horizontal_line(x_values, y_values, chart_title, x_axis_title, y_axis_title, color, horizontal_line_value, value):
    import matplotlib.pyplot as plt
    """
    Creates a single bar chart.

    Parameters:
    x_values (list): A list of x-axis values.
    y_values (list): A list of y-axis values.
    chart_title (str): The title of the chart.
    x_axis_title (str): The title of the x-axis.
    y_axis_title (str): The title of the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, width=0.3, color=color, edgecolor='black', alpha=0.7)
    plt.xlabel(x_axis_title, fontsize=16)
    plt.ylabel(y_axis_title, fontsize=16)
    plt.tick_params(axis='x', rotation=90, labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    #plt.title(chart_title, fontsize=18)

        # Highlight the horizontal line value
    plt.axhline(y=horizontal_line_value, color='blue', linestyle='--', linewidth=2, label=f'Beam {value} has the highest probability of: {horizontal_line_value:.2f}')
    plt.legend(fontsize=18)
    plt.ylim(0, 0.3)#Adjust the upper limit as needed
    # Save the figure based on the name of the chart title
    sanitized_title = re.sub(r'[^\w\s]', '', chart_title).replace(" ", "_")
    filename = sanitized_title + ".png"
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight',dpi=720)
    # Set the y-axis limit
    
    #plt.show()

def create_one_bar_chart_with_label(x_values, y_values, chart_title, x_axis_title, y_axis_title, color, value, label):
    import matplotlib.pyplot as plt
    """
    Creates a single bar chart.

    Parameters:
    x_values (list): A list of x-axis values.
    y_values (list): A list of y-axis values.
    chart_title (str): The title of the chart.
    x_axis_title (str): The title of the x-axis.
    y_axis_title (str): The title of the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, width=0.3, color=color,edgecolor='black', alpha=0.7)
    plt.xlabel(x_axis_title, fontsize=18)
    plt.ylabel(y_axis_title, fontsize=18)
    plt.tick_params(axis='x', rotation=90, labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.title(chart_title,fontsize=18)

    # put the label on the chart
    plt.text(0, max(y_values), f'{label} {value:.3f}', fontsize=18, color='red')
    
    # Save the figure based on the name of the chart title
    sanitized_title = re.sub(r'[^\w\s]', '', chart_title).replace(" ", "_")
    filename = sanitized_title + ".png"
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight',dpi=720)



# print and plot the results for the LRR and scores for all beams to select the most likely beam 
def given_evidence_output_alternative_likely_beam(n_intervals):
    if building_length == 0 and building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 0")
    if building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 1")
    if roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 2")
    if decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 3")
    if char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 4")
    if spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 5")
    if building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0 and spacing > 0:
        beam_num = alternative_likely_beam(n_intervals, " - Evidence 6")
    return beam_num


    
def plot_25_beam_output_figures(load_resist_ratios, load_resist_ratios_probs,load_resist_ratios_probs_normalized, LRR_probs_minus_target, LRR_least_error):
    create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs, beam_titles, "25 Beam types and LRR probabilities")
    create_5x5_bar_charts(load_resist_ratios, load_resist_ratios_probs_normalized, beam_titles, "25 Beam types and LRR probabilities normalized")
    # create 5x5 bar charts for the LRR minus target
    create_5x5_bar_charts(load_resist_ratios, LRR_probs_minus_target, beam_titles, "25 Beam types and LRR minus target probabilities")
    create_one_bar_chart(beam_titles, LRR_least_error, '25 Beam types and least error', 'Beam Type', 'Error', 'red')

def plot_beam_properties():
    # output the load resistance ratio and probability for the most likely beam

    beam_BN_instance = beam_BN(max_LRR_beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)

    # output the total beam volume in the roof of the building
    output_beam_volume_in_roof = beam_BN_instance.vol_output_x
    output_beam_volume_in_roof_prob = beam_BN_instance.vol_output_y

    #create_one_bar_chart(output_LRR_most_likely, output_LRR_prob_most_likely, f"Utilization ratios for the most likely beam: {most_likely_beam_1}", "Utilization ratios", "Probability", "green")

    beam_BN_instance.inference_beam_evidence(max_LRR_beam_num) #plots figures for beam properties and resistance for the most likely beam

def volume_beam_BN():
    # output the load resistance ratio and probability for the most likely beam

    beam_BN_instance = beam_BN(max_LRR_beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)

    # output the total beam volume in the roof of the building
    output_beam_volume_in_roof = beam_BN_instance.vol_output_x
    output_beam_volume_in_roof_prob = beam_BN_instance.vol_output_y
    
    if building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0 and spacing > 0:
        vol = output_beam_volume_in_roof
        prob = output_beam_volume_in_roof_prob
        vol_one = beam_BN_instance.vol_output_z
    else:
        vol = 0
        prob = 0
        vol_one = 0

    # if building_length == 0 and building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, "P(V)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, "P(V|L)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, "P(V|L, W)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, f"P(V|L, W, $\\alpha$)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, f"P(V|L, W, $\\alpha$, yr)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, f"P(V|L, W, $\\alpha$, yr, $f_{{m,k}}$)", f"Beam volume $[m^{3}]$", "Probability", "blue")
    # if building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0 and spacing > 0:
    #     create_one_bar_chart_with_mean(output_beam_volume_in_roof, output_beam_volume_in_roof_prob, f"P(V|L, W, $\\alpha$, yr, $f_{{m,k}}$, a)", f"Beam volume $[m^{3}]$", "Probability", "blue")

    return vol, prob, vol_one

def find_one_in_list(A, B):
    value_in_A = None
    for a, b in zip(A, B):
        if b == 1:
            value_in_A = a
            break

    print(f"The value in list A where the value in list B equals 1 is: {value_in_A}") 
    return value_in_A


# print and plot the results for the LRR and scores for all beams to select the most likely beam 
def given_evidence_output_likely_beam_volume(n_intervals):
    if building_length == 0 and building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 0", 0.2, 0.02)
    if building_width == 0 and roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 1", 0.25, 0.025)
    if roof_angle == 0 and decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 2", 0.3, 0.03)
    if decade == 0 and char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 3", 0.4, 0.035)
    if char_bending == 0 and spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 4", 0.6, 0.04)
    if spacing == 0 and building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 5", 0.8, 0.07)
    if building_length > 0 and building_width > 0 and roof_angle > 0 and decade > 0 and char_bending > 0 and spacing > 0:
        likely_beam_by_score_figs(n_intervals, " - Evidence 6", 1, 0.8)

""" decades = [1970, 1980, 1990, 2000, 2010, 2020]
char_bending_strengths = [18, 24, 30, 40]
spacings = [300, 350, 400, 450, 500, 550, 600]
roof_angles = [25, 30, 35, 40, 45, 50, 55] """
# try for different values of the building properties (normal operating points)
beam_num = 25 # beam number to be used in the Bayesian network

# Building properties - Evidence 0
building_length = 0 #22m
building_width =  0#8.60m
roof_angle = 0 #38degrees
decade = 0 #year of construction
char_bending = 0 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)
beam_name = beam_titles[beam_num-1]

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]



chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 0"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 0"
chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 0"
create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 1
building_length = 22 #22m
building_width =  0#8.60m
roof_angle = 0 #38degrees
decade = 0 #year of construction
char_bending = 0 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 1"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 1"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 1"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 2
building_length = 22 #22m
building_width =  8.6#8.60m
roof_angle = 0 #38degrees
decade = 0 #year of construction
char_bending = 0 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 2"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 2"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 2"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 3
building_length = 22 #22m
building_width =  8.6#8.60m
roof_angle = 38 #38degrees
decade = 0 #year of construction
char_bending = 0 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 3"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 3"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 3"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 4
building_length = 22 #22m
building_width =  8.6#8.60m
roof_angle = 38 #38degrees
decade = 1970 #year of construction
char_bending = 0 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 4"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 4"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 4"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 5
building_length = 22 #22m
building_width =  8.6#8.60m
roof_angle = 38 #38degrees
decade = 1970 #year of construction
char_bending = 40 #40N/mm^2
spacing = 0 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 5"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 5"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 5"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

# Building properties - Evidence 6
building_length = 22 #22m
building_width =  8.6#8.60m
roof_angle = 38 #38degrees
decade = 1970 #year of construction
char_bending = 40 #40N/mm^2
spacing = 300 #300mm between beams
given_evidence_output_likely_beam_volume(n_intervals)

beam_BN_instance = beam_BN(beam_num, nint, building_length, building_width, roof_angle, decade, char_bending, spacing)
output_LRR = beam_BN_instance.u_output_x[:n_intervals] #only the first n for each beam
output_LRR_prob = beam_BN_instance.u_output_y[:n_intervals] #only the first n for each beam
output_beam_resist = beam_BN_instance.beam_resist_x#[:n_intervals]
output_beam_resist_prob = beam_BN_instance.beam_resist_y#[:n_intervals]
output_beam_load = beam_BN_instance.beam_load_x#[:n_intervals]
output_beam_load_prob = beam_BN_instance.beam_load_y#[:n_intervals]


chart_title_u = f"Bldg 1 LRR for: {beam_name}- Evidence 6"
chart_title_Rd = f"Bldg 1 Rd for: {beam_name}- Evidence 6"
chart_title_Ed = f"Bldg 1 Ed for: {beam_name}- Evidence 6"

create_one_bar_chart(output_beam_resist, output_beam_resist_prob, chart_title_Rd, "Design resistance [kN/m]", "Probability", color="green")
create_one_bar_chart(output_beam_load, output_beam_load_prob, chart_title_Ed, "Design load [kN/m]", "Probability", color="red")
create_one_bar_chart(output_LRR, output_LRR_prob, chart_title_u, "Utilization ratio [-]", "Probability", color="orange")

def plot_scatter_with_labels(A, B, C, D, X, labels, building_name):
    """
    Creates a scatter plot with the values in the lists A, B, C, and D, and uses A, B, C, and D as the legend.
    Uses another list X with 7 items as the labels on the x-axis.
    
    Parameters:
    A (list): A list of values for the first dataset.
    B (list): A list of values for the second dataset.
    C (list): A list of values for the third dataset.
    D (list): A list of values for the fourth dataset.
    X (list): A list of labels for the x-axis.
    """
        # Find the global min and max values across A, B, C, and D
    all_values = A + B + C + D
    y_min = min(all_values)
    y_max = max(all_values)

     # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Plot each list with scatter and line
    plt.scatter(range(len(A)), A, label=str(labels[0])+ " MPa", color='blue')
    plt.plot(range(len(A)), A, color='blue')

    plt.scatter(range(len(B)), B, label=str(labels[1]) + " MPa", color='green')
    plt.plot(range(len(B)), B, color='green')

    plt.scatter(range(len(C)), C, label=str(labels[2])  + " MPa", color='red')
    plt.plot(range(len(C)), C, color='red')

    plt.scatter(range(len(D)), D, label=str(labels[3]) + " MPa", color='purple')
    plt.plot(range(len(D)), D, color='purple')

    # Add labels and title
    plt.xlabel('Beam spacings [mm]', fontsize=18)
    plt.ylabel('Volume [m$^{3}$]', fontsize=18)
    #plt.title(f'The effect of varying beam spacings and characteristic strength on volume - ' + building_name, fontsize=16)
    plt.legend(fontsize=18)

    # Set x-axis tick labels to represent the actual values from list X
    plt.xticks(range(len(X)), X, rotation=0, fontsize=18)
    #plt.yticks()
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    # Set y-axis limits
    plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))  # Add some padding
    # Add gridlines
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    #plt.show()
    #plt.savefig(f"{building_name}_Vol_Str_Spacing.png", bbox_inches='tight', dpi=720)


def plot_scatter_line(ax, x_data, y_data, label, color):
    ax.scatter(range(len(y_data)), y_data, label=label, color=color)
    ax.plot(range(len(y_data)), y_data, color=color)



#plot_vol_beam_figures(volumes_decades, volumes_char_bending_strengths, volumes_spacings, volumes_roof_angles)   



#plot_beam_figures()
#plot_25_beam_output_figures()
char_bending_strengths = [18, 24, 30, 40]
spacings = [300, 350, 400, 450, 500, 550, 600]

volumes_spacings_strengths = []
for char_bending in char_bending_strengths:
    sublist = []
    for spacing in spacings:
        vol = likely_beam_by_score(n_intervals, " - Evidence 6", 1, 0.8)[1]
        sublist.append(vol)
    volumes_spacings_strengths.append(sublist)
#print(f'The volume for spacing and strengths is {volumes_spacings_strengths}')
plot_scatter_with_labels(volumes_spacings_strengths[0], volumes_spacings_strengths[1], volumes_spacings_strengths[2], volumes_spacings_strengths[3], spacings, char_bending_strengths, "Bldg_1")

building_length = 10.9 #m
building_width =  7.8#m
roof_angle =  36#degrees
decade = 1970#year of construction
volumes_spacings_strengths = []
for char_bending in char_bending_strengths:
    sublist = []
    for spacing in spacings:
        vol = likely_beam_by_score(n_intervals, " - Evidence 6", 1, 0.8)[1]
        sublist.append(vol)
    volumes_spacings_strengths.append(sublist)
#print(f'The volume for spacing and strengths is {volumes_spacings_strengths}')
plot_scatter_with_labels(volumes_spacings_strengths[0], volumes_spacings_strengths[1], volumes_spacings_strengths[2], volumes_spacings_strengths[3], spacings, char_bending_strengths, "Bldg_2")

building_length = 12.7 #m
building_width =  8.4#m
roof_angle =  31#degrees
decade = 1970#year of construction
volumes_spacings_strengths = []
for char_bending in char_bending_strengths:
    sublist = []
    for spacing in spacings:
        vol = likely_beam_by_score(n_intervals, " - Evidence 6", 1, 0.8)[1]
        sublist.append(vol)
    volumes_spacings_strengths.append(sublist)
#print(f'The volume for spacing and strenghths is {volumes_spacings_strengths}')
plot_scatter_with_labels(volumes_spacings_strengths[0], volumes_spacings_strengths[1], volumes_spacings_strengths[2], volumes_spacings_strengths[3], spacings, char_bending_strengths, "Bldg_3")


building_length = 20 #m
building_width =  7.2#m
roof_angle =  37#degrees
decade = 1970#year of construction
volumes_spacings_strengths = []
for char_bending in char_bending_strengths:
    sublist = []
    for spacing in spacings:
        vol = likely_beam_by_score(n_intervals, " - Evidence 6", 1, 0.8)[1]
        sublist.append(vol)
    volumes_spacings_strengths.append(sublist)
#print(f'The volume for spacing and strenghths is {volumes_spacings_strengths}')
plot_scatter_with_labels(volumes_spacings_strengths[0], volumes_spacings_strengths[1], volumes_spacings_strengths[2], volumes_spacings_strengths[3], spacings, char_bending_strengths, "Bldg_4")

building_length = 12.8 #m
building_width =  7.73#m
roof_angle =  40#degrees
decade = 1970#year of construction
volumes_spacings_strengths = []
for char_bending in char_bending_strengths:
    sublist = []
    for spacing in spacings:
        vol = likely_beam_by_score(n_intervals, " - Evidence 6", 1, 0.8)[1]
        sublist.append(vol)
    volumes_spacings_strengths.append(sublist)
#print(f'The volume for spacing and strenghths is {volumes_spacings_strengths}')
plot_scatter_with_labels(volumes_spacings_strengths[0], volumes_spacings_strengths[1], volumes_spacings_strengths[2], volumes_spacings_strengths[3], spacings, char_bending_strengths, "Bldg_5")



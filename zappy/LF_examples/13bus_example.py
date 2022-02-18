from zappy.LF_elements.bus import ACbus, DCbus
from zappy.LF_elements.line import ACline, DCline
from zappy.LF_elements.generator import ACgenerator, DCgenerator
from zappy.LF_elements.load import ACload, DCload
from zappy.LF_elements.converter import Converter
import numpy as np

from openmdao.api import Group, IndepVarComp
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, NonlinearBlockGS

import math, cmath
import time

Vacbase = 4160. # base voltage of AC bus 
Vdcbase = 6800. # base voltage of DC bus
PowerBase = 10.0e6 # base power of system (VA)
        
class Example(Group):

    """Example from page 7, A Generalized Approach to the Load Flow Analysis of AC-DC Hybrid Distribution Systems by Ahmed et. al""" 

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
   
        nn =self.options['num_nodes']

        IVC = self.add_subsystem('IVC', IndepVarComp(), promotes=['*'])
        
        # LINES -------------------------------------------------------------
        
        IVC.add_output('R1_2', 0.218734378, units='ohm') 
        IVC.add_output('X1_2', 0.360978021, units='ohm')
        
        IVC.add_output('R1_9', 0.224454412, units='ohm')
        IVC.add_output('X1_9', 0.36093195, units='ohm')
        
        IVC.add_output('R2_3', 0.892493161, units='ohm')
        IVC.add_output('X2_3', 1.542190824, units='ohm')
        
        IVC.add_output('R3_10', 0.056583399, units='ohm')
        IVC.add_output('X3_10', 0.749529076, units='ohm')
        
        IVC.add_output('R3_11', 0.040822812, units='ohm')
        IVC.add_output('X3_11', 0.750680512, units='ohm')
        
        IVC.add_output('R4_5', 0.25349905, units='ohm')
        
        IVC.add_output('R4_11dc', 0.509605338, units='ohm')
        
        IVC.add_output('R5_6', 0.209734086, units='ohm')
     
        IVC.add_output('R6_13dc', 0.493075443, units='ohm')
        
        IVC.add_output('R7_8', 0.438930098, units='ohm')
        IVC.add_output('X7_8', 0.726902905, units='ohm')
        
        IVC.add_output('R7_12', 0.03336043, units='ohm')
        IVC.add_output('X7_12', 0.741618427, units='ohm')
        
        IVC.add_output('R7_13', 0.021166743, units='ohm')
        IVC.add_output('X7_13', 0.766806199, units='ohm')
        
        IVC.add_output('R8_9', 0.397894209, units='ohm')
        IVC.add_output('X8_9', 0.758227135, units='ohm')
        
        IVC.add_output('R10dc_12dc', 0.892317972, units='ohm')
        
        self.add_subsystem('Line1_2', ACline(num_nodes=nn), promotes=[('R','R1_2'), ('X','X1_2'),
                                                       ('Vr_in','Vr_1'), ('Vr_out','Vr_2'), 
                                                       ('Vi_in','Vi_1'), ('Vi_out','Vi_2'),
                                                       ('Ir_in','L1_2:Ir'), ('Ii_in','L1_2:Ii'),
                                                       ('Ir_out','L2_1:Ir'), ('Ii_out','L2_1:Ii')])
        
        self.add_subsystem('Line1_9', ACline(num_nodes=nn), promotes=[('R','R1_9'), ('X','X1_9'),
                                                       ('Vr_in','Vr_1'), ('Vr_out','Vr_9'), 
                                                       ('Vi_in','Vi_1'), ('Vi_out','Vi_9'),
                                                       ('Ir_in','L1_9:Ir'), ('Ii_in','L1_9:Ii'),
                                                       ('Ir_out','L9_1:Ir'), ('Ii_out','L9_1:Ii')])
        
        self.add_subsystem('Line2_3', ACline(num_nodes=nn), promotes=[('R','R2_3'), ('X','X2_3'),
                                                       ('Vr_in','Vr_2'), ('Vr_out','Vr_3'), 
                                                       ('Vi_in','Vi_2'), ('Vi_out','Vi_3'),
                                                       ('Ir_in','L2_3:Ir'), ('Ii_in','L2_3:Ii'),
                                                       ('Ir_out','L3_2:Ir'), ('Ii_out','L3_2:Ii')])
        
        self.add_subsystem('Line3_10', ACline(num_nodes=nn), promotes=[('R','R3_10'), ('X','X3_10'),
                                                       ('Vr_in','Vr_3'), ('Vr_out','Vr_10'), 
                                                       ('Vi_in','Vi_3'), ('Vi_out','Vi_10'),
                                                       ('Ir_in','L3_10:Ir'), ('Ii_in','L3_10:Ii'),
                                                       ('Ir_out','L10_3:Ir'), ('Ii_out','L10_3:Ii')])
        
        self.add_subsystem('Line3_11', ACline(num_nodes=nn), promotes=[('R','R3_11'), ('X','X3_11'),
                                                       ('Vr_in','Vr_3'), ('Vr_out','Vr_11'), 
                                                       ('Vi_in','Vi_3'), ('Vi_out','Vi_11'),
                                                       ('Ir_in','L3_11:Ir'), ('Ii_in','L3_11:Ii'),
                                                       ('Ir_out','L11_3:Ir'), ('Ii_out','L11_3:Ii')])
        
        self.add_subsystem('Line4_5', DCline(num_nodes=nn), promotes=[('R','R4_5'),
                                                       ('V_in','V_4'), ('V_out','V_5'), 
                                                       ('I_in','L4_5:I'), 
                                                       ('I_out','L5_4:I')])
        
        self.add_subsystem('Line4_11dc', DCline(num_nodes=nn), promotes=[('R','R4_11dc'),
                                                       ('V_in','V_4'), ('V_out','V_11dc'), 
                                                       ('I_in','L4_11dc:I'),
                                                       ('I_out','L11dc_4:I')])
        
        self.add_subsystem('Line5_6', DCline(num_nodes=nn), promotes=[('R','R5_6'),
                                                       ('V_in','V_5'), ('V_out','V_6'),
                                                       ('I_in','L5_6:I'), 
                                                       ('I_out','L6_5:I')])
        
        self.add_subsystem('Line6_13dc', DCline(num_nodes=nn), promotes=[('R','R6_13dc'),
                                                       ('V_in','V_6'), ('V_out','V_13dc'),
                                                       ('I_in','L6_13dc:I'),
                                                       ('I_out','L13dc_6:I')])
        
        self.add_subsystem('Line7_8', ACline(num_nodes=nn), promotes=[('R','R7_8'), ('X','X7_8'),
                                                       ('Vr_in','Vr_7'), ('Vr_out','Vr_8'), 
                                                       ('Vi_in','Vi_7'), ('Vi_out','Vi_8'),
                                                       ('Ir_in','L7_8:Ir'), ('Ii_in','L7_8:Ii'),
                                                       ('Ir_out','L8_7:Ir'), ('Ii_out','L8_7:Ii')])
        
        self.add_subsystem('Line7_12', ACline(num_nodes=nn), promotes=[('R','R7_12'), ('X','X7_12'),
                                                       ('Vr_in','Vr_7'), ('Vr_out','Vr_12'), 
                                                       ('Vi_in','Vi_7'), ('Vi_out','Vi_12'),
                                                       ('Ir_in','L7_12:Ir'), ('Ii_in','L7_12:Ii'),
                                                       ('Ir_out','L12_7:Ir'), ('Ii_out','L12_7:Ii')])
        
        self.add_subsystem('Line7_13', ACline(num_nodes=nn), promotes=[('R','R7_13'), ('X','X7_13'),
                                                       ('Vr_in','Vr_7'), ('Vr_out','Vr_13'), 
                                                       ('Vi_in','Vi_7'), ('Vi_out','Vi_13'),
                                                       ('Ir_in','L7_13:Ir'), ('Ii_in','L7_13:Ii'),
                                                       ('Ir_out','L13_7:Ir'), ('Ii_out','L13_7:Ii')])
        
        self.add_subsystem('Line8_9', ACline(num_nodes=nn), promotes=[('R','R8_9'), ('X','X8_9'),
                                                       ('Vr_in','Vr_8'), ('Vr_out','Vr_9'), 
                                                       ('Vi_in','Vi_8'), ('Vi_out','Vi_9'),
                                                       ('Ir_in','L8_9:Ir'), ('Ii_in','L8_9:Ii'),
                                                       ('Ir_out','L9_8:Ir'), ('Ii_out','L9_8:Ii')])
        
        self.add_subsystem('Line10dc_12dc', DCline(num_nodes=nn), promotes=[('R','R10dc_12dc'),
                                                       ('V_in','V_10dc'), ('V_out','V_12dc'),
                                                       ('I_in','L10dc_12dc:I'), 
                                                       ('I_out','L12dc_10dc:I')])
        
        # GENERATORS ------------------------------------------------------
        
        # Slack Generator
        # Vm_ac_bus is NOT in per unit values
        IVC.add_output('Vm_ac_bus1', 1.05*Vacbase, units='V') # based on Slack Generator 1
        IVC.add_output('thetaV_ac_bus', 0.0, units='deg')
        
        # AC Generator
        IVC.add_output('Vm_ac_bus2', 1.00*Vacbase, units='V')
        IVC.add_output('P_G2', -2.5, units='MW') # when we generate power, it leaves the generator
        
        # DC Generator
        IVC.add_output('Vm_dc_bus', 1.0*Vdcbase, units='V') # based on DC bus desired voltage
        
        # AC Generator
        IVC.add_output('Vm_ac_bus4', 1.00*Vacbase, units='V')
        IVC.add_output('P_G4', -2.5, units='MW') # When we generate power it leaves the generator 
        
        self.add_subsystem('Gen1', ACgenerator(num_nodes=nn, mode='Slack', 
                                               Vbase = Vacbase,
                                               Sbase = PowerBase), 
                           promotes=[('Vm_bus','Vm_ac_bus1'), 
                                     ('thetaV_bus','thetaV_ac_bus'),
                                     ('Vr_out','Vr_1'),('Vi_out','Vi_1'),
                                     ('Ir_out','LG1:Ir'),('Ii_out','LG1:Ii')])
        
        self.add_subsystem('Gen2', ACgenerator(num_nodes=nn, mode='P-V', Q_max = -0.1e6, Q_min = -0.75e6, 
                                               Vbase = Vacbase,
                                               Sbase = PowerBase), 
                           promotes=[('Vm_bus','Vm_ac_bus2'),('P_bus','P_G2'),
                                     ('Vr_out','Vr_3'),('Vi_out','Vi_3'),
                                     ('Ir_out','LG2:Ir'),('Ii_out','LG2:Ii')])
        
        self.add_subsystem('Gen3', DCgenerator(num_nodes=nn, P_max = -0.5e6, P_min = -2.0e6, 
                                               Vbase = Vdcbase,
                                               Sbase = PowerBase), 
                           promotes=[('V_bus','Vm_dc_bus'), # ('P_out','P_G3'),
                                     ('V_out','V_5'),
                                     ('I_out','LG3:I')])
        
        self.add_subsystem('Gen4', ACgenerator(num_nodes=nn, mode='P-V', Q_max = -0.1e6, Q_min = -0.75e6, 
                                               Vbase = Vacbase,
                                               Sbase = PowerBase),  
                           promotes=[('Vm_bus','Vm_ac_bus4'),('P_bus','P_G4'),
                                     ('Vr_out','Vr_8'),('Vi_out','Vi_8'),
                                     ('Ir_out','LG4:Ir'),('Ii_out','LG4:Ii')])
             
        # LOADS -----------------------------------------------------------

        IVC.add_output('P2', 2.0, units='MW')
        IVC.add_output('Q2',0.4, units='MV*A')
        IVC.add_output('P3', 1.5, units='MW') # real power of load on bus 3
        IVC.add_output('Q3',0.2, units='MV*A') # imaginary power of load on bus 3
        IVC.add_output('P4', 1.0, units='MW')
        IVC.add_output('P6', 1.0, units='MW')
        IVC.add_output('P7', 2.5, units='MW')
        IVC.add_output('Q7',0.5, units='MV*A')
        IVC.add_output('P8', 1.0, units='MW')
        IVC.add_output('Q8',0.1, units='MV*A')
        IVC.add_output('P9', 2.5, units='MW')
        IVC.add_output('Q9',0.5, units='MV*A')
        
        self.add_subsystem('Load2', ACload(num_nodes=nn), promotes=[('P','P2'), ('Q','Q2'), # this has some real and imaginary load 
                                                       ('Vr_in','Vr_2'), ('Vi_in','Vi_2'), # This is connected to this bus voltage
                                                       ('Ir_in','LL2:Ir'),('Ii_in','LL2:Ii')]) # this is connected to the same bus current

        self.add_subsystem('Load3', ACload(num_nodes=nn), promotes=[('P','P3'), ('Q','Q3'),
                                                       ('Vr_in','Vr_3'), ('Vi_in','Vi_3'), 
                                                       ('Ir_in','LL3:Ir'),('Ii_in','LL3:Ii')])

        self.add_subsystem('Load4', DCload(num_nodes=nn), promotes=[('P','P4'),
                                                       ('V_in','V_4'),  
                                                       ('I_in','LL4:I')])
                
        self.add_subsystem('Load6', DCload(num_nodes=nn), promotes=[('P','P6'),
                                                       ('V_in','V_6'),
                                                       ('I_in','LL6:I')])
        
        self.add_subsystem('Load7', ACload(num_nodes=nn), promotes=[('P','P7'), ('Q','Q7'),
                                                       ('Vr_in','Vr_7'), ('Vi_in','Vi_7'), 
                                                       ('Ir_in','LL7:Ir'),('Ii_in','LL7:Ii')])

        self.add_subsystem('Load8', ACload(num_nodes=nn), promotes=[('P','P8'), ('Q','Q8'),
                                                       ('Vr_in','Vr_8'), ('Vi_in','Vi_8'), 
                                                       ('Ir_in','LL8:Ir'),('Ii_in','LL8:Ii')])

        self.add_subsystem('Load9', ACload(num_nodes=nn), promotes=[('P','P9'), ('Q','Q9'),
                                                       ('Vr_in','Vr_9'), ('Vi_in','Vi_9'), 
                                                       ('Ir_in','LL9:Ir'),('Ii_in','LL9:Ii')])
        
    
        # BUSSES ----------------------------------------------------------
        
        self.add_subsystem('Bus1', 
                           ACbus(num_nodes=nn, lines=['L1_2', 'L1_9', 'LG1'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_1'), ('Vi', 'Vi_1'), 'L1_2:*', 'L1_9:*', 'LG1:*'])
        
        self.add_subsystem('Bus2', 
                           ACbus(num_nodes=nn, lines=['L2_1', 'L2_3', 'LL2'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase),
                           promotes=[('Vr', 'Vr_2'), ('Vi', 'Vi_2'), 'L2_1:*', 'L2_3:*', 'LL2:*'])
        
        self.add_subsystem('Bus3', 
                           ACbus(num_nodes=nn, lines=['L3_2', 'L3_10', 'L3_11', 'LG2', 'LL3'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_3'), ('Vi', 'Vi_3'), 'L3_2:*', 'L3_10:*', 'L3_11:*', 'LG2:*', 'LL3:*'])
        
        self.add_subsystem('Bus4', 
                           DCbus(num_nodes=nn, lines=['L4_11dc', 'L4_5', 'LL4'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase), 
                           promotes=[('V', 'V_4'), 'L4_11dc:*', 'L4_5:*', 'LL4:*'])
        
        self.add_subsystem('Bus5', 
                           DCbus(num_nodes=nn, lines=['L5_4', 'L5_6', 'LG3'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_5'), 'L5_4:*', 'L5_6:*', 'LG3:*'])
        
        self.add_subsystem('Bus6', 
                           DCbus(num_nodes=nn, lines=['L6_5', 'L6_13dc', 'LL6'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_6'), 'L6_5:*', 'L6_13dc:*', 'LL6:*'])
        
        self.add_subsystem('Bus7', 
                           ACbus(num_nodes=nn, lines=['L7_8', 'L7_12', 'L7_13', 'LL7'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_7'), ('Vi', 'Vi_7'), 'L7_8:*', 'L7_12:*', 'L7_13:*', 'LL7:*'])
        
        self.add_subsystem('Bus8', 
                           ACbus(num_nodes=nn, lines=['L8_7', 'L8_9', 'LG4', 'LL8'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_8'), ('Vi', 'Vi_8'), 'L8_7:*', 'L8_9:*', 'LG4:*', 'LL8:*'])
        
        self.add_subsystem('Bus9', 
                           ACbus(num_nodes=nn, lines=['L9_1', 'L9_8', 'LL9'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_9'), ('Vi', 'Vi_9'), 'L9_1:*', 'L9_8:*', 'LL9:*'])
        
        self.add_subsystem('Bus10', 
                           ACbus(num_nodes=nn, lines=['L10_3', 'LC10'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_10'), ('Vi', 'Vi_10'), 'L10_3:*', 'LC10:*'])
        
        self.add_subsystem('Bus10dc', 
                           DCbus(num_nodes=nn, lines=['L10dc_12dc', 'LC10dc'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_10dc'), 'L10dc_12dc:*', 'LC10dc:*'])

        self.add_subsystem('Bus11', 
                           ACbus(num_nodes=nn, lines=['L11_3', 'LC11'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_11'), ('Vi', 'Vi_11'), 'L11_3:*', 'LC11:*'])
        
        self.add_subsystem('Bus11dc', 
                           DCbus(num_nodes=nn, lines=['L11dc_4', 'LC11dc'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_11dc'),'L11dc_4:*', 'LC11dc:*'])
        
        self.add_subsystem('Bus12', 
                           ACbus(num_nodes=nn, lines=['L12_7', 'LC12'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_12'), ('Vi', 'Vi_12'), 'L12_7:*', 'LC12:*'])
        
        self.add_subsystem('Bus12dc', 
                           DCbus(num_nodes=nn, lines=['L12dc_10dc', 'LC12dc'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_12dc'), 'L12dc_10dc:*', 'LC12dc:*'])
        
        self.add_subsystem('Bus13', 
                           ACbus(num_nodes=nn, lines=['L13_7', 'LC13'],
                                 Vbase = Vacbase,
                                 Sbase = PowerBase), 
                           promotes=[('Vr', 'Vr_13'), ('Vi', 'Vi_13'), 'L13_7:*', 'LC13:*'])
        
        self.add_subsystem('Bus13dc', 
                           DCbus(num_nodes=nn, lines=['L13dc_6', 'LC13dc'],
                                 Vbase = Vdcbase,
                                 Sbase = PowerBase),  
                           promotes=[('V', 'V_13dc'), 'L13dc_6:*', 'LC13dc:*'])

        # AC / DC TRANSFORMERS --------------------------------------------
        
        IVC.add_output('Ksc', 0.611764706) # Converter constant, would be 1 if we were working in power units
        IVC.add_output('eff', 0.98) 
        IVC.add_output('PF1', 0.95)
        IVC.add_output('PF2', -0.95)
        IVC.add_output('M_10', 0.99)
        IVC.add_output('M_11', 0.99)
        IVC.add_output('M_12', 0.97)
        IVC.add_output('M_13', 0.96) 

        self.add_subsystem('TX10', Converter(num_nodes=nn, mode = 'Lead',
                                             Vdcbase = Vdcbase, 
                                             Sbase = PowerBase),  
                           promotes=[('M','M_10'),'Ksc','eff',('PF','PF1'), # Basic Settings for the converter
                                     ('Vr_ac','Vr_10'), ('Vi_ac','Vi_10'), # AC Bus Voltages (real & imaginary)
                                     ('Ir_ac','LC10:Ir'), ('Ii_ac','LC10:Ii'), # The AC Bus Currents (real & imaginary)
                                     ('V_dc','V_10dc'), # DC Bus Voltages (real only)
                                     ('I_dc','LC10dc:I')]) # The DC Bus Currents (real only)
        
        self.add_subsystem('TX11', Converter(num_nodes=nn, mode = 'Lead',
                                             Vdcbase = Vdcbase, 
                                             Sbase = PowerBase), 
                           promotes=[('M','M_11'),'Ksc','eff',('PF','PF1'),
                                     ('Vr_ac','Vr_11'), ('Vi_ac','Vi_11'),
                                     ('Ir_ac','LC11:Ir'),('Ii_ac','LC11:Ii'),
                                     ('V_dc','V_11dc'),
                                     ('I_dc','LC11dc:I')]) 
        
        self.add_subsystem('TX12', Converter(num_nodes=nn, mode = 'Lag',
                                             Vdcbase = Vdcbase,
                                             Sbase = PowerBase), 
                           promotes=[('M','M_12'),'Ksc','eff',('PF','PF2'),
                                     ('Vr_ac','Vr_12'), ('Vi_ac','Vi_12'),
                                     ('Ir_ac','LC12:Ir'),('Ii_ac','LC12:Ii'),
                                     ('V_dc','V_12dc'),
                                     ('I_dc','LC12dc:I')]) 
        
        self.add_subsystem('TX13', Converter(num_nodes=nn, mode = 'Lag',
                                             Vdcbase = Vdcbase, # TBD is this Vacbase or Vdcbase??
                                             Sbase = PowerBase), 
                           promotes=[('M','M_13'),'Ksc','eff',('PF','PF2'),
                                     ('Vr_ac','Vr_13'),('Vi_ac','Vi_13'),
                                     ('Ir_ac','LC13:Ir'),('Ii_ac','LC13:Ii'),
                                     ('V_dc','V_13dc'),
                                     ('I_dc','LC13dc:I')]) 


        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-4
        newton.options['rtol'] = 1e-4
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 3
        
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['print_bound_enforce'] = True
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = DirectSolver(assemble_jac=True)
    
if __name__ == "__main__":
        
    from openmdao.api import Problem

    prob = Problem()

    prob.model.add_subsystem('sys', Example(num_nodes=1), promotes=['*'])

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)
    prob.setup()
    prob.final_setup()

    Vacbase = 4160 # base voltage of AC bus 
    Vdcbase = 6800 # base voltage of DC bus
    
    prob['Vr_1'] = Vacbase
    prob['Vi_1'] = 0.0
    prob['Vr_2'] = Vacbase
    prob['Vi_2'] = 0.0
    prob['Vr_3'] = Vacbase
    prob['Vi_3'] = 0.0
    prob['V_4'] = Vdcbase
    prob['V_5'] = Vdcbase
    prob['V_6'] = Vdcbase
    prob['Vr_7'] = Vacbase
    prob['Vi_7'] = 0.0
    prob['Vr_8'] = Vacbase
    prob['Vi_8'] = 0.0
    prob['Vr_9'] = Vacbase
    prob['Vi_9'] = 0.0
    prob['Vr_10'] = Vacbase
    prob['Vi_10'] = 0.0
    prob['V_10dc'] = Vdcbase
    prob['Vr_11'] = Vacbase
    prob['Vi_11'] = 0.0
    prob['V_11dc'] = Vdcbase
    prob['Vr_12'] = Vacbase
    prob['Vi_12'] = 0.0
    prob['V_12dc'] = Vdcbase
    prob['Vr_13'] = Vacbase
    prob['Vi_13'] = 0.0
    prob['V_13dc'] = Vdcbase

    prob['Gen1.P_guess'] = -5.0e6
    prob['Gen3.P_guess'] = -2.0e6

    Pguess = 0.2e6
    prob['TX10.P_ac_guess'] = Pguess
    prob['TX10.P_dc_guess'] = -Pguess
    prob['TX11.P_ac_guess'] = Pguess
    prob['TX11.P_dc_guess'] = -Pguess
    prob['TX12.P_ac_guess'] = -Pguess
    prob['TX12.P_dc_guess'] = Pguess
    prob['TX13.P_ac_guess'] = -Pguess
    prob['TX13.P_dc_guess'] = Pguess

    st = time.time()
 
    prob.run_model()

    def print_phasor(name, x, y):
      r, phi = cmath.polar(complex(x,y))
      print(name, r/Vacbase, '<', math.degrees(phi))

    print_phasor('V1:', prob['Vr_1'], prob['Vi_1'])
    print_phasor('V2:', prob['Vr_2'], prob['Vi_2'])
    print_phasor('V3:', prob['Vr_3'], prob['Vi_3'])
    print('V4:', prob['V_4'][0]/Vdcbase)
    print('V5:', prob['V_5'][0]/Vdcbase)
    print('V6:', prob['V_6'][0]/Vdcbase)
    print_phasor('V7:', prob['Vr_7'], prob['Vi_7'])
    print_phasor('V8:', prob['Vr_8'], prob['Vi_8'])
    print_phasor('V9:', prob['Vr_9'], prob['Vi_9'])
    print_phasor('V10:', prob['Vr_10'], prob['Vi_10'])
    print_phasor('V11:', prob['Vr_11'], prob['Vi_11'])
    print_phasor('V12:', prob['Vr_12'], prob['Vi_12'])
    print_phasor('V13:', prob['Vr_13'], prob['Vi_13'])

    print("time", time.time() - st)


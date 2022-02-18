from zappy.LF_elements.bus import ACbus as Bus
from zappy.LF_elements.line import ACline as Line
from zappy.LF_elements.generator import ACgenerator as Generator
from zappy.LF_elements.load import ACload as Load

from openmdao.api import Group, IndepVarComp
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, NonlinearBlockGS

import math, cmath
import numpy as np

class Example(Group):

    """Example 2.8 in Electrical Engineering by Hambley""" 
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        par = self.add_subsystem('DesVars', IndepVarComp(), promotes=['*'])
        par.add_output('Vm_bus', 1.05*np.ones(nn), units='V')
        par.add_output('thetaV_bus', 0.0*np.ones(nn), units='deg')

        par.add_output('R12', np.array([0.02,0.01,0.015]), units='ohm')
        par.add_output('X12', np.array([0.04,0.05,0.03]), units='ohm')
        par.add_output('R13', np.array([0.01,0.02,0.025]), units='ohm')
        par.add_output('X13', np.array([0.03,0.02,0.015]) , units='ohm')
        par.add_output('R23', np.array([0.0125,0.015,0.02]), units='ohm')
        par.add_output('X23', np.array([0.025,0.03,0.015]), units='ohm')

        par.add_output('P2', np.array([2.566,2.45,2.6]), units='W')
        par.add_output('Q2', np.array([1.102,1.2,1.05]), units='V*A')
        par.add_output('P3', np.array([1.386,1.55,1.2]), units='W')
        par.add_output('Q3', np.array([0.452,0.6,0.5]), units='V*A')

        self.add_subsystem('Line12', Line(num_nodes=nn), promotes=[('R','R12'), ('X','X12'), ('Vr_in','Vr_1'), ('Vr_out','Vr_2'), 
                                                        ('Vi_in','Vi_1'), ('Vi_out','Vi_2'),
                                                        ('Ir_in','L12:Ir'), ('Ii_in','L12:Ii'),
                                                        ('Ir_out','L21:Ir'), ('Ii_out','L21:Ii')])
        self.add_subsystem('Line13', Line(num_nodes=nn), promotes=[('R','R13'), ('X','X13'), ('Vr_in','Vr_1'), ('Vr_out','Vr_3'), 
                                                        ('Vi_in','Vi_1'), ('Vi_out','Vi_3'),
                                                        ('Ir_in','L13:Ir'), ('Ii_in','L13:Ii'),
                                                        ('Ir_out','L31:Ir'), ('Ii_out','L31:Ii')])
        self.add_subsystem('Line23', Line(num_nodes=nn), promotes=[('R','R23'), ('X','X23'), ('Vr_in','Vr_2'), ('Vr_out','Vr_3'), 
                                                        ('Vi_in','Vi_2'), ('Vi_out','Vi_3'),
                                                        ('Ir_in','L23:Ir'), ('Ii_in','L23:Ii'),
                                                        ('Ir_out','L32:Ir'), ('Ii_out','L32:Ii')])

        self.add_subsystem('Gen1', Generator(num_nodes=nn, mode='Slack'), promotes=['Vm_bus', 'thetaV_bus', ('Vr_out','Vr_1'), ('Vi_out','Vi_1'),
                                                        ('Ir_out','LG1:Ir'),('Ii_out','LG1:Ii')])

        self.add_subsystem('Load2', Load(num_nodes=nn), promotes=[('P','P2'), ('Q','Q2'), ('Vr_in','Vr_2'), ('Vi_in','Vi_2'),
                                                        ('Ir_in','LL2:Ir'),('Ii_in','LL2:Ii')])
        self.add_subsystem('Load3', Load(num_nodes=nn), promotes=[('P','P3'), ('Q','Q3'), ('Vr_in','Vr_3'), ('Vi_in','Vi_3'),
                                                        ('Ir_in','LL3:Ir'),('Ii_in','LL3:Ii')])

        self.add_subsystem('Bus1', Bus(num_nodes=nn, lines=['L12', 'L13', 'LG1']), promotes=[('Vr', 'Vr_1'), ('Vi', 'Vi_1'), 
                                                        'L12:*', 'L13:*', 'LG1:*'])
        self.add_subsystem('Bus2', Bus(num_nodes=nn, lines=['L21', 'L23', 'LL2']), promotes=[('Vr', 'Vr_2'), ('Vi', 'Vi_2'), 
                                                        'L21:*', 'L23:*', 'LL2:*'])
        self.add_subsystem('Bus3', Bus(num_nodes=nn, lines=['L31', 'L32', 'LL3']), promotes=[('Vr', 'Vr_3'), ('Vi', 'Vi_3'), 
                                                        'L31:*', 'L32:*', 'LL3:*'])



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

    prob.model.add_subsystem('sys', Example(num_nodes=3), promotes=['*'])

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)
    prob.setup()
    # prob.final_setup()

    prob['Vr_1'] = 1.05*np.ones(3)
    prob['Vi_1'] = np.zeros(3)
    prob['Vr_2'] = 1.0*np.ones(3)
    prob['Vi_2'] = np.zeros(3)
    prob['Vr_3'] = 1.0*np.ones(3)
    prob['Vi_3'] = np.zeros(3)


    prob.run_model()

    def print_phasor(name, x, y):
        r, phi = cmath.polar(complex(x,y))
        print(name, r, '<', math.degrees(phi))


    print()
    print_phasor('V1:', prob['Vr_1'][0], prob['Vi_1'][0])
    print_phasor('V2:', prob['Vr_2'][0], prob['Vi_2'][0])
    print_phasor('V3:', prob['Vr_3'][0], prob['Vi_3'][0])
    print('V1:', prob['Vr_1'][0], prob['Vi_1'][0])
    print('V2:', prob['Vr_2'][0], prob['Vi_2'][0])
    print('V3:', prob['Vr_3'][0], prob['Vi_3'][0])
    print()
    print('LG1:I: ', prob['LG1:Ir'][0], prob['LG1:Ii'][0])
    print('LL2:I: ', prob['LL2:Ir'][0], prob['LL2:Ii'][0])
    print('LL3:I: ', prob['LL3:Ir'][0], prob['LL3:Ii'][0])
    print()
    print('Line12 S_in:', prob['Line12.P_in'][0], prob['Line12.Q_in'][0])
    print('Line12 S_out:', prob['Line12.P_out'][0], prob['Line12.Q_out'][0])
    print('Line13 S_in:', prob['Line13.P_in'][0], prob['Line13.Q_in'][0])
    print('Line13 S_out:', prob['Line13.P_out'][0], prob['Line13.Q_out'][0])
    print('Line23 S_in:', prob['Line23.P_in'][0], prob['Line23.Q_in'][0])
    print('Line23 S_out:', prob['Line23.P_out'][0], prob['Line23.Q_out'][0])
    print()
    print('Line12 S_loss:', prob['Line12.P_loss'][0], prob['Line12.Q_loss'][0])
    print('Line13 S_loss:', prob['Line13.P_loss'][0], prob['Line13.Q_loss'][0])
    print('Line23 loss:', prob['Line23.P_loss'][0], prob['Line23.Q_loss'][0])
    print()
    print('Gen Power: ', prob['Gen1.P_out'][0], prob['Gen1.Q_out'][0])
    print('------------------------------')

    print()
    print_phasor('V1:', prob['Vr_1'][1], prob['Vi_1'][1])
    print_phasor('V2:', prob['Vr_2'][1], prob['Vi_2'][1])
    print_phasor('V3:', prob['Vr_3'][1], prob['Vi_3'][1])
    print('V1:', prob['Vr_1'][1], prob['Vi_1'][1])
    print('V2:', prob['Vr_2'][1], prob['Vi_2'][1])
    print('V3:', prob['Vr_3'][1], prob['Vi_3'][1])
    print()
    print('LG1:I: ', prob['LG1:Ir'][1], prob['LG1:Ii'][1])
    print('LL2:I: ', prob['LL2:Ir'][1], prob['LL2:Ii'][1])
    print('LL3:I: ', prob['LL3:Ir'][1], prob['LL3:Ii'][1])
    print()
    print('Line12 S_in:', prob['Line12.P_in'][1], prob['Line12.Q_in'][1])
    print('Line12 S_out:', prob['Line12.P_out'][1], prob['Line12.Q_out'][1])
    print('Line13 S_in:', prob['Line13.P_in'][1], prob['Line13.Q_in'][1])
    print('Line13 S_out:', prob['Line13.P_out'][1], prob['Line13.Q_out'][1])
    print('Line23 S_in:', prob['Line23.P_in'][1], prob['Line23.Q_in'][1])
    print('Line23 S_out:', prob['Line23.P_out'][1], prob['Line23.Q_out'][1])
    print()
    print('Line12 S_loss:', prob['Line12.P_loss'][1], prob['Line12.Q_loss'][1])
    print('Line13 S_loss:', prob['Line13.P_loss'][1], prob['Line13.Q_loss'][1])
    print('Line23 loss:', prob['Line23.P_loss'][1], prob['Line23.Q_loss'][1])
    print()
    print('Gen Power: ', prob['Gen1.P_out'][1], prob['Gen1.Q_out'][1])
    print('------------------------------')


    print()
    print_phasor('V1:', prob['Vr_1'][2], prob['Vi_1'][2])
    print_phasor('V2:', prob['Vr_2'][2], prob['Vi_2'][2])
    print_phasor('V3:', prob['Vr_3'][2], prob['Vi_3'][2])
    print('V1:', prob['Vr_1'][2], prob['Vi_1'][2])
    print('V2:', prob['Vr_2'][2], prob['Vi_2'][2])
    print('V3:', prob['Vr_3'][2], prob['Vi_3'][2])
    print()
    print('LG1:I: ', prob['LG1:Ir'][2], prob['LG1:Ii'][2])
    print('LL2:I: ', prob['LL2:Ir'][2], prob['LL2:Ii'][2])
    print('LL3:I: ', prob['LL3:Ir'][2], prob['LL3:Ii'][2])
    print()
    print('Line12 S_in:', prob['Line12.P_in'][2], prob['Line12.Q_in'][2])
    print('Line12 S_out:', prob['Line12.P_out'][2], prob['Line12.Q_out'][2])
    print('Line13 S_in:', prob['Line13.P_in'][2], prob['Line13.Q_in'][2])
    print('Line13 S_out:', prob['Line13.P_out'][2], prob['Line13.Q_out'][2])
    print('Line23 S_in:', prob['Line23.P_in'][2], prob['Line23.Q_in'][2])
    print('Line23 S_out:', prob['Line23.P_out'][2], prob['Line23.Q_out'][2])
    print()
    print('Line12 S_loss:', prob['Line12.P_loss'][2], prob['Line12.Q_loss'][2])
    print('Line13 S_loss:', prob['Line13.P_loss'][2], prob['Line13.Q_loss'][2])
    print('Line23 loss:', prob['Line23.P_loss'][2], prob['Line23.Q_loss'][2])
    print()
    print('Gen Power: ', prob['Gen1.P_out'][2], prob['Gen1.Q_out'][2])
    print('------------------------------')

    print()



from zappy.NV_elements.node import Node
from zappy.NV_elements.resistor import Resistor

from openmdao.api import Group, IndepVarComp
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

class Example(Group):

    """Exercise 2.8 in Electrical Engineering by Hambley""" 

    def setup(self):

        par = self.add_subsystem('par', IndepVarComp(), promotes=['*'])
        par.add_output('V_0', 0.0, units='V')
        par.add_output('V_1', 10.0, units='V')
        par.add_output('R_1', 10.0, units='ohm')
        par.add_output('R_2', 2.0, units='ohm')
        par.add_output('R_3', 10.0, units='ohm')
        par.add_output('R_4', 5.0, units='ohm')
        par.add_output('R_5', 5.0, units='ohm')

        self.add_subsystem('N2', Node(connect_names=['c0', 'c1', 'c3']), promotes=[('V', 'V_2')])
        self.add_subsystem('N3', Node(connect_names=['c0', 'c1', 'c2']), promotes=[('V', 'V_3')])

        self.add_subsystem('R1', Resistor(), promotes=[('R', 'R_1'), ('V_in', 'V_1'), ('V_out', 'V_3')])
        self.add_subsystem('R2', Resistor(), promotes=[('R', 'R_2'), ('V_in', 'V_1'), ('V_out', 'V_2')])
        self.add_subsystem('R3', Resistor(), promotes=[('R', 'R_3'), ('V_in', 'V_2'), ('V_out', 'V_3')])
        self.add_subsystem('R4', Resistor(), promotes=[('R', 'R_4'), ('V_in', 'V_0'), ('V_out', 'V_2')])
        self.add_subsystem('R5', Resistor(), promotes=[('R', 'R_5'), ('V_in', 'V_0'), ('V_out', 'V_3')])

        self.connect('R2.I_out', 'N2.c1:I')
        self.connect('R3.I_in', 'N2.c3:I')
        self.connect('R4.I_out', 'N2.c0:I')
        self.connect('R3.I_out', 'N3.c2:I')
        self.connect('R1.I_out', 'N3.c1:I')
        self.connect('R5.I_out', 'N3.c0:I')

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        #
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1
        #
        self.linear_solver = DirectSolver()


if __name__ == "__main__":
    from openmdao.api import Problem

    prob = Problem()

    prob.model.add_subsystem('sys', Example(), promotes=['*'])

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=2)
    prob.setup(check=False)

    prob.run_model()

    print('V_0', prob['V_0'])
    print('V_1', prob['V_1'])
    print('V_2', prob['V_2'])
    print('V_3', prob['V_3'])

    print('I1', prob['R1.I_out'])
    print('I2', prob['R2.I_out'])
    print('I3', prob['R3.I_out'])
    print('I4', prob['R4.I_out'])
    print('I5', prob['R5.I_out'])

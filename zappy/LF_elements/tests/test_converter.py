import unittest
# import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, NonlinearBlockGS

from zappy.LF_elements.bus import ACbus, DCbus
from zappy.LF_elements.line import ACline
from zappy.LF_elements.generator import ACgenerator
from zappy.LF_elements.load import ACload, DCload
from zappy.LF_elements.converter import Converter


class ConverterTestCase1(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        par = self.prob.model.add_subsystem('par', IndepVarComp(), promotes=['*'])

        par.add_output('ac_line:Ir', -110.231724960752, units='A')
        par.add_output('ac_line:Ii', 42.585884570187, units='A')

        par.add_output('Ksc', 0.611764706, units=None) #0.612372436
        par.add_output('M', 0.99, units=None)
        par.add_output('eff', 0.882717253, units=None)
        par.add_output('PF', 0.953646228, units=None) #0.95072105

        par.add_output('P', 0.648706424, units='MW') 

        self.prob.model.add_subsystem('dc_load', DCload(num_nodes=1,), promotes=['P', ('V_in','V'), ('I_in','dc_line:I')])

        self.prob.model.add_subsystem('Conv', Converter(num_nodes=1,mode='Lead'), promotes=[('V_dc','V'), ('Vr_ac','Vr'), ('Vi_ac','Vi'),
                                                        ('I_dc','dc_side:I'), ('Ir_ac', 'ac_side:Ir'), ('Ii_ac', 'ac_side:Ii'),
                                                        'Ksc', 'M', 'eff', 'PF'])

        self.prob.model.add_subsystem('ac_bus', ACbus(num_nodes=1,lines=['ac_line', 'ac_side']), promotes=['Vr', 'Vi', 'ac_line:*', 'ac_side:*'])
        self.prob.model.add_subsystem('dc_bus', DCbus(num_nodes=1,lines=['dc_line', 'dc_side']), promotes=['V', 'dc_line:*', 'dc_side:*'])

        newton = self.prob.model.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 100
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        
        newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['maxiter'] = 20
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['print_bound_enforce'] = True
        newton.linesearch.options['iprint'] = -1

        self.prob.model.linear_solver = DirectSolver()

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        self.prob['ac_line:Ir'] = -155.222738488575
        self.prob['ac_line:Ii'] = 63.5689869774839
        self.prob['dc_line:I'] = 94.5929129
        self.prob['M'] = 0.99
        self.prob['eff'] = 0.98
        self.prob['PF'] = 0.95
        self.prob['Vr'] = 4160.0 #4099.72489622173
        self.prob['Vi'] = -250.0 #-291.570543264784
        self.prob['V'] = 6800. #6786.262626
        self.prob['ac_side:Ir'] = 155.198791057753
        self.prob['ac_side:Ii'] = -63.5811739175523
        self.prob['dc_side:I'] = -94.5881459871397
        self.prob['P'] = 0.641932349

        self.prob.run_model()

        tol = 5e-3

        assert_rel_error(self, self.prob['Vr'], 4099.72489622173, tol)
        assert_rel_error(self, self.prob['Vi'], -291.570543264784, tol)
        assert_rel_error(self, self.prob['V'], 6786.26262626263, tol)
        assert_rel_error(self, self.prob['Conv.P_ac'], 0.655*1e6, tol)
        assert_rel_error(self, self.prob['Conv.Q_ac'], 0.2153*1e6, tol)
        assert_rel_error(self, self.prob['Conv.P_dc'], -0.6419*1e6, tol)

    def test_case2(self):

        self.prob['ac_line:Ir'] = -110.296177987161
        self.prob['ac_line:Ii'] = 44.1329942107636
        self.prob['dc_line:I'] = 66.9658918406072
        self.prob['M'] = 0.99
        self.prob['eff'] = 0.98
        self.prob['PF'] = 0.95
        self.prob['Vr'] = 4160.0 #4118.53338714101
        self.prob['Vi'] = 0.0 #-259.808963288064
        self.prob['V'] = 6800.0 #6813.737374
        self.prob['ac_side:Ir'] = 110.296177987161
        self.prob['ac_side:Ii'] = -44.1329942107636
        self.prob['dc_side:I'] = -66.9658918406072
        self.prob['P'] = 0.456288

        self.prob.run_model()

        tol = 1e-3

        assert_rel_error(self, self.prob['Vr'], 4118.53338714101, tol)
        assert_rel_error(self, self.prob['Vi'], -259.808963288064, tol)
        assert_rel_error(self, self.prob['V'], 6813.73737373737, tol)
        assert_rel_error(self, self.prob['Conv.P_ac'], 0.4656*1e6, tol)
        assert_rel_error(self, self.prob['Conv.Q_ac'], 0.153*1e6, tol)
        assert_rel_error(self, self.prob['Conv.P_dc'], -0.456288*1e6, tol)

class ConverterTestCase2(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        par = self.prob.model.add_subsystem('par', IndepVarComp(), promotes=['*'])

        par.add_output('ac_line:Ir', -110.231724960752, units='A')
        par.add_output('ac_line:Ii', 42.585884570187, units='A')

        par.add_output('Ksc', 0.611764706, units=None) #0.612372436
        par.add_output('M', 0.99, units=None)
        par.add_output('eff', 0.882717253, units=None)
        par.add_output('PF', 0.953646228, units=None) #0.95072105

        par.add_output('P', 0.648706424, units='MW') 

        self.prob.model.add_subsystem('dc_load', DCload(num_nodes=1, ), promotes=['P', ('V_in','V'), ('I_in','dc_line:I')])

        self.prob.model.add_subsystem('Conv', Converter(num_nodes=1, mode='Lag'), promotes=[('V_dc','V'), ('Vr_ac','Vr'), ('Vi_ac','Vi'),
                                                        ('I_dc','dc_side:I'), ('Ir_ac', 'ac_side:Ir'), ('Ii_ac', 'ac_side:Ii'),
                                                        'Ksc', 'M', 'eff', 'PF'])

        self.prob.model.add_subsystem('ac_bus', ACbus(num_nodes=1, lines=['ac_line', 'ac_side']), promotes=['Vr', 'Vi', 'ac_line:*', 'ac_side:*'])
        self.prob.model.add_subsystem('dc_bus', DCbus(num_nodes=1, lines=['dc_line', 'dc_side']), promotes=['V', 'dc_line:*', 'dc_side:*'])

        newton = self.prob.model.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        
        newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['maxiter'] = 20
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['print_bound_enforce'] = True
        newton.linesearch.options['iprint'] = -1

        self.prob.model.linear_solver = DirectSolver()

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case3(self):

        self.prob['ac_line:Ir'] = 151.172584807581
        self.prob['ac_line:Ii'] = -64.7258926104048
        self.prob['dc_line:I'] = -94.5881459871397
        self.prob['M'] = 0.97
        self.prob['eff'] = 0.98
        self.prob['PF'] = -0.95
        self.prob['Vr'] = 4160.0 #3961.91888370781
        self.prob['Vi'] = 0.0 #-345.556942513739
        self.prob['V'] = 6800.0 #6701.85567
        self.prob['ac_side:Ir'] = -151.172584807581
        self.prob['ac_side:Ii'] = 64.7258926104048
        self.prob['dc_side:I'] = 94.5881459871397
        self.prob['P'] = -0.63398

        self.prob.run_model()

        tol = 1e-3

        assert_rel_error(self, self.prob['Vr'], 3961.91888370781, tol)
        assert_rel_error(self, self.prob['Vi'], -345.556942513739, tol)
        assert_rel_error(self, self.prob['V'], 6701.85567010309, tol)
        assert_rel_error(self, self.prob['Conv.P_ac'], -0.6213*1e6, tol)
        assert_rel_error(self, self.prob['Conv.Q_ac'], -0.2042*1e6, tol)
        assert_rel_error(self, self.prob['Conv.P_dc'], 0.63398*1e6, tol)

    def test_case4(self):

        self.prob['ac_line:Ir'] = 74.7289366656989
        self.prob['ac_line:Ii'] = -33.2594117436185
        self.prob['dc_line:I'] = -46.5644832065318
        self.prob['M'] = 0.96
        self.prob['eff'] = 0.98
        self.prob['PF'] = -0.95
        self.prob['Vr'] = 4160.0 #3935.99857051845
        self.prob['Vi'] = 0.0 #-398.89497173656
        self.prob['V'] = 6800.0 #6736.25
        self.prob['ac_side:Ir'] = -74.7289366656989
        self.prob['ac_side:Ii'] = 33.2594117436185
        self.prob['dc_side:I'] = 46.5644832065318
        self.prob['P'] = -0.31367

        self.prob.run_model()

        tol = 5e-3

        assert_rel_error(self, self.prob['Vr'], 3935.99857051845, tol)
        assert_rel_error(self, self.prob['Vi'], -398.894971736559, tol)
        assert_rel_error(self, self.prob['V'], 6736.25, tol)
        assert_rel_error(self, self.prob['Conv.P_ac'], -0.3074*1e6, tol)
        assert_rel_error(self, self.prob['Conv.Q_ac'], -0.1011*1e6, tol)
        assert_rel_error(self, self.prob['Conv.P_dc'], 0.31367*1e6, tol)

if __name__ == "__main__":
    unittest.main()




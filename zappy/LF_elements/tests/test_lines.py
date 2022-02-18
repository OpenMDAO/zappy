import unittest
# import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from zappy.LF_elements.line import ACline, DCline


class AClineTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

        des_vars.add_output('R', 0.2218, units='ohm')
        des_vars.add_output('X', 0.3630, units='ohm')
        des_vars.add_output('Vr_in', 1.000000784, units='V')  
        des_vars.add_output('Vi_in', -0.049999948, units='V')  
        des_vars.add_output('Vr_out', 1.000000784, units='V')  
        des_vars.add_output('Vi_out', -0.049999948, units='V')  

        self.prob.model.add_subsystem('acline', ACline(num_nodes=1), promotes=['*'])

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        self.prob['R'] = 0.2218
        self.prob['X'] = 0.3630
        self.prob['Vr_in'] = 4368.0
        self.prob['Vi_in'] = 0.0
        self.prob['Vr_out'] = 4211.34943357403
        self.prob['Vi_out'] = -151.677930945098

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 496.25376022551, tol)
        assert_rel_error(self, self.prob['Ii_in'], -128.323642997125, tol)
        assert_rel_error(self, self.prob['Ir_out'], -496.25376022551, tol)
        assert_rel_error(self, self.prob['Ii_out'], 128.323642997125, tol)

        assert_rel_error(self, self.prob['P_in'], 2.167636425*1e6, tol)
        assert_rel_error(self, self.prob['P_out'], -2.109361857*1e6, tol)
        assert_rel_error(self, self.prob['P_loss'], 0.058274568*1e6, tol)

        assert_rel_error(self, self.prob['Q_in'], 0.560517673*1e6, tol)
        assert_rel_error(self, self.prob['Q_out'], -0.465144958*1e6, tol)
        assert_rel_error(self, self.prob['Q_loss'], 0.095372715*1e6, tol)

    def test_case2(self):

        self.prob['R'] = 0.2218
        self.prob['X'] = 0.3630
        self.prob['Vr_in'] = 4368.0
        self.prob['Vi_in'] = 0.0
        self.prob['Vr_out'] = 4172.22191365666
        self.prob['Vi_out'] = -192.057264384983

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 625.20841975576, tol)
        assert_rel_error(self, self.prob['Ii_in'], -157.319170362297, tol)
        assert_rel_error(self, self.prob['Ir_out'], -625.20841975576, tol)
        assert_rel_error(self, self.prob['Ii_out'], 157.319170362297, tol)

        assert_rel_error(self, self.prob['P_in'], 2.730910377*1e6, tol)
        assert_rel_error(self, self.prob['P_out'], -2.638722559*1e6, tol)
        assert_rel_error(self, self.prob['P_loss'], 0.092187818*1e6, tol)

        assert_rel_error(self, self.prob['Q_in'], 0.687170136*1e6, tol)
        assert_rel_error(self, self.prob['Q_out'], -0.536294671*1e6, tol)
        assert_rel_error(self, self.prob['Q_loss'], 0.150875465*1e6, tol)


class DClineTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

        des_vars.add_output('R', 0.2208, units='ohm')
        des_vars.add_output('V_in', 1.000000784, units='V')  
        des_vars.add_output('V_out', 1.000000784, units='V')  

        self.prob.model.add_subsystem('dcline', DCline(num_nodes=1), promotes=['*'])

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        self.prob['R'] = 0.2208
        self.prob['V_in'] = 6779.6
        self.prob['V_out'] = 6800.0

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['I_in'], -92.39130435, tol)
        assert_rel_error(self, self.prob['I_out'], 92.39130435, tol)

        assert_rel_error(self, self.prob['P_in'], -0.626376087*1e6, tol)
        assert_rel_error(self, self.prob['P_out'], 0.62826087*1e6, tol)
        assert_rel_error(self, self.prob['P_loss'], 0.001884783*1e6, tol)


    def test_case2(self):

        self.prob['R'] = 0.2208
        self.prob['V_in'] = 6800.0
        self.prob['V_out'] = 6759.2

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['I_in'], 184.7826087, tol)
        assert_rel_error(self, self.prob['I_out'], -184.7826087, tol)

        assert_rel_error(self, self.prob['P_in'], 1.256521739*1e6, tol)
        assert_rel_error(self, self.prob['P_out'], -1.248982609*1e6, tol)
        assert_rel_error(self, self.prob['P_loss'], 0.00753913*1e6, tol)



if __name__ == "__main__":
    unittest.main()




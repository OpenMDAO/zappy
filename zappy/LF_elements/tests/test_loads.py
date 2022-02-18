import unittest
# import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from zappy.LF_elements.load import ACload, DCload


class ACloadTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

        des_vars.add_output('P', 1.386, units='W')
        des_vars.add_output('Q', 0.452, units='V*A')
        des_vars.add_output('Vr_in', 1.000000784, units='V')  #1.05
        des_vars.add_output('Vi_in', -0.049999948, units='V')  #0.0

        self.prob.model.add_subsystem('acload', ACload(num_nodes=1), promotes=['*'])

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        self.prob['P'] = 1.977871614*1e6
        self.prob['Q'] = 0.388220063*1e6
        self.prob['Vr_in'] = 4211.34943357403
        self.prob['Vi_in'] = -151.677930945098

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 465.72840849992, tol)
        assert_rel_error(self, self.prob['Ii_in'], -108.958135945842, tol)

    def test_case2(self):

        self.prob['P'] = 2.486276439*1e6
        self.prob['Q'] = 0.523403576*1e6
        self.prob['Vr_in'] = 4156.15473035497
        self.prob['Vi_in'] = -178.823536896166

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 591.701686399864, tol)
        assert_rel_error(self, self.prob['Ii_in'], -151.393248131652, tol)

    def test_case3(self):

        self.prob['P'] = 2.5*1e6
        self.prob['Q'] = 0.5*1e6
        self.prob['Vr_in'] = 3908.91265712629
        self.prob['Vi_in'] = -455.477956171222

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 616.29151654242, tol)
        assert_rel_error(self, self.prob['Ii_in'], -199.724902764739, tol)

    def test_case4(self):

        self.prob['P'] = 1.0*1e6
        self.prob['Q'] = 0.1*1e6
        self.prob['Vr_in'] = 4154.37687377418
        self.prob['Vi_in'] = -216.223936348974

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 238.810239468108, tol)
        assert_rel_error(self, self.prob['Ii_in'], -36.5004174213199, tol)

    def test_case5(self):

        self.prob['P'] = 2.5*1e6
        self.prob['Q'] = 0.5*1e6
        self.prob['Vr_in'] = 4172.22191365666
        self.prob['Vi_in'] = -192.057264384983

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['Ir_in'], 592.429234901721, tol)
        assert_rel_error(self, self.prob['Ii_in'], -147.111143869857, tol)

class DCloadTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

        des_vars.add_output('P', 1.386, units='W')
        des_vars.add_output('V_in', 1.000000784, units='V')  #1.05

        self.prob.model.add_subsystem('dcload', DCload(num_nodes=1), promotes=['*'])

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        self.prob['P'] = 1.0*1e6
        self.prob['V_in'] = 6779.6

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['I_in'], 147.5013275, tol)


    def test_case2(self):

        self.prob['P'] = 2.0*1e6
        self.prob['V_in'] = 6759.2

        self.prob.run_model()

        tol = 1e-4

        assert_rel_error(self, self.prob['I_in'], 295.8930051, tol)


if __name__ == "__main__":
    unittest.main()




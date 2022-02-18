import math, cmath
import numpy as np

from openmdao.api import ImplicitComponent

class ACbus(ImplicitComponent):
    """
    Determines the voltage of an AC bus
    """

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('lines', default=['1', '2'], desc='Names of electrical lines connecting to the bus')
        self.options.declare('Vbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        lines = self.options['lines']
        Ibase = self.options['Sbase']/self.options['Vbase']
        ar = np.arange(nn)

        self.add_output('Vr', val=np.ones(nn), units='V', desc='Voltage (real) of the bus',
                                res_ref=Ibase, res_units='A')
        self.add_output('Vi', val=np.zeros(nn), units='V', desc='Voltage (imaginary) of the bus',
                                res_ref=Ibase, res_units='A')
        
        for name in lines:
            self.add_input(name+':Ir', val=np.zeros(nn), units='A', desc='Current (real) of line '+name)
            self.add_input(name+':Ii', val=np.zeros(nn), units='A', desc='Current (imaginary) of line '+name)

            self.declare_partials('Vr', name+':Ir', rows=ar, cols=ar, val=1.0)
            self.declare_partials('Vi', name+':Ii', rows=ar, cols=ar, val=1.0)

    def guess_nonlinear(self, inputs, outputs, resids):

        outputs['Vr'] = self.options['Vbase']
        outputs['Vi'] = 0.0

    def apply_nonlinear(self, inputs, outputs, resids):

        lines = self.options['lines']
        resids['Vr'] = 0.0
        resids['Vi'] = 0.0
            
        for name in lines:
            resids['Vr'] += inputs[name+':Ir']
            resids['Vi'] += inputs[name+':Ii']

    def linearize(self, inputs, outputs, J):

        pass

class DCbus(ImplicitComponent):
    """
    Determines the voltage of a DC bus
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('lines', default=['1', '2'], desc='names of electrical lines connecting to the bus')
        self.options.declare('Vbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        lines = self.options['lines']
        Ibase = self.options['Sbase']/self.options['Vbase']
        ar = np.arange(nn)

        self.add_output('V', val=np.ones(nn), units='V', desc='Voltage of the bus',
                                res_ref=Ibase, res_units='A')
        
        for name in lines:
            self.add_input(name+':I', val=np.zeros(nn), units='A', desc='Current of line '+name)

            self.declare_partials('V', name+':I', rows=ar, cols=ar, val=1.0)

    def guess_nonlinear(self, inputs, outputs, resids):

        outputs['V'] = self.options['Vbase']

    def apply_nonlinear(self, inputs, outputs, resids):

        lines = self.options['lines']
        resids['V'] = 0.0
            
        for name in lines:
            resids['V'] += inputs[name+':I']

    def linearize(self, inputs, outputs, J):

        pass


if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('In1:Ir', 1.90003278522448*np.ones(3), units='A')
    des_vars.add_output('In1:Ii', 0.800107961803713*np.ones(3), units='A')
    des_vars.add_output('In2:Ir', 1.99999059394351*np.ones(3), units='A')
    des_vars.add_output('In2:Ii', 0.999977006616166*np.ones(3), units='A')
    des_vars.add_output('Out1:Ir', -3.9*np.ones(3), units='A')
    des_vars.add_output('Out1:Ii', -1.8*np.ones(3), units='A')

    p.model.add_subsystem('acbus', ACbus(num_nodes=3, lines=['In1', 'In2', 'Out1']), promotes=['*'])

    des_vars.add_output('In1:I', 1.90003278522448*np.ones(3), units='A')
    des_vars.add_output('In2:I', 1.99999059394351*np.ones(3), units='A')
    des_vars.add_output('Out1:I', -3.9*np.ones(3), units='A')

    p.model.add_subsystem('dcbus', DCbus(num_nodes=3, lines=['In1', 'In2', 'Out1']), promotes=['*'])


    p.setup(check=False)

    p.check_partials(compact_print=False)



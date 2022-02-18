import math, cmath
import numpy as np

from openmdao.api import ExplicitComponent

class ACload(ExplicitComponent):
    """
    Calculates the current required by an AC load
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('P', val=np.zeros(nn), units='W', desc='Real power of the load')
        self.add_input('Q', val=np.zeros(nn), units='V*A', desc='Reactive power of the load')
        self.add_input('Vr_in', val=np.ones(nn), units='V', desc='Voltage (real) of the bus supplying power')
        self.add_input('Vi_in', val=np.ones(nn), units='V', desc='Voltage (imaginary) of the bus supplying power')

        self.add_output('Ir_in', val=np.ones(nn), units='A', desc='Current (real) entering the load')
        self.add_output('Ii_in', val=np.zeros(nn), units='A', desc='Current (imaginary) entering the load')

        ar = np.arange(nn)

        self.declare_partials('Ir_in','P', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Q', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vi_in', rows=ar, cols=ar)

        self.declare_partials('Ii_in','P', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Q', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vi_in', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        V = inputs['Vr_in'] + inputs['Vi_in']*1j
        S = inputs['P'] + inputs['Q']*1j
        I = (S/V).conjugate()

        outputs['Ir_in'] = I.real
        outputs['Ii_in'] = I.imag

    def compute_partials(self, inputs, J):

        S = inputs['P'] + inputs['Q']*1j
        V = inputs['Vr_in'] + inputs['Vi_in']*1j

        J['Ir_in', 'P'] = (1./V.conjugate()).real
        J['Ii_in', 'P'] = (1./V.conjugate()).imag

        J['Ir_in', 'Q'] = (-1j/V.conjugate()).real
        J['Ii_in', 'Q'] = (-1j/V.conjugate()).imag

        J['Ir_in', 'Vr_in'] = (-S.conjugate()/V.conjugate()**2).real
        J['Ii_in', 'Vr_in'] = (-S.conjugate()/V.conjugate()**2).imag

        J['Ir_in', 'Vi_in'] = (1j*S.conjugate()/V.conjugate()**2).real
        J['Ii_in', 'Vi_in'] = (1j*S.conjugate()/V.conjugate()**2).imag

class DCload(ExplicitComponent):
    """
    Calculates the current required by a DC load
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('P', val=np.zeros(nn), units='W', desc='Real power of the load')
        self.add_input('V_in', val=np.ones(nn), units='V', desc='Voltage of the bus supplying power')

        self.add_output('I_in', val=np.zeros(nn), units='A', desc='Current entering the load')

        ar = np.arange(nn)

        self.declare_partials('I_in','P', rows=ar, cols=ar)
        self.declare_partials('I_in','V_in', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        outputs['I_in'] = inputs['P'] / inputs['V_in']

    def compute_partials(self, inputs, J):

        J['I_in', 'P'] = 1./inputs['V_in']
        J['I_in', 'V_in'] = -inputs['P'] / inputs['V_in']**2


if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('P', 1.386*np.ones(3), units='W')
    des_vars.add_output('Q', 0.452*np.ones(3), units='V*A')
    des_vars.add_output('Vr_in', 1.000000784*np.ones(3), units='V')  #1.05
    des_vars.add_output('Vi_in', -0.049999948*np.ones(3), units='V')  #0.0
    p.model.add_subsystem('acload', ACload(num_nodes=3), promotes_inputs=['*'])

    des_vars.add_output('V_in', 1.0113628476*np.ones(3), units='V')  #1.05
    p.model.add_subsystem('dcload', DCload(num_nodes=3), promotes_inputs=['*'])


    p.setup(check=False)
    p.run_model()

    print('Ir_in', p['acload.Ir_in'])
    print('Ii_in', p['acload.Ii_in'])

    print('I_in', p['dcload.I_in'])

    # p.check_partials()
    p.check_partials(compact_print=False)
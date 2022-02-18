import math, cmath
import numpy as np

from openmdao.api import ImplicitComponent

class ACgenerator(ImplicitComponent):
    """
    Determines the current supplied by an AC generator
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Slack', desc='Type of generator: Slack or P-V')

        self.options.declare('Q_min', allow_none=True, default=None, desc='Lower bound for reactive power (Q)')
        self.options.declare('Q_max', allow_none=True, default=None, desc='Upper bound for reactive power (Q)')

        self.options.declare('Vbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        mode = self.options['mode']
        if not (mode=="Slack" or mode=="P-V"):
            raise ValueError("mode must be 'Slack' or 'P-V', but '{}' was given.".format(mode))

        Vbase = self.options['Vbase']
        Sbase = self.options['Sbase']

        self.add_input('Vm_bus', val=np.ones(nn), units='V', desc='Voltage magnitude of the generator')
        self.add_input('Vr_out', val=np.ones(nn), units='V', desc='Voltage (real) of the bus receiving power')
        self.add_input('Vi_out', val=np.zeros(nn), units='V', desc='Voltage (imaginary) of the bus receiving power')

        self.add_output('Ir_out', val=np.ones(nn), units='A', desc='Current (real) sent to the bus',
                                res_ref=Vbase, res_units='V')

        # self.add_output('Ii_out', val=1.0, units='A', desc='Current (imaginary) sent to the bus')

        self.add_output('P_out', val=-np.ones(nn), units='W', desc='Real (active) power entering the line',
                                res_ref=Sbase, res_units='W')
        self.add_output('Q_out', val=-np.ones(nn), units='V*A', lower=self.options['Q_min'],
                                upper=self.options['Q_max'], desc='Reactive power entering the line',
                                res_ref=Sbase, res_units='W')

        self.declare_partials('P_out', 'Vr_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'Vi_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'Ir_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'Ii_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'P_out', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('Q_out', 'Vr_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Vi_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Ir_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Ii_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Q_out', rows=ar, cols=ar, val=-1.0)

        if mode == 'Slack':
            self.add_input('thetaV_bus', val=np.zeros(nn), units='deg', desc='Voltage phase angle of the generator')
            self.add_output('Ii_out', val=np.ones(nn), units='A', desc='Current (imaginary) sent to the bus')
            self.add_input('P_guess', val=-1.0e6*np.ones(nn), units='W', desc='Guess for power output of generator')

            self.declare_partials('Ir_out', 'Vm_bus', rows=ar, cols=ar, val=1.0)
            self.declare_partials('Ii_out', 'thetaV_bus', rows=ar, cols=ar, val=1.0)
            self.declare_partials('Ir_out', 'Vr_out', rows=ar, cols=ar)
            self.declare_partials('Ir_out', 'Vi_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Vr_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Vi_out', rows=ar, cols=ar)

        elif mode == 'P-V':
            self.add_input('P_bus', val=np.ones(nn), units='W', desc='Real power of the generator supplied to the bus')
            self.add_output('Ii_out', val=np.ones(nn), units='A', desc='Current (imaginary) sent to the bus',
                                res_ref=Sbase, res_units='W')

            self.declare_partials('Ir_out', 'Vm_bus', rows=ar, cols=ar, val=1.0)
            self.declare_partials('Ii_out', 'P_bus', rows=ar, cols=ar, val=1.0)
            self.declare_partials('Ir_out', 'Vr_out', rows=ar, cols=ar)
            self.declare_partials('Ir_out', 'Vi_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Vr_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Vi_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Ir_out', rows=ar, cols=ar)
            self.declare_partials('Ii_out', 'Ii_out', rows=ar, cols=ar)


    def apply_nonlinear(self, inputs, outputs, resids):

        mode = self.options['mode']
        # Vbase = self.options['Vbase']
        # Sbase = self.options['Sbase']

        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j

        I_out = outputs['Ir_out'] + outputs['Ii_out']*1j
        S_out = V_out*I_out.conjugate()

        resids['Ir_out'] = inputs['Vm_bus'] - abs(V_out)

        resids['P_out'] = S_out.real - outputs['P_out']
        resids['Q_out'] = S_out.imag - outputs['Q_out']

        if mode == 'Slack':
            resids['Ii_out'] = inputs['thetaV_bus'] - np.degrees(np.arctan2(V_out.imag, V_out.real))

        elif mode == 'P-V':
            resids['Ii_out'] = inputs['P_bus'] - S_out.real 

    def solve_nonlinear(self, inputs, outputs):

        # mode = self.options['mode']

        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j
        I_out = outputs['Ir_out'] + outputs['Ii_out']*1j
        S_out = V_out*I_out.conjugate()

        outputs['P_out'] = S_out.real
        outputs['Q_out'] = S_out.imag

        # if mode == 'P-V':
        #     outputs['Ii_out'] = inputs['P_bus']/complex(inputs['Vr_out'], inputs['Vi_out'])

    def guess_nonlinear(self, inputs, outputs, resids):

        mode = self.options['mode']

        if mode == 'Slack':
            S_guess = inputs['P_guess'] + inputs['P_guess']*(1.0/0.95**2-1)**0.5 * 1j

        elif mode == 'P-V':
            S_guess = inputs['P_bus'] + inputs['P_bus']*(1.0/0.95**2-1)**0.5 * 1j

        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j
        I = (S_guess/V_out).conjugate()

        outputs['Ir_out'] = I.real
        outputs['Ii_out'] = I.imag
        outputs['P_out'] = S_guess.real
        outputs['Q_out'] = S_guess.imag

    def linearize(self, inputs, outputs, J):

        mode = self.options['mode']

        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j
        I_out = outputs['Ir_out'] + outputs['Ii_out']*1j

        J['Ir_out', 'Vr_out'] = -inputs['Vr_out']/abs(V_out)
        J['Ir_out', 'Vi_out'] = -inputs['Vi_out']/abs(V_out)

        J['P_out', 'Vr_out'] = (I_out.conjugate()).real
        J['P_out', 'Vi_out'] = (1j*I_out.conjugate()).real
        J['P_out', 'Ir_out'] = V_out.real
        J['P_out', 'Ii_out'] = (-1j*V_out).real
        J['P_out', 'P_out'] = -1.0

        J['Q_out', 'Vr_out'] = (I_out.conjugate()).imag
        J['Q_out', 'Vi_out'] = (1j*I_out.conjugate()).imag
        J['Q_out', 'Ir_out'] = V_out.imag
        J['Q_out', 'Ii_out'] = (-1j*V_out).imag
        J['Q_out', 'Q_out'] = -1.0

        if mode == 'Slack':
            J['Ii_out', 'Vr_out'] = np.degrees(inputs['Vi_out']/abs(V_out)**2)
            J['Ii_out', 'Vi_out'] = np.degrees(-inputs['Vr_out']/abs(V_out)**2)

        elif mode == 'P-V':
            J['Ii_out', 'Vr_out'] = -(I_out.conjugate()).real
            J['Ii_out', 'Vi_out'] = -(1j*I_out.conjugate()).real
            J['Ii_out', 'Ir_out'] = -inputs['Vr_out']
            J['Ii_out', 'Ii_out'] = -inputs['Vi_out']

class DCgenerator(ImplicitComponent):
    """
    Determines the current supplied by a DC generator
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('P_min', allow_none=True, default=None, desc='Lower bound for active power (P)')
        self.options.declare('P_max', allow_none=True, default=None, desc='Upper bound for active power (P)')

        self.options.declare('Vbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        Vbase = self.options['Vbase']
        Sbase = self.options['Sbase']

        self.add_input('V_bus', val=np.ones(nn), units='V', desc='Voltage magnitude of the generator')
        self.add_input('V_out', val=np.ones(nn), units='V', desc='Voltage of the bus receiving power')

        self.add_output('I_out', val=-np.ones(nn), units='A', desc='Current sent to the bus',
                                res_ref=Vbase, res_units='V')

        self.add_output('P_out', val=-np.ones(nn), units='W', lower=self.options['P_min'],
                                upper=self.options['P_max'], desc='Real (active) power entering the line',
                                res_ref=Sbase, res_units='W')

        self.add_input('P_guess', val=-1.0e6*np.ones(nn), units='W', desc='Guess for power output of generator')

        self.declare_partials('I_out', 'V_bus', rows=ar, cols=ar, val=1.0)
        self.declare_partials('I_out', 'V_out', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('P_out', 'V_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'I_out', rows=ar, cols=ar)
        self.declare_partials('P_out', 'P_out', rows=ar, cols=ar, val=-1.0)


    def apply_nonlinear(self, inputs, outputs, resids):

        resids['I_out'] = inputs['V_bus'] - inputs['V_out']
        resids['P_out'] = inputs['V_out'] * outputs['I_out'] - outputs['P_out']

    def solve_nonlinear(self, inputs, outputs):

        outputs['P_out'] = inputs['V_out'] * outputs['I_out']

    def guess_nonlinear(self, inputs, outputs, resids):

        outputs['I_out'] = inputs['P_guess'] / inputs['V_out']
        outputs['P_out'] = inputs['P_guess']

    def linearize(self, inputs, outputs, J):

        J['P_out', 'V_out'] = outputs['I_out']
        J['P_out', 'I_out'] = inputs['V_out']

if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('Vm_bus', 4368.*np.ones(3), units='V')
    des_vars.add_output('thetaV_bus', 0.0*np.ones(3), units='deg')
    des_vars.add_output('Vr_out', 4300.*np.ones(3), units='V')
    des_vars.add_output('Vi_out', 100.0*np.ones(3), units='V')
    des_vars.add_output('P_bus', 5.0*1e6*np.ones(3), units='W')

    p.model.add_subsystem('acgen', ACgenerator(num_nodes=3, mode='P-V'), promotes_inputs=['*'])

    des_vars.add_output('V_bus', 1.05*np.ones(3), units='V')
    des_vars.add_output('V_out', 1.0*np.ones(3), units='V')

    p.model.add_subsystem('dcgen', DCgenerator(num_nodes=3), promotes_inputs=['*'])

    p.setup(check=False)

    p['acgen.Ir_out'] = 1121.46217998127*np.ones(3)
    p['acgen.Ii_out'] = -285.642813359423*np.ones(3)

    p.check_partials(compact_print=False)
import math, cmath
import numpy as np

from openmdao.api import ImplicitComponent

class Converter(ImplicitComponent):
    """
    Determines the flow through a converter
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Lead', values=['Lead', 'Lag'], desc='Specifies weather AC currentl leads or lags the voltage')

        self.options.declare('Vdcbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        Vbase = self.options['Vdcbase']
        Sbase = self.options['Sbase']

        self.add_input('V_dc', val=np.ones(nn), units='V', desc='Voltage on the DC side of the converter')
        self.add_input('Vr_ac', val=np.ones(nn), units='V', desc='Voltage (real) on the AC side of the converter')
        self.add_input('Vi_ac', val=np.zeros(nn), units='V', desc='Voltage (imaginary) on the AC side of the converter')

        self.add_input('Ksc', val=np.ones(nn), units=None, desc='Converter constant')
        self.add_input('M', val=np.ones(nn), units=None, desc='Converter modulation index')
        self.add_input('eff', val=np.ones(nn), units=None, desc='Converter efficiency')
        self.add_input('PF', val=np.ones(nn), units=None, desc='Converter power factor')

        self.add_output('I_dc', val=-np.ones(nn), units='A', desc='Current sent to the DC bus',
                                res_ref=Vbase, res_units='V')
        self.add_output('Ir_ac', val=np.ones(nn), units='A', desc='Current (real) sent to the AC bus',
                                res_ref=Sbase, res_units='W')
        self.add_output('Ii_ac', val=np.ones(nn), units='A', desc='Current (imaginary) sent to the AC bus')

        self.add_output('P_dc', val=np.zeros(nn), units='W', desc='Power entering the DC bus',
                                res_ref=Sbase, res_units='W')
        self.add_output('P_ac', val=np.zeros(nn), units='W', desc='Real power entering the AC bus',
                                res_ref=Sbase, res_units='W')
        self.add_output('Q_ac', val=np.zeros(nn), units='V*A', desc='Reactive power entering the AC bus',
                                res_ref=Sbase, res_units='W')

        self.add_input('P_ac_guess', val=-1.0e6*np.ones(nn), units='W', desc='Guess for AC power')
        self.add_input('P_dc_guess', val=-1.0e6*np.ones(nn), units='W', desc='Guess for DC power')

        self.declare_partials('I_dc', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('I_dc', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('I_dc', 'Ksc', rows=ar, cols=ar)
        self.declare_partials('I_dc',  'M', rows=ar, cols=ar)
        self.declare_partials('I_dc', 'V_dc', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'Ir_ac', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'Ii_ac', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'V_dc', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'I_dc', rows=ar, cols=ar)
        self.declare_partials('Ir_ac', 'eff', rows=ar, cols=ar)
        self.declare_partials('Ii_ac', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('Ii_ac', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('Ii_ac', 'Ir_ac', rows=ar, cols=ar)
        self.declare_partials('Ii_ac', 'Ii_ac', rows=ar, cols=ar)
        self.declare_partials('Ii_ac', 'PF', rows=ar, cols=ar)

        self.declare_partials('P_dc', 'V_dc', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'I_dc', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'P_dc', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('P_ac', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'Ir_ac', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'Ii_ac', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'P_ac', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('Q_ac', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Ir_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Ii_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Q_ac', rows=ar, cols=ar, val=-1.0)

    def apply_nonlinear(self, inputs, outputs, resids):

        V_ac = inputs['Vr_ac'] + inputs['Vi_ac']*1j
        I_ac = outputs['Ir_ac'] + outputs['Ii_ac']*1j
        S_ac = V_ac * I_ac.conjugate()
        P_dc = inputs['V_dc'] * outputs['I_dc']

        resids['P_dc'] = P_dc - outputs['P_dc']
        resids['P_ac'] = S_ac.real - outputs['P_ac']
        resids['Q_ac'] = S_ac.imag - outputs['Q_ac']

        resids['I_dc'] = abs(V_ac) - inputs['Ksc'] * inputs['M'] * inputs['V_dc']
        # print(self.pathname, resids['I_dc'], abs(V_ac) - inputs['Ksc'] * inputs['M'] * inputs['V_dc'])

        if self.options['mode'] == 'Lead':
            theta = np.arccos(inputs['PF'])
        else:
            theta = -np.arccos(inputs['PF'])
        # print(self.pathname, theta, np.arctan2(S_ac.imag,S_ac.real))
        resids['Ii_ac'] = theta - np.arctan2(S_ac.imag,S_ac.real)

        if abs(S_ac.real) > abs(P_dc): # power from from AC to DC
            resids['Ir_ac'] = S_ac.real * inputs['eff'] + P_dc 
            # print(self.pathname, 'AC to DC', S_ac.real, inputs['eff'], P_dc[0], resids['Ir_ac'][0])

        else: # power from from DC to AC
            resids['Ir_ac'] = S_ac.real + P_dc * inputs['eff']
            # print('DC to AC', S_ac.real, P_dc[0])

    def solve_nonlinear(self, inputs, outputs):
        V_ac = inputs['Vr_ac'] + inputs['Vi_ac']*1j
        I_ac = outputs['Ir_ac'] + outputs['Ii_ac']*1j
        S_ac = V_ac * I_ac.conjugate()
        P_dc = inputs['V_dc'] * outputs['I_dc']

        outputs['P_dc'] = P_dc
        outputs['P_ac'] = S_ac.real
        outputs['Q_ac'] = S_ac.imag

    def guess_nonlinear(self, inputs, outputs, resids):

        S_guess = inputs['P_ac_guess'] + inputs['P_ac_guess']*(1.0/inputs['PF']**2-1)**0.5*1j
        V_ac = inputs['Vr_ac'] + inputs['Vi_ac']*1j
        I_ac = (S_guess/V_ac).conjugate()

        outputs['Ir_ac'] = I_ac.real
        outputs['Ii_ac'] = I_ac.imag
        outputs['I_dc'] = inputs['P_dc_guess']/inputs['V_dc']

    def linearize(self, inputs, outputs, J):

        V_ac = inputs['Vr_ac'] + inputs['Vi_ac']*1j
        I_ac = outputs['Ir_ac'] + outputs['Ii_ac']*1j
        S_ac = V_ac * I_ac.conjugate()
        Sm_ac = abs(S_ac)
        P_dc = inputs['V_dc'] * outputs['I_dc']

        J['P_dc', 'V_dc'] = outputs['I_dc']
        J['P_dc', 'I_dc'] = inputs['V_dc']
        # J['P_dc', 'P_dc'] = -1.0

        J['P_ac', 'Vr_ac'] = (I_ac.conjugate()).real
        J['P_ac', 'Vi_ac'] = (1j*I_ac.conjugate()).real
        J['P_ac', 'Ir_ac'] = V_ac.real
        J['P_ac', 'Ii_ac'] = (-1j*V_ac).real
        # J['P_ac', 'P_ac'] = -1.0

        J['Q_ac', 'Vr_ac'] = (I_ac.conjugate()).imag
        J['Q_ac', 'Vi_ac'] = (1j*I_ac.conjugate()).imag
        J['Q_ac', 'Ir_ac'] = V_ac.imag
        J['Q_ac', 'Ii_ac'] = (-1j*V_ac).imag
        # J['Q_ac', 'Q_ac'] = -1.0

        J['I_dc', 'Vr_ac'] = inputs['Vr_ac'] / abs(V_ac)
        J['I_dc', 'Vi_ac'] = inputs['Vi_ac'] / abs(V_ac)
        J['I_dc', 'Ksc'] = -inputs['M'] * inputs['V_dc']
        J['I_dc', 'M'] = -inputs['Ksc'] * inputs['V_dc']
        J['I_dc', 'V_dc'] = -inputs['Ksc'] * inputs['M']

        # Partials change basd on which way the power is flowing
        if abs(S_ac.real) > abs(P_dc): # power from from AC to DC
            J['Ir_ac', 'Vr_ac'] = (I_ac.conjugate()).real * inputs['eff']
            J['Ir_ac', 'Vi_ac'] = (1j*I_ac.conjugate()).real * inputs['eff']
            J['Ir_ac', 'Ir_ac'] = V_ac.real * inputs['eff']
            J['Ir_ac', 'Ii_ac'] = (-1j*V_ac).real * inputs['eff']
            J['Ir_ac', 'V_dc'] = outputs['I_dc']
            J['Ir_ac', 'I_dc'] = inputs['V_dc']
            J['Ir_ac', 'eff'] = S_ac.real
        else: # resids['Ir_ac'] = S_ac.real + inputs['V_dc'] * outputs['I_dc'] * inputs['eff'] 
            J['Ir_ac', 'Vr_ac'] = (I_ac.conjugate()).real
            J['Ir_ac', 'Vi_ac'] = (1j*I_ac.conjugate()).real
            J['Ir_ac', 'Ir_ac'] = V_ac.real
            J['Ir_ac', 'Ii_ac'] = (-1j*V_ac).real 
            J['Ir_ac', 'V_dc'] = outputs['I_dc'] * inputs['eff'] 
            J['Ir_ac', 'I_dc'] = inputs['V_dc'] * inputs['eff']
            J['Ir_ac', 'eff'] = inputs['V_dc'] * outputs['I_dc']

        # J['Ii_ac', 'Vr_ac'] = outputs['Ir_ac'] - inputs['PF'] * 0.5 / Sm_ac * (2 * inputs['Vr_ac'] * (outputs['Ir_ac']**2 + outputs['Ii_ac']**2))
        # J['Ii_ac', 'Vi_ac'] = outputs['Ii_ac'] - inputs['PF'] * 0.5 / Sm_ac * (2 * inputs['Vi_ac'] * (outputs['Ir_ac']**2 + outputs['Ii_ac']**2))
        # J['Ii_ac', 'Ir_ac'] = inputs['Vr_ac'] - inputs['PF'] * 0.5 / Sm_ac * (2 * outputs['Ir_ac'] * (inputs['Vr_ac']**2 + inputs['Vi_ac']**2))
        # J['Ii_ac', 'Ii_ac'] = inputs['Vi_ac'] - inputs['PF'] * 0.5 / Sm_ac * (2 * outputs['Ii_ac'] * (inputs['Vr_ac']**2 + inputs['Vi_ac']**2))
        # J['Ii_ac', 'PF'] = -Sm_ac



        J['Ii_ac', 'Vr_ac'] = -(S_ac.real * -outputs['Ii_ac'] - S_ac.imag * outputs['Ir_ac']) / Sm_ac**2
        J['Ii_ac', 'Vi_ac'] = -(S_ac.real * outputs['Ir_ac'] - S_ac.imag * outputs['Ii_ac']) / Sm_ac**2
        J['Ii_ac', 'Ir_ac'] = -(S_ac.real * inputs['Vi_ac'] - S_ac.imag * inputs['Vr_ac']) / Sm_ac**2
        J['Ii_ac', 'Ii_ac'] = -(S_ac.real * -inputs['Vr_ac'] - S_ac.imag * inputs['Vi_ac']) / Sm_ac**2
        if self.options['mode'] == 'Lead':
            J['Ii_ac', 'PF'] = -1.0 / (1.0 - inputs['PF']**2)**0.5
        else:
            J['Ii_ac', 'PF'] = 1.0 / (1.0 - inputs['PF']**2)**0.5


if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    # des_vars.add_output('Vr_ac', 1.05, units='V')
    des_vars.add_output('Vr_ac', 0.990032064216588, units='V')
    des_vars.add_output('Vi_ac', 0.0624540777134769, units='V')
    des_vars.add_output('V_dc', 1.0020202020202, units='V')
    des_vars.add_output('M', 0.99, units=None)
    des_vars.add_output('Ksc', 1.0, units=None)
    des_vars.add_output('eff', 0.98, units=None)
    des_vars.add_output('PF', 0.95, units=None)

    p.model.add_subsystem('con', Converter(num_nodes=1, mode='Lead'), promotes=['*'])

    p.setup(check=False)

    p.check_partials(compact_print=False)
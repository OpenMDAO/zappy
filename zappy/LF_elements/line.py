import math, cmath
import numpy as np

from openmdao.api import ExplicitComponent

class ACline(ExplicitComponent):
    """
    Calculates the current and power in a line.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('R', val=np.ones(nn), units='ohm', desc='Resistance of the line')
        self.add_input('X', val=np.ones(nn), units='ohm', desc='Reactance of the line')
        self.add_input('Vr_in', val=np.ones(nn), units='V', desc='Voltage (real) entering the line')
        self.add_input('Vi_in', val=np.ones(nn), units='V', desc='Voltage (imaginary) entering the line')
        self.add_input('Vr_out', val=np.ones(nn), units='V', desc='Voltage (real) exiting the line')
        self.add_input('Vi_out', val=np.ones(nn), units='V', desc='Voltage (imaginary) exiting the line')

        self.add_output('Ir_in', val=np.ones(nn), units='A', desc='Current (real) entering the line')
        self.add_output('Ii_in', val=np.ones(nn), units='A', desc='Current (imaginary) entering the line')
        self.add_output('Ir_out', val=np.ones(nn), units='A', desc='Current (real) exiting the line')
        self.add_output('Ii_out', val=np.ones(nn), units='A', desc='Current (imaginary) exiting the line')
        self.add_output('P_in', val=np.zeros(nn), units='W', desc='Real (active) power entering the line')
        self.add_output('P_out', val=np.zeros(nn), units='W', desc='Real (active) power exiting the line')
        self.add_output('P_loss', val=np.zeros(nn), units='W', desc='Real (active) power lost in the line')
        self.add_output('Q_in', val=np.zeros(nn), units='V*A', desc='Reactive power entering the line')
        self.add_output('Q_out', val=np.zeros(nn), units='V*A', desc='Reactive power exiting the line')
        self.add_output('Q_loss', val=np.zeros(nn), units='V*A', desc='Reactive power lost in the line')

        ar = np.arange(nn)

        # self.declare_partials('*','*')
        self.declare_partials('Ir_in','R', rows=ar, cols=ar)
        self.declare_partials('Ir_in','X', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vi_in', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vr_out', rows=ar, cols=ar)
        self.declare_partials('Ir_in','Vi_out', rows=ar, cols=ar)

        self.declare_partials('Ii_in','R', rows=ar, cols=ar)
        self.declare_partials('Ii_in','X', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vi_in', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vr_out', rows=ar, cols=ar)
        self.declare_partials('Ii_in','Vi_out', rows=ar, cols=ar)

        self.declare_partials('Ir_out','R', rows=ar, cols=ar)
        self.declare_partials('Ir_out','X', rows=ar, cols=ar)
        self.declare_partials('Ir_out','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ir_out','Vi_in', rows=ar, cols=ar)
        self.declare_partials('Ir_out','Vr_out', rows=ar, cols=ar)
        self.declare_partials('Ir_out','Vi_out', rows=ar, cols=ar)

        self.declare_partials('Ii_out','R', rows=ar, cols=ar)
        self.declare_partials('Ii_out','X', rows=ar, cols=ar)
        self.declare_partials('Ii_out','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Ii_out','Vi_in', rows=ar, cols=ar)
        self.declare_partials('Ii_out','Vr_out', rows=ar, cols=ar)
        self.declare_partials('Ii_out','Vi_out', rows=ar, cols=ar)

        self.declare_partials('P_in', 'R', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'R', rows=ar, cols=ar)

        self.declare_partials('P_in', 'X', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'X', rows=ar, cols=ar)

        self.declare_partials('P_in', 'Vr_out', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'Vr_out', rows=ar, cols=ar)

        self.declare_partials('P_in', 'Vi_out', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'Vi_out', rows=ar, cols=ar)

        self.declare_partials('P_in', 'Vr_in', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'Vr_in', rows=ar, cols=ar)

        self.declare_partials('P_in', 'Vi_in', rows=ar, cols=ar)
        self.declare_partials('Q_in', 'Vi_in', rows=ar, cols=ar)

        self.declare_partials('P_out', 'R', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'R', rows=ar, cols=ar)

        self.declare_partials('P_out', 'X', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'X', rows=ar, cols=ar)

        self.declare_partials('P_out', 'Vr_in', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Vr_in', rows=ar, cols=ar)

        self.declare_partials('P_out', 'Vi_in', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Vi_in', rows=ar, cols=ar)

        self.declare_partials('P_out', 'Vr_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Vr_out', rows=ar, cols=ar)

        self.declare_partials('P_out', 'Vi_out', rows=ar, cols=ar)
        self.declare_partials('Q_out', 'Vi_out', rows=ar, cols=ar)

        self.declare_partials('P_loss','R', rows=ar, cols=ar)
        self.declare_partials('Q_loss','R', rows=ar, cols=ar)

        self.declare_partials('P_loss','X', rows=ar, cols=ar)
        self.declare_partials('Q_loss','X', rows=ar, cols=ar)

        self.declare_partials('P_loss','Vr_in', rows=ar, cols=ar)
        self.declare_partials('Q_loss','Vr_in', rows=ar, cols=ar)

        self.declare_partials('P_loss','Vi_in', rows=ar, cols=ar)
        self.declare_partials('Q_loss','Vi_in', rows=ar, cols=ar)

        self.declare_partials('P_loss','Vr_out', rows=ar, cols=ar)
        self.declare_partials('Q_loss','Vr_out', rows=ar, cols=ar)

        self.declare_partials('P_loss','Vi_out', rows=ar, cols=ar)
        self.declare_partials('Q_loss','Vi_out', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        V_in = inputs['Vr_in'] + inputs['Vi_in']*1j
        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j
        Y = 1.0/(inputs['R'] + inputs['X']*1j)

        # Compute complex values for currents and powers
        I_in = Y*(V_in-V_out)
        I_out = Y*(V_out-V_in)
        S_in = V_in*I_in.conjugate()
        S_out = V_out*I_out.conjugate()
        S_loss = S_in+S_out

        # Convert computed complex values to required outputs
        outputs['Ir_in'] = I_in.real
        outputs['Ii_in'] = I_in.imag
        outputs['Ir_out'] = I_out.real
        outputs['Ii_out'] = I_out.imag

        outputs['P_in'] = S_in.real
        outputs['Q_in'] = S_in.imag
        outputs['P_out'] = S_out.real
        outputs['Q_out'] = S_out.imag
        outputs['P_loss'] = S_loss.real
        outputs['Q_loss'] = S_loss.imag

    def compute_partials(self, inputs, J):

        # Create complex values based on inputs
        V_in = inputs['Vr_in'] + inputs['Vi_in']*1j
        V_out = inputs['Vr_out'] + inputs['Vi_out']*1j
        Y = 1.0/(inputs['R'] + inputs['X']*1j)
        Yconj = Y.conjugate()
        I_in = Y*(V_in-V_out)
        I_out = Y*(V_out-V_in)

        dI_in_dR = -(V_in-V_out)*Y**2
        dI_in_dX = -1j*(V_in-V_out)*Y**2

        # Compute partial derivatives 
        J['Ir_in','R'] = dI_in_dR.real
        J['Ir_in','X'] = dI_in_dX.real
        J['Ir_in','Vr_in'] = Y.real
        J['Ir_in','Vi_in'] = (Y*1j).real
        J['Ir_in','Vr_out'] = -Y.real
        J['Ir_in','Vi_out'] = -(Y*1j).real

        J['Ii_in','R'] = dI_in_dR.imag
        J['Ii_in','X'] = dI_in_dX.imag
        J['Ii_in','Vr_in'] = Y.imag
        J['Ii_in','Vi_in'] = (Y*1j).imag
        J['Ii_in','Vr_out'] = -Y.imag
        J['Ii_in','Vi_out'] = -(Y*1j).imag

        J['Ir_out','R'] = -J['Ir_in','R']
        J['Ir_out','X'] = -J['Ir_in','X']
        J['Ir_out','Vr_in'] = -J['Ir_in','Vr_in']
        J['Ir_out','Vi_in'] = -J['Ir_in','Vi_in']
        J['Ir_out','Vr_out'] = -J['Ir_in','Vr_out']
        J['Ir_out','Vi_out'] = -J['Ir_in','Vi_out']

        J['Ii_out','R'] = -J['Ii_in','R']
        J['Ii_out','X'] = -J['Ii_in','X']
        J['Ii_out','Vr_in'] = -J['Ii_in','Vr_in']
        J['Ii_out','Vi_in'] = -J['Ii_in','Vi_in']
        J['Ii_out','Vr_out'] = -J['Ii_in','Vr_out']
        J['Ii_out','Vi_out'] = -J['Ii_in','Vi_out']

        J['P_in', 'R'] = (-V_in*(V_in-V_out).conjugate()*Yconj**2).real
        J['Q_in', 'R'] = (-V_in*(V_in-V_out).conjugate()*Yconj**2).imag

        J['P_in', 'X'] = (1j*V_in*(V_in-V_out).conjugate()*Yconj**2).real
        J['Q_in', 'X'] = (1j*V_in*(V_in-V_out).conjugate()*Yconj**2).imag

        J['P_in', 'Vr_out'] = (-V_in*Yconj).real
        J['Q_in', 'Vr_out'] = (-V_in*Yconj).imag

        J['P_in', 'Vi_out'] = (V_in*Yconj*1j).real
        J['Q_in', 'Vi_out'] = (V_in*Yconj*1j).imag

        J['P_in', 'Vr_in'] = (Yconj*(2*inputs['Vr_in']-V_out.conjugate())).real
        J['Q_in', 'Vr_in'] = (Yconj*(2*inputs['Vr_in']-V_out.conjugate())).imag

        J['P_in', 'Vi_in'] = (Yconj*(2*inputs['Vi_in']-1j*V_out.conjugate())).real
        J['Q_in', 'Vi_in'] = (Yconj*(2*inputs['Vi_in']-1j*V_out.conjugate())).imag

        J['P_out', 'R'] = (-V_out*(V_out-V_in).conjugate()*Yconj**2).real
        J['Q_out', 'R'] = (-V_out*(V_out-V_in).conjugate()*Yconj**2).imag

        J['P_out', 'X'] = (1j*V_out*(V_out-V_in).conjugate()*Yconj**2).real
        J['Q_out', 'X'] = (1j*V_out*(V_out-V_in).conjugate()*Yconj**2).imag

        J['P_out', 'Vr_in'] = (-V_out*Yconj).real
        J['Q_out', 'Vr_in'] = (-V_out*Yconj).imag

        J['P_out', 'Vi_in'] = (V_out*Yconj*1j).real
        J['Q_out', 'Vi_in'] = (V_out*Yconj*1j).imag

        J['P_out', 'Vr_out'] = (Yconj*(2*inputs['Vr_out']-V_in.conjugate())).real
        J['Q_out', 'Vr_out'] = (Yconj*(2*inputs['Vr_out']-V_in.conjugate())).imag

        J['P_out', 'Vi_out'] = (Yconj*(2*inputs['Vi_out']-1j*V_in.conjugate())).real
        J['Q_out', 'Vi_out'] = (Yconj*(2*inputs['Vi_out']-1j*V_in.conjugate())).imag

        J['P_loss','R'] = J['P_in','R']+J['P_out','R']
        J['Q_loss','R'] = J['Q_in','R']+J['Q_out','R']

        J['P_loss','X'] = J['P_in','X']+J['P_out','X']
        J['Q_loss','X'] = J['Q_in','X']+J['Q_out','X']

        J['P_loss','Vr_in'] = J['P_in','Vr_in']+J['P_out','Vr_in']
        J['Q_loss','Vr_in'] = J['Q_in','Vr_in']+J['Q_out','Vr_in']

        J['P_loss','Vi_in'] = J['P_in','Vi_in']+J['P_out','Vi_in']
        J['Q_loss','Vi_in'] = J['Q_in','Vi_in']+J['Q_out','Vi_in']

        J['P_loss','Vr_out'] = J['P_in','Vr_out']+J['P_out','Vr_out']
        J['Q_loss','Vr_out'] = J['Q_in','Vr_out']+J['Q_out','Vr_out']

        J['P_loss','Vi_out'] = J['P_in','Vi_out']+J['P_out','Vi_out']
        J['Q_loss','Vi_out'] = J['Q_in','Vi_out']+J['Q_out','Vi_out']

class DCline(ExplicitComponent):
    """
    Calculates the current and power in a line.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('R', val=np.ones(nn), units='ohm', desc='Resistance of the line')
        self.add_input('V_in', val=np.ones(nn), units='V', desc='Voltage entering the line')
        self.add_input('V_out', val=np.ones(nn), units='V', desc='Voltage  exiting the line')

        self.add_output('I_in', val=np.ones(nn), units='A', desc='Current entering the line')
        self.add_output('I_out', val=np.ones(nn), units='A', desc='Current exiting the line')
        self.add_output('P_in', val=np.zeros(nn), units='W', desc='Power entering the line')
        self.add_output('P_out', val=np.zeros(nn), units='W', desc='Power exiting the line')
        self.add_output('P_loss', val=np.zeros(nn), units='W', desc='Power lost in the line')

        ar = np.arange(nn)

        self.declare_partials('I_in','R', rows=ar, cols=ar)
        self.declare_partials('I_in','V_in', rows=ar, cols=ar)
        self.declare_partials('I_in','V_out', rows=ar, cols=ar)

        self.declare_partials('I_out','R', rows=ar, cols=ar)
        self.declare_partials('I_out','V_in', rows=ar, cols=ar)
        self.declare_partials('I_out','V_out', rows=ar, cols=ar)

        self.declare_partials('P_in', 'R', rows=ar, cols=ar)
        self.declare_partials('P_in', 'V_out', rows=ar, cols=ar)
        self.declare_partials('P_in', 'V_in', rows=ar, cols=ar)

        self.declare_partials('P_out', 'R', rows=ar, cols=ar)
        self.declare_partials('P_out', 'V_in', rows=ar, cols=ar)
        self.declare_partials('P_out', 'V_out', rows=ar, cols=ar)

        self.declare_partials('P_loss','R', rows=ar, cols=ar)
        self.declare_partials('P_loss','V_in', rows=ar, cols=ar)
        self.declare_partials('P_loss','V_out', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        Y = 1.0/inputs['R']

        outputs['I_in'] =  Y*(inputs['V_in']-inputs['V_out'])
        outputs['I_out'] = Y*(inputs['V_out']-inputs['V_in'])

        outputs['P_in'] = inputs['V_in']*outputs['I_in']
        outputs['P_out'] = inputs['V_out']*outputs['I_out']
        outputs['P_loss'] = outputs['P_in']+outputs['P_out']

    def compute_partials(self, inputs, J):

        Y = 1.0/inputs['R']

        # Compute partial derivatives 
        J['I_in','R'] = -(inputs['V_in']-inputs['V_out'])*Y**2
        J['I_in','V_in'] = Y
        J['I_in','V_out'] = -Y

        J['I_out','R'] = -J['I_in','R']
        J['I_out','V_in'] = -J['I_in','V_in']
        J['I_out','V_out'] = -J['I_in','V_out']

        J['P_in', 'R'] = -inputs['V_in']*(inputs['V_in']-inputs['V_out'])*Y**2
        J['P_in', 'V_out'] = -inputs['V_in']*Y
        J['P_in', 'V_in'] = Y*(2*inputs['V_in']-inputs['V_out'])

        J['P_out', 'R'] = -inputs['V_out']*(inputs['V_out']-inputs['V_in'])*Y**2
        J['P_out', 'V_in'] = -inputs['V_out']*Y
        J['P_out', 'V_out'] = Y*(2*inputs['V_out']-inputs['V_in'])

        J['P_loss','R'] = J['P_in','R']+J['P_out','R']
        J['P_loss','V_in'] = J['P_in','V_in']+J['P_out','V_in']
        J['P_loss','V_out'] = J['P_in','V_out']+J['P_out','V_out']

if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('R', 0.02*np.ones(3), units='ohm') #0.02
    des_vars.add_output('X', 0.04*np.ones(3), units='ohm') #0.04
    des_vars.add_output('Vr_in', 1.05*np.ones(3), units='V')  #1.05
    des_vars.add_output('Vi_in', 0.0*np.ones(3), units='V')
    des_vars.add_output('Vr_out', 0.979995025823362*np.ones(3), units='V') #0.98183
    des_vars.add_output('Vi_out', -0.0599991521729049*np.ones(3), units='V') #0.98183

    p.model.add_subsystem('acline', ACline(num_nodes=3), promotes_inputs=['*'])

    des_vars.add_output('V_in', 1.05*np.ones(3), units='V')  #1.05
    des_vars.add_output('V_out', 0.979995025823362*np.ones(3), units='V') #0.98183

    p.model.add_subsystem('dcline', DCline(num_nodes=3), promotes_inputs=['*'])


    p.setup(check=False)
    p.run_model()

    # print('Ir_in', p['Ir_in'])
    # print('Ii_in', p['Ii_in'])
    # print('Ir_out', p['Ir_out'])
    # print('Ii_out', p['Ii_out'])
    # print('P_in', p['P_in'])
    # print('Q_in', p['Q_in'])
    # print('P_out', p['P_out'])
    # print('Q_out', p['Q_out'])
    # print('P_loss', p['P_loss'])
    # print('Q_loss', p['Q_loss'])

    # p.check_partials()
    p.check_partials(compact_print=False)
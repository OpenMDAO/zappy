import math, cmath
import numpy as np

from openmdao.api import ImplicitComponent, Group
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from pyLF.LF_elements.generator import ACgenerator, DCgenerator
from pyLF.LF_elements.load import ACload, DCload


class InverterCalcs(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Phase', desc='Control Mode: Phase or PF')

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        mode = self.options['mode']
        if not (mode=="Phase" or mode=="PF"):
            raise ValueError("mode must be 'Phase' or 'PF', but '{}' was given.".format(mode))

        self.add_input('M', val=np.ones(nn), units=None, desc='Inverter modulation index (V_ac/V_dc)')
        self.add_input('eff', val=np.ones(nn), units=None, desc='Inverter efficiency (P_ac/P_dc')

        self.add_input('P_ac', val=np.ones(nn), units='W', desc='Real power leaving inverter')
        self.add_input('V_dc', val=np.ones(nn), units='V', desc='Voltage entering inverter')

        self.add_output('P_dc', val=np.ones(nn), units='W', desc='Power entering inverter')
        self.add_output('Vm_ac', val=np.ones(nn), units='V', desc='Voltage magnitude leaving inverter')
        self.add_output('thetaV_bus', val=np.zeros(nn), units='deg', desc='Voltage phase angle')
        self.add_output('P_loss', val=np.ones(nn), units='W', desc='Power lost through inverter')

        self.declare_partials('P_dc', 'eff', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'P_ac', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'P_dc', rows=ar, cols=ar,val=1.0)
        self.declare_partials('Vm_ac', 'V_dc', rows=ar, cols=ar)
        self.declare_partials('Vm_ac', 'M', rows=ar, cols=ar)
        self.declare_partials('Vm_ac', 'Vm_ac', rows=ar, cols=ar,val=-1.0)
        self.declare_partials('P_loss','P_ac', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_dc', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_loss', rows=ar, cols=ar, val=-1.0)

        if mode == 'Phase':
            self.add_input('thetaV_target', val=np.zeros(nn), units='deg', desc='Target voltage phase angle output')

            self.declare_partials('thetaV_bus', 'thetaV_target', rows=ar, cols=ar, val=1.0)
            self.declare_partials('thetaV_bus', 'thetaV_bus', rows=ar, cols=ar, val=-1.0)

        else:
            self.add_input('PF', val=np.ones(nn), units=None, desc='Inverter power factor')
            self.add_input('Q_ac', val=np.ones(nn), units='V*A', desc='Reactive power leaving inverter')

            self.declare_partials('thetaV_bus', 'P_ac', rows=ar, cols=ar)
            self.declare_partials('thetaV_bus', 'Q_ac', rows=ar, cols=ar)
            self.declare_partials('thetaV_bus', 'PF', rows=ar, cols=ar)
            self.declare_partials('thetaV_bus', 'PF', rows=ar, cols=ar, val=-1.0)

    def apply_nonlinear(self, inputs, outputs, resids):

        mode = self.options['mode']

        resids['P_dc'] = inputs['P_ac'] / inputs['eff'] + outputs['P_dc']
        resids['Vm_ac'] = inputs['V_dc'] * inputs['M'] - outputs['Vm_ac']
        resids['P_loss'] = inputs['P_ac'] + outputs['P_dc'] - outputs['P_loss']

        if mode == 'Phase':
            resids['thetaV_bus'] = inputs['thetaV_target'] - outputs['thetaV_bus']
        else:
            resids['thetaV_bus'] = inputs['P_ac'] / (inputs['P_ac']**2 + inputs['Q_ac']**2)**0.5 - inputs['PF']


    def solve_nonlinear(self, inputs, outputs):

        mode = self.options['mode']

        outputs['P_dc'] = -inputs['P_ac'] / inputs['eff']
        outputs['Vm_ac'] = inputs['V_dc'] * inputs['M']
        outputs['P_loss'] = inputs['P_ac'] + outputs['P_dc']

        if mode == 'Phase':
            outputs['thetaV_bus'] = inputs['thetaV_target']


    def linearize(self, inputs, outputs, J):

        mode = self.options['mode']

        J['P_dc', 'eff'] = -inputs['P_ac'] / inputs['eff']**2
        J['P_dc', 'P_ac'] = 1 / inputs['eff']

        J['Vm_ac', 'V_dc'] = inputs['M']
        J['Vm_ac', 'M'] = inputs['V_dc']

        if mode == 'PF':
            J['thetaV_bus', 'P_ac'] = inputs['Q_ac']**2 / (inputs['P_ac']**2 + inputs['Q_ac']**2)**1.5
            J['thetaV_bus', 'Q_ac'] = -inputs['P_ac'] * inputs['Q_ac'] / (inputs['P_ac']**2 + inputs['Q_ac']**2)**1.5


class Inverter(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Phase', desc='Control Mode: Phase or PF')
        self.options.declare('Q_min', allow_none=True, default=None, desc='Lower bound for reactive power (Q)')
        self.options.declare('Q_max', allow_none=True, default=None, desc='Upper bound for reactive power (Q)')
        self.options.declare('Vbase', default=5000.0, desc='Base voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        mode = self.options['mode']
        Q_min = self.options['Q_min']
        Q_max = self.options['Q_max']
        Vbase = self.options['Vbase']
        Sbase = self.options['Sbase']

        self.add_subsystem('load', DCload(num_nodes=nn), promotes=[('P','P_dc'),('V_in','V_dc'),('I_in','I_dc')])

        self.add_subsystem('gen', ACgenerator(num_nodes=nn, mode='Slack', Q_min=Q_min, Q_max=Q_max, Vbase=Vbase, Sbase=Sbase),
                                             promotes=[('Vm_bus','Vm_ac'),('Vr_out','Vr_ac'),('Vi_out','Vi_ac'),
                                            ('Ir_out','Ir_ac'),('Ii_out','Ii_ac'),('P_out','P_ac'),('Q_out','Q_ac'),
                                            'thetaV_bus','P_guess'])

        if mode == 'Phase':
            self.add_subsystem('calcs', InverterCalcs(num_nodes=nn, mode=mode), promotes=['eff','M','P_ac','V_dc','P_dc',
                                            'Vm_ac','thetaV_bus','thetaV_target'])
        else:
            self.add_subsystem('calcs', InverterCalcs(num_nodes=nn, mode=mode), promotes=['eff','M','P_ac','V_dc','P_dc',
                                            'Vm_ac','thetaV_bus','Q_ac','PF'])


        # newton = self.nonlinear_solver = NewtonSolver()
        # newton.options['atol'] = 1e-4
        # newton.options['rtol'] = 1e-4
        # newton.options['iprint'] = 2
        # newton.options['maxiter'] = 10
        # newton.options['solve_subsystems'] = True
        # newton.options['max_sub_solves'] = 3
        
        # newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1

        # self.linear_solver = DirectSolver(assemble_jac=True)





class InverterCalcs2(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Lead', values=['Lead', 'Lag'], desc='Specifies weather AC currentl leads or lags the voltage')

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)
        mode = self.options['mode']

        self.add_input('M', val=np.ones(nn), units=None, desc='Inverter modulation index (V_ac/V_dc)')
        self.add_input('eff', val=np.ones(nn), units=None, desc='Inverter efficiency (P_ac/P_dc')
        self.add_input('PF', val=np.ones(nn), units=None, desc='Inverter power factor')

        self.add_input('V_dc', val=np.ones(nn), units='V', desc='Voltage entering inverter')
        self.add_input('Vr_ac', val=np.ones(nn), units='V', desc='Real voltage leaving inverter')
        self.add_input('Vi_ac', val=np.ones(nn), units='V', desc='Imaginary voltage leaving inverter')
        self.add_input('P_dc_guess', val=np.zeros(nn), units='W', desc='Guess for power entering inverter')

        self.add_output('P_dc', val=np.ones(nn), units='W', desc='Power entering inverter')
        self.add_output('P_ac', val=np.ones(nn), units='W', desc='Real power leaving inverter')
        self.add_output('Q_ac', val=np.ones(nn), units='V*A', desc='Reactive power leaving inverter')
        self.add_output('P_loss', val=np.ones(nn), units='W', desc='Power lost through inverter')

        self.declare_partials('P_dc', 'V_dc', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'M', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('P_dc', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'P_ac', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_ac', 'P_dc', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'eff', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'P_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'PF', rows=ar, cols=ar)
        self.declare_partials('P_loss','P_ac', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_dc', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_loss', rows=ar, cols=ar, val=-1.0)

        if mode == 'Lead':
            self.declare_partials('Q_ac', 'Q_ac', rows=ar, cols=ar, val=1.0)
        else:
            self.declare_partials('Q_ac', 'Q_ac', rows=ar, cols=ar, val=-1.0)

    def apply_nonlinear(self, inputs, outputs, resids):

        mode = self.options['mode']

        resids['P_dc'] = inputs['V_dc'] * inputs['M'] - (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5
        resids['P_ac'] = outputs['P_ac'] + outputs['P_dc'] * inputs['eff']
        resids['P_loss'] = outputs['P_ac'] + outputs['P_dc'] - outputs['P_loss']

        if mode=='Lead':
            resids['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5 + outputs['Q_ac']
        else:
            resids['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5 - outputs['Q_ac']

    def guess_nonlinear(self, inputs, outputs, resids):

        mode = self.options['mode']

        outputs['P_dc'] = inputs['P_dc_guess']
        outputs['P_ac'] = -outputs['P_dc'] * inputs['eff']
        if mode == 'Lead':
            outputs['Q_ac'] = -outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5
        else:
            outputs['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5

    def solve_nonlinear(self, inputs, outputs):

        mode = self.options['mode']

        outputs['P_ac'] = -outputs['P_dc'] * inputs['eff']
        outputs['P_loss'] = outputs['P_ac'] + outputs['P_dc']

        if mode == 'Lead':
            outputs['Q_ac'] = -outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5
        else:
            outputs['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5

    def linearize(self, inputs, outputs, J):

        mode = self.options['mode']

        J['P_dc', 'V_dc'] = inputs['M']
        J['P_dc', 'M'] = inputs['V_dc']
        J['P_dc', 'Vr_ac'] = -inputs['Vr_ac'] / (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5
        J['P_dc', 'Vi_ac'] = -inputs['Vi_ac'] / (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5

        J['P_ac', 'P_dc'] = inputs['eff']
        J['P_ac', 'eff'] = outputs['P_dc']

        J['Q_ac', 'P_ac'] = ((1.0 / inputs['PF'])**2 - 1.0)**0.5 
        J['Q_ac', 'PF'] = -outputs['P_ac'] / ((1.0 / inputs['PF'])**2 - 1.0)**0.5 / inputs['PF']**3


class Inverter2(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mode', default='Lead', values=['Lead', 'Lag'], desc='Specifies weather AC currentl leads or lags the voltage')

    def setup(self):

        nn = self.options['num_nodes']
        mode = self.options['mode']

        self.add_subsystem('dc_load', DCload(num_nodes=nn), promotes=[('P','P_dc'),('V_in','V_dc'),('I_in','I_dc')])

        self.add_subsystem('ac_load', ACload(num_nodes=nn), promotes=[('P','P_ac'),('Q','Q_ac'),('Vr_in','Vr_ac'),('Vi_in','Vi_ac'),
                                            ('Ir_in','Ir_ac'),('Ii_in','Ii_ac')])

        self.add_subsystem('calcs', InverterCalcs2(num_nodes=nn, mode=mode), promotes=['eff','M','PF','V_dc','Vr_ac','Vi_ac',
                                            'P_dc','P_ac','Q_ac','P_dc_guess'])

        # newton = self.nonlinear_solver = NewtonSolver()
        # newton.options['atol'] = 1e-4
        # newton.options['rtol'] = 1e-4
        # newton.options['iprint'] = 2
        # newton.options['maxiter'] = 10
        # newton.options['solve_subsystems'] = True
        # newton.options['max_sub_solves'] = 3
        
        # newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['bound_enforcement'] = 'scalar'
        # newton.linesearch.options['print_bound_enforce'] = True
        # newton.linesearch.options['iprint'] = -1

        # self.linear_solver = DirectSolver(assemble_jac=True)



if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    # des_vars.add_output('Vr_ac', 1.05, units='V')
    des_vars.add_output('Vr_ac', 0.990032064216588*np.ones(3), units='V')
    des_vars.add_output('Vi_ac', 0.0624540777134769*np.ones(3), units='V')
    des_vars.add_output('V_dc', 1.0020202020202*np.ones(3), units='V')
    des_vars.add_output('M', 0.99*np.ones(3), units=None)
    des_vars.add_output('eff', 0.98*np.ones(3), units=None)
    des_vars.add_output('PF', 0.95*np.ones(3), units=None)
    des_vars.add_output('thetaV_target', 0.0*np.ones(3), units='deg')

    # p.model.add_subsystem('con', Inverter2(num_nodes=3, mode='Lead'), promotes=['*'])
    p.model.add_subsystem('con', Inverter(num_nodes=3, mode='Phase'), promotes=['*'])

    p.setup(check=False)
    p.final_setup()

    # p.view_model()

    p.check_partials(compact_print=False)
import math, cmath
import numpy as np

from openmdao.api import ImplicitComponent, Group
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from pyLF.LF_elements.generator import ACgenerator, DCgenerator
from pyLF.LF_elements.load import ACload, DCload


class RectifierCalcs(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):

        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('M', val=np.ones(nn), units=None, desc='Rectifier modulation index (Vm_dc/Vm_ac)')
        self.add_input('eff', val=np.ones(nn), units=None, desc='Rectifier efficiency (P_out/P_in')
        self.add_input('PF', val=np.ones(nn), units=None, desc='Rectifier power factor')

        self.add_input('P_dc', val=np.ones(nn), units='W', desc='Power leaving rectifier')
        self.add_input('Vr_ac', val=np.ones(nn), units='V', desc='Real component of voltage entering rectifier')
        self.add_input('Vi_ac', val=np.ones(nn), units='V', desc='Imaginary component of voltage entering rectifier')

        self.add_output('P_ac', val=np.ones(nn), units='W', desc='Power entering rectifier')
        self.add_output('Vm_dc', val=np.ones(nn), units='V', desc='Voltage leaving rectifier')
        self.add_output('Q_ac', val=np.ones(nn), units='V*A', desc='Reactive power leaving rectifier')
        self.add_output('P_loss', val=np.ones(nn), units='W', desc='Power lost through rectifier')

        self.declare_partials('P_ac', 'eff', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'P_dc', rows=ar, cols=ar)
        self.declare_partials('P_ac', 'P_ac', rows=ar, cols=ar, val=1.0)
        self.declare_partials('Vm_dc', 'Vr_ac', rows=ar, cols=ar)
        self.declare_partials('Vm_dc', 'Vi_ac', rows=ar, cols=ar)
        self.declare_partials('Vm_dc', 'M', rows=ar, cols=ar)
        self.declare_partials('Vm_dc', 'Vm_dc', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('Q_ac', 'P_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Q_ac', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'PF', rows=ar, cols=ar)
        self.declare_partials('Q_ac', 'Q_ac', rows=ar, cols=ar, val=-1.0)
        self.declare_partials('P_loss','P_ac', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_dc', rows=ar, cols=ar, val=1.0)
        self.declare_partials('P_loss','P_loss', rows=ar, cols=ar, val=-1.0)

    def apply_nonlinear(self, inputs, outputs, resids):

        resids['P_ac'] = inputs['P_dc'] / inputs['eff'] + outputs['P_ac']
        resids['Vm_dc'] = (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5 * inputs['M'] - outputs['Vm_dc']
        resids['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5 - outputs['Q_ac']
        resids['P_loss'] = outputs['P_ac'] + inputs['P_dc'] - outputs['P_loss']

    def solve_nonlinear(self, inputs, outputs):

        outputs['P_ac'] = -inputs['P_dc'] / inputs['eff']
        outputs['Vm_dc'] = (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5 * inputs['M']
        outputs['Q_ac'] = outputs['P_ac'] * ((1.0 / inputs['PF'])**2 - 1.0)**0.5
        outputs['P_loss'] = outputs['P_ac'] + inputs['P_dc']

    def linearize(self, inputs, outputs, J):

        J['P_ac', 'eff'] = -inputs['P_dc'] / inputs['eff']**2
        J['P_ac', 'P_dc'] = 1 / inputs['eff']

        J['Vm_dc', 'Vr_ac'] = inputs['M'] * inputs['Vr_ac'] / (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5
        J['Vm_dc', 'Vi_ac'] = inputs['M'] * inputs['Vi_ac'] / (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5
        J['Vm_dc', 'M'] = (inputs['Vr_ac']**2 + inputs['Vi_ac']**2)**0.5

        J['Q_ac', 'P_ac'] = ((1.0 / inputs['PF'])**2 - 1.0)**0.5
        J['Q_ac', 'PF'] = -outputs['P_ac'] / (inputs['PF']**3 * ((1.0 / inputs['PF'])**2 - 1.0)**0.5)

class Rectifier(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('P_min', allow_none=True, default=None, desc='Lower bound for active power (P)')
        self.options.declare('P_max', allow_none=True, default=None, desc='Upper bound for active power (P)')
        self.options.declare('Vbase', default=5000.0, desc='Base DC voltage in units of volts')
        self.options.declare('Sbase', default=10.0E6, desc='Base DC power in units of watts')

    def setup(self):

        nn = self.options['num_nodes']
        P_min = self.options['P_min']
        P_max = self.options['P_max']
        Vbase = self.options['Vbase']
        Sbase = self.options['Sbase']

        self.add_subsystem('load', ACload(num_nodes=nn), promotes=[('P','P_ac'),('Q','Q_ac'),('Vr_in','Vr_ac'),('Vi_in','Vi_ac'),
                                            ('Ir_in','Ir_ac'),('Ii_in','Ii_ac')])

        self.add_subsystem('gen', DCgenerator(num_nodes=nn, P_min=P_min, P_max=P_max, Vbase=Vbase, Sbase=Sbase), 
                                            promotes=[('V_bus','Vm_dc'),('V_out','V_dc'),
                                            ('I_out','I_dc'),('P_out','P_dc'),'P_guess'])

        self.add_subsystem('calcs', RectifierCalcs(num_nodes=nn), promotes=['eff','M','PF','P_dc','Vr_ac',
                                            'Vi_ac','P_ac','Vm_dc','Q_ac'])

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

    p.model.add_subsystem('con', Rectifier(num_nodes=3), promotes=['*'])

    p.setup(check=False)
    p.final_setup()

    # p.view_model()

    p.check_partials(compact_print=False)
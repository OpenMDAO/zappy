from openmdao.api import ExplicitComponent

class Resistor(ExplicitComponent):

    def setup(self):

        self.add_input('V_in', val=0.0, units='V', desc='voltage on one end of the resistor')
        self.add_input('V_out', val=0.0, units='V', desc='voltage at the other end of the resistor')
        self.add_input('R', val=0.0, units='ohm', desc='resistance')

        self.add_output('I_in', val=0.0, units='A', desc='current entering the resistor')
        self.add_output('I_out', val=0.0, units='A', desc='current leaving the resistor')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        outputs['I_in'] = (inputs['V_out'] - inputs['V_in']) / inputs['R']
        outputs['I_out'] = (inputs['V_in'] - inputs['V_out']) / inputs['R']

    def compute_partials(self, inputs, J):

        J['I_in', 'V_in'] = -1.0/inputs['R']
        J['I_in', 'V_out'] = 1.0/inputs['R']
        J['I_in', 'R'] = -(inputs['V_out'] - inputs['V_in']) / inputs['R']**2

        J['I_out', 'V_in'] = 1.0/inputs['R']
        J['I_out', 'V_out'] = -1.0/inputs['R']
        J['I_out', 'R'] = -(inputs['V_in'] - inputs['V_out']) / inputs['R']**2

if __name__ == '__main__':

    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('res', Resistor(), promotes=['*'])

    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('V_in', 0.0, units='V')
    des_vars.add_output('V_out', 10.0, units='V')
    des_vars.add_output('R', 5.0, units='ohm')

    p.setup(check=False)
    p.run()

    print(p['I_in'], p['I_out'])

    p.check_partials()

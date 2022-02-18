from openmdao.api import ExplicitComponent


class ThermalMass(ExplicitComponent):
    """
    Calculates the the convective heat transfer and temperture rise of an object.
    """

    def setup(self):

        self.add_input('area', val=1.0, units='m**2', desc='Surface area for convective heat transfer')
        self.add_input('mass', val=1.0, units='kg', desc='Mass of the object')
        self.add_input('T_f', val=300.0, units='K', desc='Temperature of fluid used in convective heat transfer')
        self.add_input('T_b', val=300.0, units='K', desc='Temperature of the body')
        self.add_input('h', val=0.0, units='W/(m**2*K)', desc='Heat transfer coefficient')
        self.add_input('c_p', val=1.0, units='J/(K*kg)', desc='Specific heat capacity of the object')
        self.add_input('Qdot_in', val=0.0, units='W', desc='Heat generated by the object')

        self.add_output('Qdot_out', val=0.0, units='W', desc='Heat lost by the object due to convection')
        self.add_output('dTdt', val=0.0, units='K/s', desc='Rate of temperature change of object')

        self.declare_partials('Qdot_out', ['h', 'area', 'T_f', 'T_b'])
        self.declare_partials('dTdt', '*')

    def compute(self, inputs, outputs):
        
        outputs['Qdot_out'] = inputs['h']*inputs['area']*(inputs['T_f']-inputs['T_b'])
        outputs['dTdt'] = (inputs['Qdot_in']-outputs['Qdot_out'])/(inputs['mass']*inputs['c_p'])

    def compute_partials(self, inputs, J):

        J['Qdot_out', 'h'] = inputs['area']*(inputs['T_f']-inputs['T_b'])
        J['Qdot_out', 'area'] = inputs['h']*(inputs['T_f']-inputs['T_b'])
        J['Qdot_out', 'T_f'] = inputs['h']*inputs['area']
        J['Qdot_out', 'T_b'] = -inputs['h']*inputs['area']

        J['dTdt', 'Qdot_in'] = 1.0/(inputs['mass']*inputs['c_p'])
        J['dTdt', 'h'] = -inputs['area']*(inputs['T_f']-inputs['T_b'])/(inputs['mass']*inputs['c_p'])
        J['dTdt', 'area'] = -inputs['h']*(inputs['T_f']-inputs['T_b'])/(inputs['mass']*inputs['c_p'])
        J['dTdt', 'T_f'] = -inputs['h']*inputs['area']/(inputs['mass']*inputs['c_p'])
        J['dTdt', 'T_b'] = inputs['h']*inputs['area']/(inputs['mass']*inputs['c_p'])
        J['dTdt', 'mass'] = -(inputs['Qdot_in']-inputs['h']*inputs['area']*(inputs['T_f']-inputs['T_b']))/(inputs['mass']**2*inputs['c_p'])
        J['dTdt', 'c_p'] = -(inputs['Qdot_in']-inputs['h']*inputs['area']*(inputs['T_f']-inputs['T_b']))/(inputs['mass']*inputs['c_p']**2)

if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('area', 1.0, units='m**2')
    des_vars.add_output('mass', 5.0, units='kg')
    des_vars.add_output('T_f', 273.0, units='K')
    des_vars.add_output('T_b', 500.0, units='K')
    des_vars.add_output('h', 7.0, units='W/(m**2*K)')
    des_vars.add_output('c_p', 48.0, units='J/(K*kg)')
    des_vars.add_output('Qdot_in', 100.0, units='W')

    p.model.add_subsystem('tm', ThermalMass(), promotes=['*'])

    p.setup(check=False)
    p.run_model()

    p.check_partials()
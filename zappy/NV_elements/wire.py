import numpy as np

from openmdao.api import Group, ExplicitComponent

from thermal_mass import ThermalMass

class WireMassVolume(ExplicitComponent):
    """
    Calculates the mass and volume of a wire.
    """

    def setup(self):

        self.add_input('length', val=0.0, units='m', desc='Length of the wire')
        self.add_input('dia', val=0.0, units='m', desc='Diameter of the wire')
        self.add_input('density', val=0.0, units='kg/m**3', desc='Density of wire material')
        
        self.add_output('area', val=0.0, units='m**2', desc='Surface area of the wire')
        self.add_output('volume', val=0.0, units='m**3', desc='Volume of the wire')
        self.add_output('mass', val=0.0, units='kg', desc='Mass of the wire')

        self.declare_partials('area', ['dia', 'length'])
        self.declare_partials('volume', ['dia', 'length'])
        self.declare_partials('mass', '*')

    def compute(self, inputs, outputs):

        outputs['area'] = np.pi * inputs['dia'] * inputs['length']
        outputs['volume'] = np.pi * (inputs['dia']/2.)**2 * inputs['length']
        outputs['mass'] = inputs['density'] * np.pi * (inputs['dia']/2.)**2 * inputs['length']

    def compute_partials(self, inputs, J):

        J['area', 'dia'] = np.pi * inputs['length']
        J['area', 'length'] = np.pi * inputs['dia']

        J['volume', 'dia'] = np.pi / 2. * inputs['dia'] * inputs['length']
        J['volume', 'length'] = np.pi * (inputs['dia']/2.)**2

        J['mass', 'dia'] = inputs['density'] * np.pi / 2. * inputs['dia'] * inputs['length']
        J['mass', 'length'] = inputs['density'] * np.pi * (inputs['dia']/2.)**2
        J['mass', 'density'] = np.pi * (inputs['dia']/2.)**2 * inputs['length']

class WirePerf(ExplicitComponent):
    """
    Calculates the performance of a wire
    """
    
    def setup(self):

        self.add_input('length', val=0.0, units='m', desc='Length of the wire')
        self.add_input('I_in', val=0.0, units='A', desc='Current entering the wire')
        self.add_input('V_in', val=0.0, units='V', desc='Voltage entering the wire')
        self.add_input('P_in', val=0.0, units='W', desc='Power entering the wire')
        self.add_input('rho', val=0.0, units='ohm*m', desc='Resistivity (resistance per length) of wire')
        self.add_input('dia', val=0.0, units='m', desc='Diameter of the wire')

        self.add_output('I_out', val=0.0, units='A', desc='Current exiting the wire')
        self.add_output('V_out', val=0.0, units='V', desc='Voltage exiting the wire')
        self.add_output('P_out', val=0.0, units='W', desc='Power exiting the wire')
        self.add_output('Qdot_elec', val=0.0, units='W', desc='Heat generated by the wire')
        self.add_output('R', val=0.0, units='ohm', desc='Resistance of full wire length')

        self.declare_partials('R', ['rho', 'length', 'dia'])
        self.declare_partials('I_out', 'I_in', val=1.0)
        self.declare_partials('V_out', 'V_in', val=1.0)
        self.declare_partials('V_out', ['I_in', 'rho', 'length', 'dia'])
        self.declare_partials('P_out', 'P_in', val=1.0)
        self.declare_partials('P_out', ['I_in', 'rho', 'length', 'dia'])
        self.declare_partials('Qdot_elec', ['I_in', 'rho', 'length', 'dia'])

    def compute(self, inputs, outputs):

        outputs['R'] = 4.0 * inputs['rho'] * inputs['length'] / (np.pi * inputs['dia']**2)
        outputs['I_out'] = inputs['I_in']
        outputs['V_out'] = inputs['V_in'] - inputs['I_in'] * outputs['R']
        outputs['P_out'] = inputs['P_in'] - inputs['I_in']**2 * outputs['R']
        outputs['Qdot_elec'] = inputs['I_in'] * outputs['R']

    def compute_partials(self, inputs, J):

        # cs_area = 4.0 / (np.pi * inputs['dia']**2)
        R = 4.0 / (np.pi * inputs['dia']**2) * inputs['rho'] * inputs['length']
        dR_drho = 4.0 / (np.pi * inputs['dia']**2) * inputs['length']
        dR_dlen = 4.0 / (np.pi * inputs['dia']**2) * inputs['rho'] * inputs['length']
        dR_ddia = -8.0 * inputs['rho'] * inputs['length'] / (np.pi * inputs['dia']**3)

        J['R', 'rho'] = dR_drho
        J['R', 'length'] = dR_dlen
        J['R', 'dia'] = dR_ddia

        J['V_out', 'I_in'] = -R
        J['V_out', 'rho'] = -inputs['I_in'] * dR_drho
        J['V_out', 'length'] = -inputs['I_in'] * dR_dlen
        J['V_out', 'dia'] = -inputs['I_in'] * dR_ddia

        J['P_out', 'I_in'] = -2.0 * inputs['I_in'] * R
        J['P_out', 'rho'] = -inputs['I_in']**2 * dR_drho
        J['P_out', 'length'] = -inputs['I_in']**2 * dR_dlen
        J['P_out', 'dia'] = -inputs['I_in']**2 * dR_ddia

        J['Qdot_elec', 'I_in'] = R
        J['Qdot_elec', 'rho'] = inputs['I_in'] * dR_drho
        J['Qdot_elec', 'length'] = inputs['I_in'] * dR_dlen
        J['Qdot_elec', 'dia'] = inputs['I_in'] * dR_ddia

class Wire(Group):
    """
    Group that models an electric wire.
    """

    def initialize(self):
        self.metadata.declare('n', type_=int, default=1, desc='Number of analysis points')
        self.metadata.declare('compute_thermal', default=True, desc='Flag to include thermal mass calculations')

    def setup(self):

        n = self.metadata['n']

        self.add_subsystem('mv', WireMassVolume(), promotes=['*'])
        self.add_subsystem('perf', WirePerf(), promotes=['*'])

        if self.metadata['compute_thermal']:
            self.add_subsystem('therm', ThermalMass(), 
                            promotes_inputs=['area', 'mass', 'T_f', 'T_b', 'h', 'c_p', ('Qdot_in', 'Qdot_elec')],
                            promotes_outputs=['*'])

            self.set_order(['mv', 'perf', 'therm'])

        else:
            self.set_order(['mv', 'perf'])



if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp

    p = Problem()
    p.model = Group()
    des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    des_vars.add_output('length', 1.0, units='m')
    des_vars.add_output('dia', 1.0, units='cm')
    des_vars.add_output('density', 8.96, units='g/cm**3')
    des_vars.add_output('I_in', 20.0, units='A')
    des_vars.add_output('V_in', 50.0, units='V')
    des_vars.add_output('P_in', 1000.0, units='W')
    des_vars.add_output('rho', 1.72E-8, units='ohm*m')
    des_vars.add_output('T_f', 300.0, units='K')
    des_vars.add_output('T_b', 400.0, units='K')
    des_vars.add_output('h', 7.0, units='W/(m**2*K)')
    des_vars.add_output('c_p', 0.385, units='J/(K*g)')

    # p.model.add_subsystem('mv', WireMassVolume(), promotes=['*'])
    # p.model.add_subsystem('perf', WirePerf(), promotes=['*'])
    p.model.add_subsystem('wire', Wire(n=1), promotes=['*'])


    p.setup(check=False)
    p.run_model()

    print('area = ', p['area'][0])
    print('vol = ', p['volume'][0])
    print('mass = ', p['mass'][0])
    print('R = ', p['R'][0])
    print('V_out = ', p['V_out'][0])
    print('I_out = ', p['I_out'][0])
    print('P_out = ', p['P_out'][0])
    print('Qdot_out = ', p['Qdot_out'][0])
    print('dTdt = ', p['dTdt'][0])


    # p.check_partials()
from openmdao.api import ImplicitComponent

class Node(ImplicitComponent):

    def initialize(self):
        self.options.declare('connect_names', default=['in', 'out'], desc='names of electrical connections to the node')

    def setup(self):

        connect_names = self.options['connect_names']

        self.add_output('V', val=1.0, units='V', desc='voltage at node')

        for name in connect_names:
            self.add_input(name+':I', val=0.0, units='A', desc='current from connection')
            self.declare_partials('V', name+':I')

    def apply_nonlinear(self, inputs, outputs, resids):

        connect_names = self.options['connect_names']
        resids['V'] = 0.0

        for name in connect_names:
            resids['V'] += inputs[name+':I']

    def linearize(self, inputs, outputs, J):

        connect_names = self.options['connect_names']

        for name in connect_names:
            J['V', name+':I'] = 1.0
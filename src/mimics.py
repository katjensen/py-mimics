import config
import ground
import util

import importlib
import math
import numpy as np
import os
import shutil



class Mimics:
    _src_path = os.path.dirname(os.path.realpath(__file__))
    _required_inputs = ['sensor.csv', 'ground.csv']

    def __init__(self, simulation_mode, vegetation_cover=False, input_dir=None, output_dir=None,
                 overwrite_output=True):
        self.simulation_mode = simulation_mode
        self.vegetation_cover = vegetation_cover

        ###########################################################################################################
        # Configure input directory, check if necessary input files exist
        ###########################################################################################################
        if input_dir is None:
            # if no input dir is specified, will default to input dir found in mimics source directory
            self.input_dir = os.path.join(self._src_path, '../input')
        else:
            self.input_dir = input_dir
        if self.vegetation_cover:
            self._required_inputs += ['veg_layers.csv', 'veg_scatterers.csv']
        required_inputs = [os.path.join(self.input_dir, i) for i in self._required_inputs]
        for req_in in required_inputs:
            if not os.path.exists(req_in):
                util.CodeError('Required input file [' + os.path.split(req_in)[-1] + '] does not exist in input '
                               'directory. Check if selected input directory is correct and check contents')

        ###########################################################################################################
        # Set up output directory
        ###########################################################################################################
        if output_dir is None:
            # if no output dir is specified, will create one in source directory -- will overwrite any existing one
            self.output_dir = os.path.join(self._src_path, '../output')
        else:
            self.output_dir = output_dir

        if os.path.exists(self.output_dir):
            if overwrite_output:
                shutil.rmtree(self.output_dir)      # Delete existing output directory
                os.makedirs(self.output_dir)        # Create a new, empty directory with same path
            else:
                util.CodeError('"output" directory exists. Either set "overwrite_output" to True '
                               '-- or -- specify a custom new output directory with "output_dir"')
        else:
            os.makedirs(self.output_dir)

        ###########################################################################################################
        # Set up logging & configurations
        ###########################################################################################################
        # Create a logfile
        self.log = open(os.path.join(self.output_dir, 'logfile.txt'), 'a')

        # Load configurations
        if self.simulation_mode == 'snapshot':
            self.config = config.Snapshot(input_dir=self.input_dir, veg_cover=self.vegetation_cover, logfile=self.log)
        elif self.simulation_mode == 'timeseries':
            self.config = config.Timeseries(input_dir=self.input_dir, veg_cover=self.vegetation_cover, logfile=self.log)


    def run(self):
        # Convert Theta degrees --> radians
        self.config.simulations['theta'] = np.deg2rad(self.config.simulations['theta'])
        print self.config.simulations

        """ Execute simulations, iteratively """
        for nsim in xrange(len(self.config.simulations)):
            S = Simulation(self.config.simulations[nsim: nsim + 1])
            o = self.single_simulation(S)

        """  
        for index, row in self.config.simulations.iterrows():
            print '*****', row
            o = self.single_simulation(row)
        """


    def single_simulation(self, simulation):

        # Ground layer --------------------------------------------------------------------------------------------
        if simulation['surface_type'] == 1:
            self.ground = ground.Soil(simulation, self.log)
        elif simulation['surface_type'] == 2:
            self.ground = ground.Water(simulation, self.log)
        elif simulation['surface_type'] == 3:
            self.ground = ground.Ice(simulation, self.log)
        else:
            util.CodeError('Ground Surface type not recognized. Must be integer: 1, 2, or 3', self.log)

        print self.ground.epsilon


        # Vegetation (potentially multi-layered)

        # Organize output
        if self.vegetation_cover:
            # need to implement!
            pass
        else:
            sigma0_dB = np.full((2, 2), -9999.)
            sigma0 = self.ground.back_matrix[:2, :2] * 4.0 * np.pi * np.cos(simulation['theta'], dtype=np.float32)
            valid_mask = (sigma0 > 0)
            sigma0_dB[valid_mask] = util.pow2db(sigma0[valid_mask])
            print "sigma0_dB= ", sigma0_dB


class Simulation():

    def __init__(self, df):
        self.array = df
        self.columns = list(df.columns)

    def __getitem__(self, key):
        return self.array[key].item()


if __name__ == '__main__':
    Test_SS = Mimics(simulation_mode='snapshot', vegetation_cover=False, input_dir='../input/input_test_eps_ground')
    print Test_SS.config.simulations

    Test_SS.run()

    #Test_TS = Mimics(simulation_mode='timeseries', vegetation_cover=False, input_dir='../input/input_test_ts')
    #print Test_TS.config.simulations
    #Test_TS.run()
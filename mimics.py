import ancil
import ground
import importlib
import numpy as np
import os
import shutil


class ConfigLoader():

        def __init__(self, logfile, input_csv=None):
            self.logfile = logfile

            ###########################################################################################################
            ## Import configuration settings
            ###########################################################################################################
            try:
                self.default = importlib.import_module('config')
            except ImportError:
                ancil.CodeError('"config.py" could not be found in module.', self.logfile)

            params_list = [p.lower() for p in self.default.__dict__ if not p.startswith('__')]

            # Import CSV file, if specified --> need to work on this!!
            if input_csv is not None:
                csv_values = np.genfromtxt(input_csv, delimiter=',', names=True, dtype=None)
            else:
                csv_values = None

            #for att in params_list:
            #    setattr(self, att, self.assign_attr(att.upper()))




        def assign_attr(self, attr):
            try:
                return self.args.__dict__[attr]

            except KeyError:
                try:
                    return self.default.__dict__[attr]

                except:
                    print "No values specified for: ", attr
                    return None



class Mimics():
    _mimics_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, output_dir=None, overwrite_output=True):
        # Set up output directory
        if output_dir is None:
            # if no output dir is specified, will create one in mimics directory -- will overwrite any existing one
            self.output_dir = os.path.join(self._mimics_path, 'output')
        else:
            self.output_dir = output_dir

        if os.path.exists(self.output_dir):
            if overwrite_output:
                shutil.rmtree(self.output_dir)      # Delete existing output directory
                os.makedirs(self.output_dir)        # Create a new, empty directory with same path
            else:
                ancil.CodeError('"output" directory exists. Either set "overwrite_output" to True '
                                '-- or -- specify a custom new output directory with "output_dir"')
        else:
            os.makedirs(self.output_dir)

        # Create a logfile
        self.log = open(os.path.join(self.output_dir, 'logfile.txt'), 'a')

        # Load configurations
        config = ConfigLoader(logfile=self.log)
        self.simulations = config.simulations


    def run(self):
        """ need to work on this --> method for executing mimics """


        # Organize set of simulations


    def single_simulation(self, sim_num):

        # Ground layer
        if self.config.ground_surface == 1:
            self.ground = ground.Soil(freq, temp, sand_frac, clay_frac, mv, snow_layer, snow_depth, theta)

        # Trunk layer

        # Crown layer

        # Canopy

        # Organize output
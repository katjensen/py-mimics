import ancil
import ground
import importlib
import os
import shutil


class ConfigLoader():
        def __init__(self, logfile, args):
            self.args = args
            self.path = os.path.dirname(os.path.realpath(__file__))
            self.cwd = os.getcwd()
            self.logfile = logfile

            ###########################################################################################################
            ## Import configuration settings
            ###########################################################################################################
            try:
                self.default = importlib.import_module('config')
            except ImportError:
                ancil.CodeError('"config.py" could not be found in module.', self.logfile)

            # Import CSV file, if specified --> need to work on this!!



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
        self.config = ConfigLoader(logfile=self.log)


    def run(self):
        """ need to work on this --> method for executing mimics """

        # Ground layer
        if self.config.ground_surface == 1:
            self.ground = ground.Soil(freq, temp, sand_frac, clay_frac, mv, snow_layer, snow_depth, theta)

        # Trunk layer

        # Crown layer

        # Canopy

        # Organize output
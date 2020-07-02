import util
import dielectric
import csv
import itertools
import os
import numpy as np
import pandas as pd
import re


###########################################################################################################
## Import configuration settings
###########################################################################################################


class ComplexConverter(dict):
    column_name_pattern = re.compile('eps')

    def __getitem__(self, k):
        if k in self:
            return ComplexConverter.convert
        else:
            raise KeyError(k)

    def __contains__(self, k):
        return self.column_name_pattern.search(k) is not None

    @staticmethod
    def convert(txt):
        return np.complex64(txt)


class ConfigLoader:
    _surface_types = {1:     'soil',
                      2:     'standing_water',
                      3:     'snow'}

    # Valid ranges for all input parameters
    _params = { # SENSOR INPUTS ------------------------------------------------------------------------------------
                'sensor':            {'frequency':      (0.25,   20),
                                     'theta':           (0, 80)},

               # GROUND INPUTS, incl. subsets: SOIL, STANDING_WATER, SNOW ------------------------------------------
               'ground':            {'surface_type':    (1, 3),
                                     'surface_model':   (0, 5)},

               'soil':              {'mv_soil':         (0, 1),
                                     'rms_soil':        (0, 100),
                                     'ls_soil':         (0, 100),
                                     'eps_soil':        (1, 40)},   # check on parameter

               'eps_soil':          {'sand':            (0, 100),
                                     'clay':            (0, 100),
                                     'temp_soil':       (0.1, 40)},  # modify this when frozen conditions implemented

               'standing_water':    {'temp_water':      (0.1, 40),
                                     'salinity':        (0, 10)},

               'snow':              {'snow_depth':       (0, 0)},   # modify this when snow layer is implemented

               # VEGETATION INPUTS ---------------------------------------------------------------------------------
               'veg_layers':        {},

               'veg_scatterers':    {},

               'eps_veg':           {'temp_veg':    (0.1, 40)}}

    def __init__(self, input_dir, veg_cover, logfile, verbose=True):
        self.log = logfile
        self.vegetation_cover = veg_cover
        self.verbose = verbose

        # Ingest & check sensor config parameters
        sensor_file = os.path.join(input_dir, 'sensor.csv')
        self.sensor = pd.read_csv(sensor_file, delimiter=',')
        self.check_columns(self.sensor, 'sensor')

        # Ingest & check ground config parameters
        ground_file = os.path.join(input_dir, 'ground.csv')
        self.ground = pd.read_csv(ground_file, delimiter=',', converters=self.get_converter(ground_file))
        self.check_columns(self.ground, 'ground')

        # Ingest & check vegetation config parameters (if needed)
        if self.vegetation_cover:
            self.veg_layers = pd.read_csv(os.path.join(input_dir, 'veg_layers.csv'), delimiter=',')
            self.check_columns(self.veg_layers, 'veg_layers')

            self.veg_scatterers = pd.read_csv(os.path.join(input_dir, 'veg_scatterers.csv'), delimiter=',')
            self.check_columns(self.veg_scatterers, 'veg_scatterers')


    def get_converter(self, csv_file):
        # Get header
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

        str2cmplx = lambda x: complex(x)
        return {var: str2cmplx for var in header if 'eps' in var}


    def get_relevant_ground_params(self, dat):
        """ Check ground config data if required parameters for specified ground types are present """

        if 'surface_type' in dat.columns.values:
            unique_surfaces = [self._surface_types[s] for s in np.unique(dat['surface_type'])]

        else:
            util.CodeError('GROUND parameter "surface_type" is missing from config file', self.log)


    def get_additional_inputs(self, input_type, available_inputs):
        specified_eps = {'soil': 'eps_soil', 'veg_scatterers': 'eps_veg'}  #
        sub_params = self._params[input_type].keys()

        if input_type in specified_eps:
            if specified_eps[input_type] not in available_inputs:
                util.write_text('"' + specified_eps[input_type] + '" is not provided in config file. Dielectric ' \
                                 'constants will be calculated from specified ' + input_type + ' properties', self.log)
                sub_params.remove(specified_eps[input_type])
                sub_params += self._params[specified_eps[input_type]].keys()
            else:
                util.write_text('Specified "' + specified_eps[input_type] + '" used as the dielectric constants for '
                                + input_type.upper() + ' layer.', self.log)
        return sub_params

    def check_columns(self, dat, config_type):
        """ Check to see if the required parameters are included in datafile, and if so, check if all
            values are valid """
        dat_columns = dat.columns.values
        required_params = self._params[config_type].keys()
        params_dict = util.merge_dictionaries([self._params[p] for p in self._params])    # collapse nested parameters dictionary

        if config_type == 'ground':
            if 'surface_type' in dat.columns.values:
                unique_surfaces = [self._surface_types[s] for s in np.unique(dat['surface_type'])]
                for surf in unique_surfaces:
                    required_params += self.get_additional_inputs(surf, dat_columns)
            else:
                util.CodeError('GROUND input column "surface_type" is missing in config file.', self.log)

        elif config_type == 'veg_scatterers':
            required_params += self.get_additional_inputs(config_type, dat_columns)

        column_mask = np.in1d(required_params, dat_columns)

        if np.sum(column_mask) == len(required_params):
            for param in required_params:
                print param, dat[param].dtype
                values = dat[param][~np.isnan(dat[param])]
                if np.sum((values < params_dict[param][0]) | (values > params_dict[param][1])) > 0:
                    util.CodeError('Invalid value detected in column named "' + param + '" in ' + config_type +
                                    ' config datafile', self.log)
                else:
                    if self.verbose:
                        print config_type + ': all "' + param + '" values = valid'
        else:
            util.CodeError(config_type + ' config file is missing the following required parameters: ' +
                            str(np.array(required_params)[~column_mask]), self.log)





class Snapshot(ConfigLoader):

    def __init__(self, input_dir, veg_cover, logfile):
        ConfigLoader.__init__(self, input_dir, veg_cover, logfile)
        self.simulations = self.organize_simulations()


    def organize_simulations(self):
        """ Create DF of unique combinations of all given config parameters """
        unique_parameters = [self.sensor[p][~np.isnan(self.sensor[p])] for p in self.sensor] + \
                            [self.ground[p][~np.isnan(self.ground[p])] for p in self.ground]
        params_names = [p for p in self.sensor] + [p for p in self.ground]

        if self.vegetation_cover:
            unique_parameters += [self.veg_layers[p][~np.isnan(self.veg_layers[p])] for p in self.veg_layers]
            params_names += [p for p in self.veg_layers]

        combination_tuples = list(itertools.product(*unique_parameters))

        # Combine all pertinent parameters
        return pd.DataFrame(combination_tuples, columns=params_names)
        #sims = sims.assign(id=np.arange(len(sims)))    # add an identifier column
        #return sims


class Timeseries(ConfigLoader):

    def __init__(self, input_dir, veg_cover, logfile):
        ConfigLoader.__init__(self, input_dir, veg_cover, logfile)
        self.simulations = self.organize_simulations()


    def organize_simulations(self):
        """ Create DF from row-wise combinations of parameters """

        # TODO: allow more flexibility in 't' formatting (?)

        time_column = self.check_time()

        if self.check_rows_complete():
            # Remove 't' column from Ground, Sensor
            self.sensor = self.sensor.drop(['t'], axis=1)
            self.ground = self.ground.drop(['t'], axis=1)

            # Combine remaining parameters, assign a single column 't' as index
            sims = pd.concat([self.sensor, self.ground], axis=1)

            if self.vegetation_cover:
                # Remove t
                # ....

                sims = pd.concat([sims, self.veg_layers[self._params['veg_layers'].keys()]], axis=1)

            t = pd.DataFrame({'t': time_column})
            sims = pd.concat([sims, t], axis=1)
            sims = sims.set_index('t')
            return sims

        else:
            return None


    def check_time(self):
        # Check if there is are matching identifier columns 't', infer format ... either int or date (m/d/Y)
        # in all config files

        if ('t' not in list(self.sensor)) or ('t' not in list(self.ground)):
            t_sensor = None
            util.CodeError('Column named "t" is required in SENSOR, GROUND, and VEG_LAYERS config files, '
                            'identifying time stamp for each row', self.log)
        else:
            try:
                t_sensor = [util.str2date(d) for d in self.sensor['t']]
                t_ground = [util.str2date(d) for d in self.ground['t']]

            except TypeError:
                t_sensor = [int(d) for d in self.sensor['t']]
                t_ground = [int(d) for d in self.ground['t']]

            if t_sensor != t_ground:
                util.CodeError('Time identifier column (t) does not match in SENSOR and GROUND config files', self.log)

        """ need to implement this!! """
        if self.vegetation_cover:
            if ('t' not in list(self.veg_layers)):
                util.CodeError('Column named "t" is required in SENSOR, GROUND, and VEG_LAYERS config files, '
                                'identifying time stamp for each row', self.log)
        return np.array(t_sensor)


    def check_rows_complete(self):
        """ Simple check for no NaNs in all parameters """
        for p in list(self.sensor):
            if np.sum(~pd.notnull(self.sensor[p])) > 0:
                util.CodeError('Missing data detected in SENSOR config file for column: ' + p, self.log)
        for p in list(self.ground):
            if np.sum(~pd.notnull(self.ground[p])) > 0:
                util.CodeError('Missing data detected in GROUND config file for column: ' + p, self.log)
        if self.vegetation_cover:
            for p in list(self.veg_layers):
                if np.sum(~pd.notnull(self.veg_layers[p])) > 0:
                    util.CodeError('Missing data detected in VEG_LAYERS config file for column: ' + p, self.log)

        return True

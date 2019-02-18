

#   Options:    (1) Specify a single value,
#               (2) List/numpy array of values,

#######################################################################################################################
### Sensor Parameters
#######################################################################################################################

# ***** Frequency (GHz) *****
FREQUENCY = 1.25

# ***** Incidence angle - theta (degrees) *****
THETA = 30.

#######################################################################################################################
### Ground Parameters
#######################################################################################################################

# ***** Ground surface type *****
#   Options:    1: Soil
#               2: Standing water
#               3: Ice
GROUND_SURFACE = 1

# ***** Presence of snow layer on top *****
#   Options: True or False
SNOW_LAYER = False

# ***** Soil volumetric moisture content *****
#   (water weight in sample) / (volume of sample), 0 - 1
MV_SOIL = 0.2

# ***** RMS height (cm) *****
RMS_GROUND = 1.

# ***** Correlation length (cm) *****
LS_GROUND = 15.

# ***** Soil Texture *****
#   Percent (by weight) sand, silt, and clay; %Sand + %Silt %Clay = 100%
#   Percent silt is inferred from other two
PERC_SAND = 20
PERC_CLAY = 20

# ***** Salinity of standing water (ppt) *****
SALINITY = 2.50

# ***** Snow layer depth (thickness) (m) *****
SNOW_DEPTH = 0.

# ***** Surface scattering model descriptors *****
#   Options:    1:  Specular
#               2:  Geometrical Optics
#               3:  Physical Optics
#               4:  Small Perturbation
#               5:  UMich Empirical
SURFACE_MODEL = 1

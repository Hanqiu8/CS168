import re

mha_re = re.compile('.*mha$')
phase_re = [
    re.compile('.*Pre-Contrast.*'),
    re.compile('.*Coricomedullary.*'),
    re.compile('.*Nephrographic.*'),
    re.compile('.*Excret.*')
]

# Location of image data
data_dir = '/media/hansen/Data/data/'

# Save path to feature folder
features_path = './featuresHistoEqu/'

# Location of normalization value csv
csv_file_location = './RCC_normalization_values.csv'

# Phase names used in csv
phase_type_csv = [ 'Pre-contrast CORTEX',
				   'Corticomedullary CORTEX', 
				   'Nephrtographic CORTEX', 
				   'Excretory CORTEX' ]

# Tumor types in csv to binary
tumor_types = { 'oncocytoma': 0, 'Clear Cell RCC': 1 }

# Number of phases used
num_phases = len(phase_re)

# Update if change/add features
num_features = 6

# Decides to use cache features
cache = True
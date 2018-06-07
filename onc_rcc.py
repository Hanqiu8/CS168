from scipy import interp, stats
import SimpleITK as sitk, numpy, scipy.io, scipy.ndimage, pylab, os, re, csv, math
import pandas as pd

from filters import *
from myconstants import *
from classifiers import *

# Taken from https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[numpy.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=numpy.sum([p*numpy.log2(1.0/p) for p in propab])
        return ent


# Get data from csv and set up itk
csv_file = open(csv_file_location, 'rU') 
csv_reader = csv.reader(csv_file)
csv_dataframe = pd.read_csv(csv_file_location, sep = ',', index_col = "SUBJECT ID")
itk_reader = sitk.ImageSeriesReader()

# Don't want header row (https://stackoverflow.com/questions/16108526/count-how-many-lines-are-in-a-csv-python)
num_samples = sum(1 for row in csv_reader) - 1

# Go back to start of file
csv_file.seek(0) 
csv_reader.next()

# Set up features and ground actualTum numpy arrays
features = numpy.zeros((num_samples, num_phases * num_features))
actualTum = numpy.zeros(num_samples)

# Get all patient directory names

patients = []
if os.path.isdir(data_dir):
    patients = os.listdir(data_dir)
else:
    print "Data directory is wrong! Crashing now"
    exit()

# Used to keep track of feature numpy arrays
patientIndex = 0
for row in csv_reader:
    # Get tumor type, patient id, and record ground actualTum for each patient
    tumorClass = row[0]
    patID = row[1]
    actualTum[patientIndex] = tumorTypes[tumorClass]

    # Use feature cache if set to True and already cached
    cache_filename = features_path + patID + '.npy'
    if cache and os.path.isfile(cache_filename):
        features[patientIndex] = numpy.load(cache_filename)
        patientIndex += 1
        continue

    # Match patID with patient data
    patient_re = re.compile('.*' + patID + '.*')
    try:
        patient_dir = filter(patient_re.match, patients)[0]
    except IndexError:
        print "Error: Folder " + patID + " is not in data folder"
        continue
   
    phases = os.listdir(data_dir + patient_dir)
    phase_index = 0
    for p_re, cortex in zip(phase_re, phase_type_csv):
        phase_dir = filter(p_re.match, phases)[0]
        
        # Grabbing mask
        try:
            maskLoc = data_dir + patient_dir + '/' + phase_dir
            maskLabel = filter(mha_re.match, os.listdir(maskLoc))[0]
        except IndexError:
            print "Error: .mha file at " + maskLoc + " is missing!"
            break
        
        print "     "
        
        full_path = maskLoc + '/' + maskLabel
        print "Getting  data from " + full_path
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(full_path))

        # grab image
        imgNames = itk_reader.GetGDCMSeriesFileNames(maskLoc + '/images')
        itk_reader.SetFileNames(imgNames)
        pre_img = numpy.flipud(sitk.GetArrayFromImage(itk_reader.Execute()))

        # filter Image
        im_filter = ImageFilter(pre_img)
        # img = pre_img
        img = im_filter.histoEqualization(toShow=False, z = 0)
        # img = im_filter.deconvolution()

        #get normalized values
        normalized = 0
        if patID in csv_dataframe.index.values:
            normalized = csv_dataframe.loc[patID, cortex]
        else:
            print 'Error: No normalized value for: ' + patID
            break

        maskedImgs = numpy.multiply(img, mask)
        flatImgs = numpy.ndarray.flatten(img[mask!=0])

        max_roi, roi = im_filter.truncationROIfinder(mask, maskedImgs, normalized)

        # Use for debugging purposes
        # print "ROI Intensity:"
        # print roi

        print "Features - " + tumorClass + ":"
        
        print "#1: Peak ROI relative intensity:",
        features[patientIndex][phase_index] = max_roi
        print features[patientIndex][phase_index], "%"
        
        print "#2: Entropy:",
        features[patientIndex][phase_index + num_phases * 1] = entropy(flatImgs)
        print features[patientIndex][phase_index + num_phases * 1]
        
        print "#3: Standard deviation:",
        features[patientIndex][phase_index + num_phases * 2] = numpy.std(flatImgs)
        print features[patientIndex][phase_index + num_phases * 2]
        
        print "#4: Inter-quartile range:",
        features[patientIndex][phase_index + num_phases * 3] = scipy.stats.iqr(flatImgs)
        print features[patientIndex][phase_index + num_phases * 3]
        
        print "#5: Kurtosis:",
        features[patientIndex][phase_index + num_phases * 4] = scipy.stats.kurtosis(flatImgs)
        print features[patientIndex][phase_index + num_phases * 4]

        print "#6: Skew:",
        features[patientIndex][phase_index + num_phases * 5] = scipy.stats.skew(flatImgs)
        print features[patientIndex][phase_index + num_phases * 5]

        # Decided to not use this feature in final classifier
        # print "#7 - 2nd Statistic:",
        # features[patientIndex][phase_index + num_phases * 6] = scipy.stats.kstat(flatImgs)
        # print features[patientIndex][phase_index + num_phases * 6]

        phase_index += 1
        
    if cache:
        numpy.save(features_path + patID, features[patientIndex])
    patientIndex += 1

print "===Training==="

cf = imgClassifier()
cf.classify_and_plot(features, actualTum)
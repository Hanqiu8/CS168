# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import roc_curve, auc

from skimage import exposure, feature, measure
from skimage.morphology import disk
from skimage.filters import rank, gabor_kernel
from skimage import color, data, restoration

import pydicom

from itertools import cycle
from scipy import ndimage as ndi
from scipy import interp, stats
import SimpleITK as sitk, numpy, scipy.io, scipy.ndimage, pylab, os, re, csv, math
import matplotlib.pyplot as plt, pandas as pd
import numpy as np

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

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

# Regex for mha files
mha_re = re.compile('.*mha$')

# Regex for each phase
phase_re = [
    re.compile('.*Pre-contrast.*'),
    re.compile('.*Corticomedullary.*'),
    re.compile('.*Nephrtographic.*'),
    re.compile('.*Excretory.*')
]

# Save path to feature folder
features_path = '../features/'

# Phase names used in csv
phase_type_csv = [ 'Pre-contrast CORTEX', 'Corticomedullary CORTEX', 'Nephrtographic CORTEX', 'Excretory CORTEX' ]

# Tumor types in csv to binary
tumor_types = { 'oncocytoma': 0, 'Clear Cell RCC': 1 }

# Number of phases we use to train
num_phases = len(phase_re)

# Be sure to update this if change features
num_features = 6

# Get data from csv and set up itk
csv_file = open('./RCC_normalization_values.csv', 'rU') 
csv_reader = csv.reader(csv_file)
csv_dataframe = pd.read_csv('./RCC_normalization_values.csv', sep = ',', index_col = "SUBJECT ID")
itk_reader = sitk.ImageSeriesReader()

# Don't want header row (https://stackoverflow.com/questions/16108526/count-how-many-lines-are-in-a-csv-python)
num_samples = sum(1 for row in csv_reader) - 1
csv_file.seek(0) 
csv_reader.next()

# Set up features and ground truth numpy arrays
features = numpy.zeros((num_samples, num_phases * num_features))
truth = numpy.zeros(num_samples)

# Use feature cache
cache = True

# Get all patient directory names
data_dir = '/media/hansen/Data/data/'
patients = []   
if os.path.isdir(data_dir):
    patients = os.listdir(data_dir)
else:
    print "Data directory is wrong! Crashing now"
    exit()


# Used to keep track of feature numpy arrays
patient_index = 0
for row in csv_reader:
    # Get tumor type, patient id, and record ground truth for each patient
    tumor_type = row[0]
    patient_id = row[1]
    truth[patient_index] = tumor_types[tumor_type]

    # Use feature cache if set to True and already cached
# <<<<<<< HEAD
    feature_cache_filename = './featuresLocEqu/' + patient_id + '.npy'
    if cache and os.path.isfile(feature_cache_filename):
        features[patient_index] = numpy.load(feature_cache_filename)
        patient_index += 1
        continue
# =======
#     cache_filename = features_path + patient_id + '.npy'
#     if cache and os.path.isfile(cache_filename):
#         features[i] = numpy.load(cache_filename)
#         patient_index += 1
# >>>>>>> 31028bcae47033275efb192e4ce0c9c1858e3a0c
#         continue

    # Otherwise calculate features for patient

    # Match patient_id with patient data folder
    patient_re = re.compile('.*' + patient_id + '.*')
    try:
        patient_dir = filter(patient_re.match, patients)[0]
    except IndexError:
        print "Error: Folder " + patient_id + " is not in data folder"
        continue
   
    # Get 4 phases for patient
    phases = os.listdir(data_dir + patient_dir)

    phase_index = 0

    for p_re, cortex in zip(phase_re, phase_type_csv):
        try:
            phase_dir = filter(p_re.match, phases)[0]
        except IndexError:
            print "Error: Phase " + phase_dir + " is missing"
            break
        
        # Use mask from .mha file
        try:
            mask_path = data_dir + patient_dir + '/' + phase_dir
            mask_name = filter(mha_re.match, os.listdir(mask_path))[0]
        except IndexError:
            print "Error: .mha file at " + mask_path + " is missing!"
            break
        
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        
        full_path = mask_path + '/' + mask_name
        print "Getting  data from " + full_path
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(full_path))

        # Raw image
        img_names = itk_reader.GetGDCMSeriesFileNames(mask_path + '/images')
        itk_reader.SetFileNames(img_names)
        
        # Flip z axis for DICOM
        img = numpy.flipud(sitk.GetArrayFromImage(itk_reader.Execute()))
        img_rescale = img  
        
        from scipy.signal import convolve2d as conv2
        # img = color.rgb2gray(img)
        # psf = np.ones((10,10))/100
        # imgd = conv2(img[0], psf, 'same')
        # imgd += 0.1 * imgd.std() * np.random.standard_normal(imgd.shape)
        # deconvolved, _ = restoration.unsupervised_wiener(imgd, psf)

        pylab.imshow(img[24], cmap=pylab.cm.bone)
        pylab.show()
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
        # print img
        # for i in range(len(img)):
        #     img[i] = measure.find_contours(img[i], 0.8)
        



        # Global Histogram Equalization
        # img_rescale = exposure.equalize_hist(img)
        # selem = disk(30)
        # img_rescale = rank.equalize(img, selem = selem)

        # feats = compute_feats(ndi.rotate(img, angle=190, reshape=True), kernels)

        #find normalized cortex values to facilitate computation of relative enhancement
        normalized = 0
        if patient_id in csv_dataframe.index.values:
            normalized = csv_dataframe.loc[patient_id, cortex]
        else:
            print 'Error: No normalized value for: ' + patient_id
            break

        # Identify lesion
        non_zeros = numpy.where(mask!=0)
        start = non_zeros[0][0]
        end = non_zeros[0][-1]
        print "Lesion: "+ str(start) + "-" + str(end)

        # Segment lesion
        masked_img_arr = numpy.multiply(img_rescale, mask)

        # psf = np.ones((10,10))/100
        # imgd = conv2(img[0], psf, 'same')
        # imgd += 0.1 * imgd.std() * np.random.standard_normal(imgd.shape)
        # deconvolved, _ = restoration.unsupervised_wiener(imgd, psf)

       
        flat_img_arr = numpy.ndarray.flatten(img_rescale[mask!=0])
       
        #extract 3x3x3 ROI with max intensity
        max_roi = -float('inf')
        roi = []
        
        for z,y,x in zip(non_zeros[0], non_zeros[1], non_zeros[2]):
            acc = []
            valid = True
            
            for _z in range(z, min(end, z + 3)):
                for _y in range(y, y + 3):
                    for _x in range(x, x + 3):
                        # Invalidate option if mask is 0 or value in masked image array is too big
                        if mask[_z][_y][_x] == 0 or masked_img_arr[_z][_y][_x] > 300:
                            valid = False
                        else:
                            acc.append(masked_img_arr[_z][_y][_x])
            
            # Check if we found a new max intesity ROI
            if valid and len(acc) >= 9 * (min(end - start, 3)) and len(acc) != 0:
                avg = numpy.mean(numpy.asarray(acc))
                if avg > max_roi:
                    max_roi = avg
                    roi = numpy.asarray(acc[:])

        if max_roi != -float('inf'):
            max_roi = (max_roi - normalized) / normalized * 100
        else:
            # If this failed somehow take 90th percentile and normalize
            max_roi = (numpy.percentile(flat_img_arr, 90) - normalized) / normalized * 100

        print "ROI Intensity:"
        print roi

        print "Features - " + tumor_type + ":"
        
        print "#1 - Peak ROI relative intensity:",
        features[patient_index][phase_index] = max_roi
        print features[i][phase_index], "%"
        
        print "#2 - Entropy:",
        features[patient_index][phase_index + num_phases * 1] = entropy(flat_img_arr)
        print features[patient_index][phase_index + num_phases * 1]
        
        print "#3 - Standard deviation:",
        features[patient_index][phase_index + num_phases * 2] = numpy.std(flat_img_arr)
        print features[patient_index][phase_index + num_phases * 2]
        
        print "#4 - Inter-quartile range:",
        features[patient_index][phase_index + num_phases * 3] = scipy.stats.iqr(flat_img_arr)
        print features[patient_index][phase_index + num_phases * 3]
        
        print "#5 - Kurtosis:",
        features[patient_index][phase_index + num_phases * 4] = scipy.stats.kurtosis(flat_img_arr)
        print features[patient_index][phase_index + num_phases * 4]

        print "#6 - Skew:",
        features[patient_index][phase_index + num_phases * 5] = scipy.stats.skew(flat_img_arr)
        print features[patient_index][phase_index + num_phases * 5]

        # print "#7 - 2nd Statistic:",
        # features[patient_index][phase_index + num_phases * 6] = scipy.stats.kstat(flat_img_arr)
        # print features[patient_index][phase_index + num_phases * 6]

        phase_index += 1
        
    if cache:
        numpy.save('./featuresLocEqu/'+ patient_id, features[i])
    patient_index += 1


print "ccRCC mean peak ROI relative attenuation: " + str(numpy.mean(features[truth == 1]))
print "Oncocytoma mean peak ROI relative attenuation:" + str(numpy.mean(features[truth == 0]))

print "Training"

# Could use GridSearchCV to pick params for each

classifiers = {
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors = 7, algorithm = "auto"),
    "SVC (Linear Kernel)": SVC(kernel = "linear", C = 0.01, probability = True),
    "SVC": SVC(C = 0.01, probability = True),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start = True),
    "Decision Tree": DecisionTreeClassifier(max_depth = 1),
    "Random Forest": RandomForestClassifier(max_depth = 2, n_estimators = 5, random_state = 1),
    "Multi-layer Perception": MLPClassifier(hidden_layer_sizes = (15,15,), alpha = 0.1, activation = 'tanh', solver = 'lbfgs'),
    "AdaBoost": AdaBoostClassifier(n_estimators = 5),
    "Gaussian Naive-bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators = 10, learning_rate = 1.0, max_depth = 1, random_state = 0),
    # Using Voting Classifier to perform majority vote
    "Voting Classifier": VotingClassifier(
        estimators = [
            ("K Nearest Neighbors", KNeighborsClassifier(n_neighbors = 7, algorithm = "auto")),
            ("SVC (Linear Kernel)", SVC(kernel = "linear", C = 0.01, probability = True)),
            ( "SVC", SVC(C = 0.01, probability = True)),
            ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
            ("Decision Tree", DecisionTreeClassifier(max_depth = 1)),
            ("Random Forest", RandomForestClassifier(max_depth = 2, n_estimators = 5, random_state = 1)),
            ("Multi-layer Perception", MLPClassifier(hidden_layer_sizes = (15, 15, ), alpha = 0.1, activation = 'tanh', solver = 'lbfgs')),
            ("AdaBoost", AdaBoostClassifier(n_estimators = 5)),
            ("Gaussian Naive-bayes", GaussianNB()),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators = 10, learning_rate = 1.0, max_depth = 1, random_state = 0)),
        ], voting = 'soft', flatten_transform = True)
}

# Ensure each fold has its own color
# Can try StratifiedShuffleSplit instead
cv = StratifiedKFold(n_splits = 5)
colors = cycle(['green', 'orange', 'blue', 'red', 'brown'])

for classifier in classifiers:
    # ROC evaluation with cross validation taken from scikit-learn
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    i = 0
    tprs = []
    aucs = []
    mean_fpr = numpy.linspace(0,1,100)
    print "Training " + classifier
    for (train, test), color in zip(cv.split(features,truth), colors):
        # print "# Train # Test: " + str(len(train)), str(len(test))
        probs = classifiers[classifier].fit(features[train], truth[train]).predict_proba(features[test])
        
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(truth[test], probs[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 2, alpha = 0.4, color = color, label='Fold: %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    
    print "Graph for " + classifier + ":"
    
    # Final alterations before plotting
    mean_tpr = numpy.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = numpy.std(aucs)
    
    # Graph shows 95% confidence interval in grey
    std_tpr = numpy.std(tprs, axis = 0)
    tprs_upper = numpy.minimum(mean_tpr + 2 * std_tpr, 1)
    tprs_lower = numpy.maximum(mean_tpr - 2 * std_tpr, 0)
    
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = .2, label = '95% Confidence Interval')
    plt.plot([], [], color='green', linewidth=10)
    plt.plot([0, 1], [0, 1], alpha = 0.8, linestyle='--', lw = 2, color = 'black', label = 'Luck')
    plt.plot(mean_fpr, mean_tpr, color='darkblue', linestyle='--', label='Mean ROC (AUC = %0.2f %s %0.2f)' % (mean_auc, u'±', 2 * std_auc), lw = 4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + classifier + '\n Oncocytoma vs. Clear Cell RCC')
    plt.legend(loc = "lower right")
    plt.subplots_adjust(top = 0.85)
    plt.show()
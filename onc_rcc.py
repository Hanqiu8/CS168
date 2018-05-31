from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from itertools import cycle
from scipy import interp, stats

import SimpleITK as sitk, numpy, scipy.io, scipy.ndimage, pylab, os, re, csv, math
import matplotlib.pyplot as plt, pandas as pd

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

# Regex for mha files
mha_re = re.compile('.*mha$')

# Regex for each phase
phase_re = [
    re.compile('.*Pre-Contrast.*'),
    re.compile('.*Coricomedullary.*'),
    re.compile('.*Nephrographic.*'),
    re.compile('.*Excret.*')
]

# Phase names used in csv
phase_type_csv = [ 'Pre-contrast CORTEX', 'Corticomedullary CORTEX', 'Nephrtographic CORTEX', 'Excretory CORTEX' ]

# Tumor types in csv to binary
tumor_types = { 'oncocytoma': 0, 'Clear Cell RCC': 1 }

num_phases = len(phase_re)

num_features = 5

# Get data from csv and set up itk
csv_file = open('./RCC_normalization_values.csv', 'rU') 
csv_reader = csv.reader(csv_file)
csv_dataframe = pd.read_csv('./RCC_normalization_values.csv',sep=',',index_col="SUBJECT ID")
itk_reader = sitk.ImageSeriesReader()

# Don't want header row
num_samples = sum(1 for row in csv_reader) - 1

# Go back to start of file
csv_file.seek(0) 
csv_reader.next()

# Set up features and ground truth
features = numpy.zeros((num_samples, num_phases * num_features))
truth = numpy.zeros(num_samples)

# Decides whether or not to use feature cache
cache = True

# Get all patient directory names
data_dir = '../data/'
patients = []
if os.path.isdir(data_dir):
    patients = os.listdir(data_dir)
else:
    print "Data directory is wrong! Crashing now"
    exit()


 
i = 0
for row in csv_reader:
    tumor_type = row[0]
    patient_id = row[1]
    truth[i] = tumor_types[tumor_type]

    # Use feature cache if set to True and already cached
    feature_cache_filename = '../features/' + patient_id + '.npy'
    if cache and os.path.isfile(feature_cache_filename):
        features[i] = numpy.load(feature_cache_filename)
        i += 1
        continue

    # Otherwise calculate features for patient

    # Match patient_id with patient data folder


    # BEWARE THAT PATIENT ID PROBABLY HAS DATE ALONG WITH IT???
    

    patient_re = re.compile('.*' + patient_id + '.*')
    try:
        patient_dir = filter(patient_re.match, patients)[0]
    except IndexError:
        print "Error: Folder " + patient_id + " is not in data folder"
        continue
   
    # Get 4 phases for patient
    phases = os.listdir(data_dir + patient_dir)

    j = 0
    for p_re, cortex in zip(phase_re, phase_type_csv):
        try:
            phase_dir = filter(p_re.match, phases)[0]
        except IndexError:
            print "Error: Phase "+ phase_dir +" is missing"
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

        #find normalized cortex values to facilitate computation of relative enhancement
        normalized = 0
        if patient_id in csv_dataframe.index.values:
            normalized = csv_dataframe.loc[patient_id, cortex]
            print normalized
        else:
            print 'Error: No normalized value for: ' + patient_id
            break

        # Identify lesion
        non_zeros = numpy.where(mask!=0)
        start = non_zeros[0][0]
        end = non_zeros[0][-1]
        print "Lesion: "+ str(start) + "-" + str(end)

        # Segment lesion
        masked_img_arr = numpy.multiply(img, mask)
        flat_img_arr = numpy.ndarray.flatten(img[mask!=0])

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
            if valid and len(acc) >= 9*(min(end - start, 3)) and len(acc) != 0:
                avg = numpy.mean(numpy.asarray(acc))
                if avg > max_roi:
                    max_roi = avg
                    roi = numpy.asarray(acc[:])

        if max_roi != -float('inf'):
            max_roi = (max_roi - normalized)/ normalized * 100
        else:
            # If this failed somehow take 90th percentile and normalize
            max_roi = (numpy.percentile(flat_img_arr, 90) - normalized) / normalized * 100

        print "ROI Intensity:"
        print roi

        print "Features - " + tumor_type + ":"
        
        print "#1 - Peak ROI relative intensity:",
        features[i][j] = max_roi
        print features[i][j], "%"
        
        print "#2 - Standard deviation:",
        features[i][j + num_phases * 1] = numpy.std(flat_img_arr)
        print features[i][j + num_phases * 1]
        
        print "#3 - Entropy:",
        features[i][j + num_phases * 2] = entropy(flat_img_arr)
        print features[i][j + num_phases * 2]
        
        print "#4 - Kurtosis:",
        features[i][j + num_phases * 3] = scipy.stats.kurtosis(flat_img_arr)
        print features[i][j + num_phases * 3]
        
        print "#5 - Inter-quartile range:",
        features[i][j + num_phases * 4] = scipy.stats.iqr(flat_img_arr)
        print features[i][j + num_phases * 4]
        
        # Debugging code to display images side by side
        # f = pylab.figure()
        # f.add_subplot(1,2,1)
        # pylab.imshow(img[start + 2],cmap=pylab.cm.bone)
        # f.add_subplot(1,2,2)
        # pylab.imshow(masked_img_arr[start + 2],cmap=pylab.cm.bone)
        # print "Slice "+str(start+2)
        # pylab.show()
        # j = j + 1
    if cache:
        numpy.save('../features/'+ patient_id, features[i])
    i += 1

print "ccRCC mean peak ROI relative attenuation: " + numpy.mean(features[truth == 1])
print "Oncocytoma mean peak ROI relative attenuation:" + numpy.mean(features[truth == 0])

print "Training..... :)"

classifiers = {
    "K Nearest Neighbors": KNeighborsClassifier(17),
    "SVC (Linear Kernel)": SVC(kernel="linear", C=0.01,probability=True),
    "SVC": SVC(C=0.01,probability=True),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=1),
    "Random Forest": RandomForestClassifier(max_depth=2, n_estimators=3),
    "Multi-layer Perception": MLPClassifier(hidden_layer_sizes=(15,15,),alpha=0.1,activation='tanh',solver='lbfgs'),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian Naive-bayes": GaussianNB(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
}


crossValidator = StratifiedKFold(n_splits=10)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','red','magenta','pink','purple','teal'])

for classifier in classifiers:
    # ROC evaluation with cross validation taken from scikit-learn
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    i = 0
    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0,1,100)
    print "Training " + classifier + "..."
    for (train, test), color in zip(crossValidator.split(features,truth), colors):
        probs = classifiers[classifier].fit(features[train], truth[train]).predict_proba(features[test])
        
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(truth[test], probs[:,1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        area_under_curve = auc(fpr,tpr)
        #plt.plot(fpr, tpr, lw=2, color=color, label='ROC fold %d (area = %0.2f)' % (i, area_under_curve))
        i = i + 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    
    print "Results for " + classifier + " (opens in new window):"
    
    # Do final stats changes
    mean_tpr /= crossValidator.get_n_splits(features, truth)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + classifier + '\n Oncocytoma vs. Clear Cell RCC')
    plt.legend(loc="lower right")
    plt.subplots_adjust(top=0.85)
    plt.show()
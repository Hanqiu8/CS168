import SimpleITK, numpy, scipy.io, scipy.ndimage, pylab, os, re, csv, math
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
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
from scipy import interp

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

NUM_PHASES = 1
NUM_FEATURES = 5

pathologyCSV=open('./RCC_normalization_values_cc_onc.csv') #pathology data
csvReader=csv.reader(pathologyCSV)
normalizedValue_df=pd.read_csv('./RCC_normalization_values_cc_onc.csv',sep=',',index_col="SUBJECT ID")
column_names=' '.join(normalizedValue_df.columns.values)

reader = SimpleITK.ImageSeriesReader()
NUM_SAMPLES = sum(1 for row in csvReader)-1 #subtract the first header row
pathologyCSV.seek(0) #go back to beginning of file

#this vector will contain the features of each tumor
features = numpy.zeros((NUM_SAMPLES,NUM_PHASES*NUM_FEATURES))
#this vector will contain the ground truth tumor type of each tumor
truth = numpy.zeros(NUM_SAMPLES)

# Set to false to re-extract features from the image dataset
use_feature_cache = True
#If we're re-extracting features, then script must run in same directory as the main data dir
DATADIR = './data/'
if os.path.isdir(DATADIR):
    patients=os.listdir(DATADIR)
else:
    patients = [] #placeholder so script doesn't crash

#Regular expression for the segmentation file
mhaRE=re.compile('.*mha$')

#Regular expression for the subdirectories containing phase-specific image data
phase_REs = [
#re.compile('.*Pre-Contrast.*'),
re.compile('.*Coricomedullary.*'),
#re.compile('.*Nephrographic.*'),
#re.compile('.*Excret.*')
]

#column names from the csv file containing tumor histology and normalized cortex intensities
phase_type_in_normal = [
#'Pre-contrast CORTEX',
'Corticomedullary CORTEX',
#'Nephrtographic CORTEX',
#'Excretory CORTEX'
]

#Map tumor types from histology csv file to a binary value
tumorTypes={
'oncocytoma':0,
'Clear Cell RCC':1,
}

csvReader.next() # skip the first row (histology data starts on second row)

i=0
#for each patient
for row in csvReader:
    patient_id=row[1]
    tumor_type=row[0]
    #store the true tumor type in appropriate entry in truth vector
    truth[i]=tumorTypes[tumor_type]
    #check if we've already extracted features for this patient; if so,
    #then simply load the features instead of reprocessing the image
    if use_feature_cache and os.path.isfile('./features_cortico_peakROI_and_texture/'+patient_id+'.npy'):
        features[i]=numpy.load('./features_cortico_peakROI_and_texture/'+patient_id+'.npy')
        i=i+1
        continue;

    #match patient ID in CSV with a subdirectory in the data folder
    patientRE=re.compile('.*'+patient_id +'.*')
    try:
        patientDir=filter(patientRE.match,patients)[0]
    except IndexError:
        print "Warning: Patient "+ patient_id +" from CSV doesn't exist in dataset!"
        continue;
    #get the directories that contain 4 phases for each patient
    phases=os.listdir(DATADIR + patientDir)
    #for each phase used for feature extraction
    j=0
    for phase_re,cortex in zip(phase_REs,phase_type_in_normal):
        #filter the phase
        try:
            phaseDir = filter(phase_re.match, phases)[0]
        except IndexError:
            print "Warning: Patient "+ row[1] +" from CSV has missing phase(s)!"
            break;
        #get mask, which is a .mha file in the directory for the phase
        try:
            maskName = filter(mhaRE.match, os.listdir(DATADIR + patientDir+'/'+phaseDir))[0]
        except IndexError:
            print "Warning: Patient "+ row[1] +" from CSV has no segmentation .mha file!"
            break;
        print "-------------------------"
        print "Loading patient data from "+DATADIR + patientDir+'/'+phaseDir+'/'
        mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(DATADIR + patientDir+'/'+phaseDir+'/'+maskName))

        #get raw image
        imgNames=reader.GetGDCMSeriesFileNames(DATADIR + patientDir+'/'+phaseDir+'/images')
        reader.SetFileNames(imgNames)
        #note: we use flipud() to flip the dicom series along the Z axis
        #because the image series has a different Z-orientation than the .mha
        #binary mask file
        image=numpy.flipud(SimpleITK.GetArrayFromImage(reader.Execute()))

        normalized_val = 0
        #find normalized cortex values to facilitate computation of relative enhancement
        if patient_id in normalizedValue_df.index.values:
            normalized_val = float(normalizedValue_df.loc[patient_id,cortex])
        else:
            print 'could not find normalized value for patient ' + patient_id
            break;

        nonzeros=numpy.where(mask!=0)
        #identify first and last Z slice where not all pixels are zeros
        startSlice=nonzeros[0][0]
        endSlice=nonzeros[0][-1]
        print "Lesion at slices "+ str(startSlice)+"-"+str(endSlice)

        #multiply mask+image together to segment lesion
        maskedImgArr = numpy.multiply(image,mask)
        flattenedImgArr = numpy.ndarray.flatten(image[mask!=0])

        #extract cubic ROI 3x3x3 voxels of max avg intensity
        maxROI = -999
        ROI = []
        for z,y,x in zip(nonzeros[0],nonzeros[1],nonzeros[2]):
            accumulator = []
            valid = True
            for zz in range(z, min(endSlice,z+3)):
                for yy in range(y,y+3):
                    for xx in range(x,x+3):
                        if mask[zz][yy][xx] == 0 or maskedImgArr[zz][yy][xx] > 300:
                            valid = False
                        else:
                            accumulator.append(maskedImgArr[zz][yy][xx])
            if valid and len(accumulator) != 0 and len(accumulator) >= 9*(min(endSlice-startSlice,3)):
                avg = numpy.mean(numpy.asarray(accumulator))
                if avg > maxROI:
                    maxROI = avg
                    ROI = numpy.asarray(accumulator[:])

        if maxROI != -999:
            maxROI = (maxROI - normalized_val)/normalized_val * 100
        else: # if ROI extraction fails, take the 95th percentile of intensity values (should not happen)
            maxROI = (numpy.percentile(flattenedImgArr,95)-normalized_val)/normalized_val * 100

        print "Intensity values for extracted ROI:"
        print ROI

        print "Features for " + tumor_type + ":"
        print "Feature #1 - Peak ROI relative intensity:",
        features[i][j] = maxROI
        print features[i][j],"%"
        print "Feature #2 - Standard deviation:",
        features[i][j+NUM_PHASES*1] = numpy.std(flattenedImgArr)
        print features[i][j+NUM_PHASES*1]
        print "Feature #3 - Entropy:",
        features[i][j+NUM_PHASES*2] = entropy(flattenedImgArr)
        print features[i][j+NUM_PHASES*2]
        print "Feature #4 - Kurtosis:",
        features[i][j+NUM_PHASES*3] = scipy.stats.kurtosis(flattenedImgArr)
        print features[i][j+NUM_PHASES*3]
        print "Feature #5 - Inter-quartile range:",
        features[i][j+NUM_PHASES*4] = scipy.stats.iqr(flattenedImgArr)
        print features[i][j+NUM_PHASES*4]
        # the following code displays the raw/segmented image side by side for
        # debugging purposes
        # uncomment it to see a slice of image data from each patient as they are
        #loaded
        # f=pylab.figure()
        # f.add_subplot(1,2,1)
        # pylab.imshow(image[startSlice+2],cmap=pylab.cm.bone)
        # f.add_subplot(1,2,2)
        # pylab.imshow(maskedImgArr[startSlice+2],cmap=pylab.cm.bone)
        # print "Showing slice "+str(startSlice+2)
        # pylab.show()
        # j = j + 1
    if use_feature_cache:
        numpy.save('./features_cortico_peakROI_and_texture/'+patient_id,features[i])
    i = i + 1



#binary Train classifiers
print "Training....."

classifiers = {
    "K Nearest Neighbors":KNeighborsClassifier(28),
    "SVC (Linear Kernel)":SVC(kernel="linear", C=0.01,probability=True),
    "SVC": SVC(C=0.01,probability=True),
    "Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree":DecisionTreeClassifier(max_depth=7),
    "Random Forest":RandomForestClassifier(max_depth=8, n_estimators=23),
    "Multi-layer Perception":MLPClassifier(hidden_layer_sizes=(10,10,),alpha=0.3,activation='relu',solver='lbfgs'),
    "AdaBoost":AdaBoostClassifier(),
    "Gaussian Naive-bayes":GaussianNB(),
    "Quadratic Discriminant Analysis":QuadraticDiscriminantAnalysis()
}


crossValidator = StratifiedKFold(n_splits=10)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','red','magenta','pink','purple'])

for classifier in classifiers:
    # combine ROC evaluation with cross validation
    #drawn from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    i=0
    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0,1,100)
    print "Training " + classifier + "..."
    #for each fold
    for (train, test),color in zip(crossValidator.split(features,truth),colors):
        #train classifier
        probas = classifiers[classifier].fit(features[train], truth[train]).predict_proba(features[test])
        #computer roc curve for this fold
        fpr,tpr,thresholds = roc_curve(truth[test], probas[:,1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        area_under_curve = auc(fpr,tpr)
        #plt.plot(fpr, tpr, lw=2, color=color, label='ROC fold %d (area = %0.2f)' % (i, area_under_curve))
        i=i+1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
    print "Results for " + classifier + " (opens in new window):"
    #finalize mean roc curve
    mean_tpr /= crossValidator.get_n_splits(features, truth)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
    label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for '+classifier + '\n Oncocytoma vs. Clear Cell RCC\n Using Peak ROI Enhancement and 1st-Order Texture Features')
    plt.legend(loc="lower right")
    plt.subplots_adjust(top=0.85)
    plt.show()

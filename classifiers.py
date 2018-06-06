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
from itertools import cycle

from scipy import interp, stats
import SimpleITK as sitk, numpy, scipy.io, scipy.ndimage, pylab, os, re, csv, math
import matplotlib.pyplot as plt, pandas as pd

class imgClassifier:

	def __init__(self):
		return
	def classify_and_plot(self,features, truth):
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
			    temp = classifiers[classifier].fit(features[train], truth[train])
			    probs = temp.predict_proba(features[test])
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
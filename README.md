# UCLA CS168 - Computational Methods for Medical Imaging
# Connor Kenny (304437322) and Hansen Qiu (004490085)

### Dependencies
We wrote our project in python, so be sure to have python 2.7 installed. 
On top of python, we used numerous libraries to perform our classification. 
They are listed below with a brief explanation of what they were used for.

- scikit-learn: Utilized numerous classifiers and analysis metrics
- scikit-image: Performed filtering on our raw images
- SimpleITK: Read the images provided by Heidi Coy
- pandas: Read the .csv file into a dataframe
- scipy: Utilized statistics functions for features and classifier analysis
- matplotlib: Created graphs to compare different classifiers
- numpy: numpy arrays held our features to speed up training
- pylab: Viewed images pre and post filtering


### File Description
We have our classification script split into numerous files.
We also have a .csv file with the normalization values used to create a relative peak ROI intensity from each image.
We have not included our data because it is too big to submit on ccle.
The data was provided by Heidi Coy.

- onc_rcc.py: Main script to run that exctracts features and performs classification
- myconstants.py: Holds constants that are changed to run on different machines
- classifiers.py: Holds all the classifiers and plots analysis graphs
- filters.py: Performs filtering and ROI extracting
- RCC_normalization_values.csv: Normalization values provided by Ms. Coy


### How to Use
Since we modularized our approach, running our classification is quite simple.
First, be sure to acquire the data and put the location of the data directory in "data_dir" in myconstants.py.
Second, make sure to have all dependencies downloaded and updated.
Third, to save features create an empty feature folder and put its location in "features_path" in myconstants.py.
Fourth, put the RCC_normalization_values.csv location in "csv_file_location" in myconstants.py.
Finally, run onc_rcc.py using python 2.7 to see the features extracted, as well as models trained and analyzed.
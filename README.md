# Rinse_Over_Run
Rinse Over Run:
forecasting the global energy consumption of a building

https://www.drivendata.org/competitions/56/predict-cleaning-time-series/

Generating submission:
Copy the contest data to /data directory.

Ensure the requirements in requirements.txt are satisfied

I use the spyder ide but this will work in a python enviroment 
navigate to Rinse__Over_Run base folder
Run following python script:
Rinse_Over_Run.py
The recreated final submission is stored in data/test_predictions.csv.


Details about generating predictions:
when running in spyder:
    Python 3.7.1 (default, Dec 10 2018, 22:54:23) [MSC v.1915 64 bit (AMD64)]
    Type "copyright", "credits" or "license" for more information.

    IPython 7.2.0 -- An enhanced Interactive Python.

    runfile('C:/Users/YourNameHere/Documents/GitHub/Rinse_Over_Rum/Rinse_Over_Run.py', wdir='C:/Users/YourNameHere/Documents/GitHub/Rinse_Over_Run')

Running Rinse_Over_Run.py then prints the following progress updates:

Libraries imported and functions defined    -confirming required libraries have been imported and functions have been defined correctly 
train_values.csv uploaded                   -train_values.csv found and read
recipe_metadata.csv uploaded                -recipe_metadata.csv found and read
train_labels.csv uploaded                   -train_labelss.csv found and read
test_values.csv uploaded                    -test_values.csv found and read
Launching                                   -other variables defined and the model workflow is being launched
Training Data prep successful               -Prepped the training data for the Flows 
Test Data prep successful                   -Prepped the test data for the Flows
Object Flow1 successful                     -Object Flow1 executed succesfully
Clustering Flow2 successful                 -Cluster Flow2 executed succesfully
Rinse Over Run successful: prediction can be found at data/test/test_predictions.csv 

It takes about 10 minutes to for the script to execute  and recreate the solution, the most time consuming part is uploading the csv's as they are quite large.


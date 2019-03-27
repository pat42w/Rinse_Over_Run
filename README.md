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

It takes about 2hrs to recreate the solution, the most time consuming part is 20seeds computation.

Solution writeup
The brief description of the solution is in the reports directory.
Code structure
 |_ source
 |_ reports: documents made during the challenge according to CRISP-DM methodology
 |_ tests: folder with tests for the library
 |_ data: folder with light data from teh challenge
 |_ rules: the official rules of the challenge
Final solution
You can find a script to recreate the final solution on scripts/final_solution folder

Challenge workflow
Start of the challenge
Add dates to the calendar

Download rules of the challenge

Create a repository for the code using cookiecutter

Create a Google keep label for tasks and ideas of the challenge

Download the challenge data

Create a conda environment for the challenge

conda create -n repo_name pytest rope pylint tqdm numpy pandas sklearn source activate repo_name conda install -c conda-forge jupyter_contrib_nbextensions

Work on the challenge

Use TDD methodology whenever possible, this will save time because errors won't be propagated along the challenge.

Have an aprentice aptitude, collaborate on the forum, I have a lot to learn from Kaggle.

Prepare a report with a summary of the aproach to the challenge

End of the challenge
Prepare a report with a summary of the aproach to the challenge
Download the Google keep tasks to the repository in pdf format
Delete the tasks on google keep and the label
Delete unnecessary data
Update the environment yml
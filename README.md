# Rinse_Over_Run

## Rinse Over Run:
forecasting the global energy consumption of a building
<https://www.drivendata.org/competitions/56/predict-cleaning-time-series/>
## Authors:
The BI Sharpes: Pat Walsh & David Belton
[Pat's Github](https://github.com/pat42w)
March 2019

## Generating submission:
 1. Copy the contest data to `/data` directory.
This data should be available at:
<https://www.drivendata.org/competitions/56/predict-cleaning-time-series/page/125/#datasets>

 2. Ensure the requirements in `requirements.txt` are satisfied.
 >>I use the Spyder ide but this will work in any python environment with the requirements met.

 3. Getting Predictions:
	 - navigate to `Rinse_Over_Run` base folder
	- Run following python script: `Rinse_Over_Run.py`
 4. The recreated final submission is stored in `data/test_predictions.csv`

## Details about generating predictions:
When running in you should see:
>(base) C:\Users`\YourNameHere\`Documents\GitHub\Rinse_Over_Run>`python Rinse_Over_Run.py`
train_values.csv uploaded
recipe_metadata.csv uploaded
train_labels.csv uploaded
test_values.csv uploaded
Launching
Training Data prep successful
Test Data prep successful
Object Flow1 successful
Clustering Flow2 successful
Rinse Over Run successful: prediction can be found at data/test_predictions.csv
...It's been done.

Running Rinse_Over_Run.py then prints the following progress updates:

Libraries imported and functions defined 
 > - confirming required libraries have been imported and functions have been defined correctly

`train_values.csv` uploaded 
> - train_values.csv found and read

`recipe_metadata.csv` uploaded 
> - recipe_metadata.csv found and read

`train_labels.csv` uploaded 
>- train_labels.csv found and read

`test_values.csv` uploaded 
> - test_values.csv found and read

Launching 
>- other variables defined and the model workflow is being launched

Training Data prep successful 
> - Prepped the training data for the Flows

Test Data prep successful 
> - Prepped the test data for the Flows

Object Flow1 successful 
> - Object Flow1 executed succesfully

Clustering Flow2 successful 
>- Cluster Flow2 executed succesfully

Rinse Over Run successful: prediction can be found at `data/test_predictions.csv`
...It's been done.
 > Verifying completion and referencing a classic Simpsons quote.
 > 
It takes about **10 minutes** to for the script to execute and recreate the solution, the most time consuming part is uploading the csv's as they are quite large at 2.13GB.



# stringmatch

This package matches each string in a list of strings to its closest match in a set of possible candidate strings. One use case is matching an abbreaviation for a column name to its full length descriptive column name.

## Installation
This package is installed by downloading the package from github and using pip.

To download the python package (assuming SSH is used instead of HTTPS), run the following command:

`git clone git@github.com:FundingCircle/funding-circle-risk-tools.git`

It is recommended to install in editable mode so that local modifications to the code are immediately effective (very useful for development). To install, run the following command:

`pip install --editable ~/path/to/funding-circle-risk-tools`

## Usage
For a complete example using this package please refer to [a relative link](notebooks/Predicting%20Column%20Names.ipynb)

Import the package along with any other packages that may be useful for your analyses.
```python
import pandas as pd
import stringmatch.model as sm
```

Load the data
```python
df_all_versions_cols = pd.read_excel('../resources/all_versions_of_columns.xlsx')
```

Perform wrangling and transformations
```python
df_athena_gpo = pd.DataFrame(df_all_versions_cols['Athena GPO'].dropna())
df_calculated_cols = pd.DataFrame(df_all_versions_cols[['Calculated Non-GPO', 'Calculated GPO']])
```

Make Predictions
```python
matches = sm.find_best_match_for_list_of_strings(df_all_versions_cols['Athena Non-GPO'].dropna(),
                              df_all_versions_cols['Calculated Non-GPO'],
                              algorithm='damerau-levenshtein',
                              weights=(8, 1, 10, 9))
```

## Future
Example showing training

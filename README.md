# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
## Develped by : Manoj M
## Reg no : 212221240027

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.

### ALGORITHM:
```
1.Load the dataset.
2.Calculate the mean and variance of FinalGrade.
3.Normalize the FinalGrade column using the mean and variance.
4.Pre-allocate an array to store the autocorrelation values.
5.Calculate the autocorrelation for each lag from 1 to 35.
6.Plot the autocorrelation values against their corresponding lags
```
### PROGRAM:


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the airline complaints dataset
df = pd.read_csv('/content/student_performance.csv')

# Calculate the mean and variance
mean_X = np.mean(df['FinalGrade'])
var_X = np.var(df['FinalGrade'])

# Normalize the data
X_normalized = (df['FinalGrade'] - mean_X) / np.sqrt(var_X)

# Pre-allocate autocorrelation table
acf_table = np.zeros((35, 1))

# Calculate autocorrelation for each lag
for k in range(1, 36):
    autocorrelation_k = np.sum(X_normalized[:-k] * X_normalized[k:]) / (len(X_normalized) - k)
    acf_table[k-1] = autocorrelation_k

# Display the ACF graph
plt.plot(acf_table)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('ACF of Student FinalGrade (First 35 Lags)')
plt.show()
```
### OUTPUT:
![download](https://github.com/user-attachments/assets/10366aef-158c-4f94-86ee-52c2a21d6d8e)




### RESULT:
        the program have successfully implemented the auto correlation function in python.

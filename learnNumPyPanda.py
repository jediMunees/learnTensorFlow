import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

oneDimensionalArray = np.array([1 , 2, 3, 4 , 5])

print(oneDimensionalArray)
twoDimeArray = np.array([[6,5], [11, 7], [4, 8]])

print(twoDimeArray)
onesArray = np.ones((2,3), np.int8)

print(onesArray)
zerosArray = np.zeros((2,2))

sequenceOfNumbers = np.arange(5, 10)
print(sequenceOfNumbers)

randomnum = np.random.randint(low = 10, high = 20, size=(6))
print(randomnum)

randomFloat = np.random.random([6])
print(randomFloat)

addedRandom = randomnum + randomFloat
print(addedRandom)

print("#######################\n\n")

features = np.arange(6, 21)
label = 3 * features + 4
#print(label)

sizeOfLabel = label.size
noiseRandomElements = np.random.uniform(low = -2, high = 2, size=[sizeOfLabel])
print(noiseRandomElements)

label = label + noiseRandomElements
print(label)

print("\n\n ####### pandas try ######\n ")
my_data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
my_column_names = ['temperature', 'activity']
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)
print(my_dataframe)

my_dataframe['adjusted'] = my_dataframe['activity'] + 4
print(my_dataframe)

print("\nRow \t#0, \t#1, \t#2 :")
print(my_dataframe.head(3), '\n')

print(" Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print(" Row #1, #2, #3 :")
print(my_dataframe[1:4], '\n')

print("Column temperature: ")
print(my_dataframe['temperature'])

############################################################################################
print("####### task ########## \n")

input_values = np.random.randint(1,100,(3,4))
# print(input_values, '\n')

input_columns = ["Eleanor", "Chidi", "Tahani", "Jason"]

input_matrix = pd.DataFrame(data=input_values, columns=input_columns)

# the entire DataFrame
print(input_matrix, '\n')

# the value in the cell of row #1 of the Eleanor column
print("Second row of the Eleanor column:", input_matrix.loc[1, 'Eleanor'], '\n')

#Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason
input_matrix['Janet'] = input_matrix['Tahani'] + input_matrix['Jason']
print(input_matrix, '\n')

############################################################################################
input_ref_matrix = input_matrix
print("starting value of input_matrx: %d\n" % input_matrix['Eleanor'][0])
print("starting value of input_ref_matrix: %d\n" % input_ref_matrix['Eleanor'][0])

input_matrix.at[0, 'Eleanor'] = input_matrix.at[0, 'Eleanor'] + 1
print("updated value of input_matrx: %d\n" % input_matrix['Eleanor'][0])
print("updated value of input_ref_matrix: %d\n" % input_ref_matrix['Eleanor'][0])

print("##### Experiment true copy:\n")
input_copy_matrix = input_matrix.copy()
print("starting value of input_matrx: %d\n" % input_matrix['Eleanor'][0])
print("starting value of input_copy_matrix: %d\n" % input_copy_matrix['Eleanor'][0])

input_matrix.at[0, 'Eleanor'] = input_matrix.at[0, 'Eleanor'] + 1
print("updated value of input_matrx: %d\n" % input_matrix['Eleanor'][0])
print("updated value of input_copy_matrix: %d\n" % input_copy_matrix['Eleanor'][0])

############################################################################################


#### Linear Regression with Synthetic Data.ipynb:






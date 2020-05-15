import operator
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt 


class ForceWithYou:
    def __init__(self, file_path):
        # Learn Linear Regression with real data
        # The following lines adjust the granularity of reporting.
        pd.options.display.max_rows = 10 
        pd.options.display.float_format = "{:.1f}".format

        # Import the dataset
        self.training_df = pd.read_csv(filepath_or_buffer=file_path)
        self.my_model = None
        self.my_feature = ""
        self.my_label = ""

        # Scale the median_house_value.
        self.training_df["median_house_value"] /= 1000.0

        print("First few values of training set:\n", self.training_df.head(), '\n')

        # Get statistics of the dataset from pandas DataFrame
        print("Statistics of training set:\n", self.training_df.describe(), '\n')

        print("Max values of training set:\n", self.training_df.max(), '\n')


    def perform_opearator_fn(self, operation):
        return {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
        }[operation]
    

    def create_new_feature(self, feature1, feature2, operation, new_feature_name):
        # Create new feature using feature1 and feature2.
        self.training_df[new_feature_name] = self.perform_opearator_fn(operation)(self.training_df[feature1], self.training_df[feature2])
        # print(self.training_df[new_feature_name].head())


    def gen_corelated_feature(self):
        print("Correlation matrix:\n", self.training_df.corr(), '\n')

    #@title Define the function that build the model
    def build_model(self, my_learning_rate):
        """ Create and complile simple linear regression model. """

        # Most simple tf.keras models are sequential.
        self.my_model = tf.keras.models.Sequential()

        # Describe the topography of the model
        # The topography of a simple linear regression model is a single node in a single layer
        self.my_model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

        # Compile the model topography into code that TensorFlow efficiently execute
        # Configure the model to minimize the model's mean squared error.
        self.my_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate), 
                                loss="mean_squared_error",
                                metrics=[tf.keras.metrics.RootMeanSquaredError()])


    def train_model(self, epochs, batch_size):
        """ Train the model by feeding it data """

        # Feed the feature and label to model
        # The model will train for the specified number of epochs.
        history = self.my_model.fit(x=self.training_df[self.my_feature], y=self.training_df[self.my_label],
                            batch_size=batch_size,
                            epochs=epochs)

        # Gather the trained model's weight and bias
        trained_weight = self.my_model.get_weights()[0]
        trained_bias = self.my_model.get_weights()[1]

        # The list of epochs is stored separately from the rest of history
        # This is vector of iterations
        epochs = history.epoch

        # Isolate the error of each epoch. 
        hist = pd.DataFrame(history.history)

        # To track the progression of training, we are going to take a snapshot
        #   of the model's root mean squared error at each epoch.
        #   Basically get the cost J(theta) at each epoch.
        rmse = hist["root_mean_squared_error"]

        return trained_weight, trained_bias, epochs, rmse


    # @title Define the plotting functions
    def plot_the_model(self, trained_weight, trained_bias):
        """ Plot the trained model against 200 random training examples """

        # Label the axes.
        plt.xlabel(self.my_feature)
        plt.ylabel(self.my_label)

        # Create a scatter plot from 200 random points of the dataset.
        random_examples = self.training_df.sample(n=200)
        plt.scatter(random_examples[self.my_feature], random_examples[self.my_label])

        # create a redline represents the model. The red line starts 
        #   at co-ordinates (x0,y0) and ends at co-ordinates (x1,y1).
        x0 = 0
        y0 = trained_bias

        x1 = 10000
        y1 = trained_bias + (trained_weight * x1)
        plt.plot([x0, x1], [y0, y1], c='r')

        # Render the scatter plot and the red line.
        plt.show()


    def plot_the_loss_curve(self, epochs, rmse):
        """ Plot the curve of loss - J(theta) vs epoch - vector of iterations."""

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")

        plt.plot(epochs, rmse, label="loss")
        plt.legend()
        plt.ylim([rmse.min()*0.97, rmse.max()])
        plt.show()


    def predict_house_values(self, n):
        """ Predict house values based on a feature. """
        batch = self.training_df[self.my_feature][10000:10000 + n]
        # If n = 10, then batch returns the following for the given .cvs file.
        #Name: total_rooms, dtype: float64 
        #Value:  10000   1960.0
        #10001   3400.0
        #10002   3677.0
        #10003   2202.0
        #10004   2403.0
        #10005   5652.0
        #10006   3318.0
        #10007   2552.0
        #10008   1364.0
        #10009   3468.0
        #
        predicted_values = self.my_model.predict_on_batch(x=batch)
        print("feature   label          predicted")
        print("  value   value          value")
        print("          in thousand$   in thousand$")
        print("--------------------------------------")
        for i in range(n):
            print("%5.0f %6.0f %15.0f" % (self.training_df[self.my_feature][10000+i],
                                        self.training_df[self.my_label][10000+i],
                                        predicted_values[i][0]) )


    def machine_learning(self, learning_rate, epochs, batch_size, my_feature, my_label):
        # Ex: Feature and label
        # my_feature = "total_rooms" # the total number of rooms on a specific city block.
        # my_label = "median_house_value" # the median value of a house on a specific city block.
        # That is, you're going to create a model that predicts house value based 
        # solely on total_rooms.  

        # Discard any pre-existing version of the model.
        self.my_model = None
        self.my_feature = my_feature
        self.my_label = my_label

        # Invoke the functions.
        self.build_model(learning_rate)

        weight, bias, epochs, rmse = self.train_model(epochs, batch_size)

        print("The learned weight for your model is %.4f" % weight)
        print("The learned bias for your model is %.4f" % bias)

        self.plot_the_model(weight, bias)
        self.plot_the_loss_curve(epochs, rmse)

        # Invoke the house prediction on 10 examples
        self.predict_house_values(10)


jedi = ForceWithYou('california_housing_train.csv')

# Try with "total_rooms" feature
# jedi.executeMachineLearning(0.01, 30, 30, "total_rooms", "median_house_value")

# Try with "population" feature
# jedi.executeMachineLearning(0.05, 18, 3, "population", "median_house_value")

# Try with new "rooms_per_person" feature
jedi.create_new_feature("total_rooms", "population", "/", "rooms_per_person")
#jedi.machine_learning(0.06, 24, 30, "rooms_per_person", "median_house_value")

# Generate correlational matrix to get the understanding of feature selection w.r.t median_house_value.
# The correlation matrix shows nine potential features (including a synthetic feature) and one label (median_house_value). 
# A strong negative correlation or strong positive correlation with the label suggests a potentially good feature.
# A correlation matrix indicates how each attribute's raw values relate to the other attributes' raw values. Correlation values have the following meanings:
#   1.0: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
#   -1.0: perfect negative correlation; that is, when one attribute rises, the other attribute falls.
#   0.0: no correlation; the two column's are not linearly related.
jedi.gen_corelated_feature()

# # The `median_income` correlates 0.7 with the label 
# (median_house_value), so median_income` might be a 
# good feature. The other seven potential features
# all have a correlation relatively close to 0. 
#              longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value  rooms_per_person
#median_income -0.0      -0.1                -0.1          0.2            -0.0        -0.0         0.0            1.0                 0.7               0.2
jedi.machine_learning(0.06, 48, 60, "median_income", "median_house_value")




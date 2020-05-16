import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

class ForceWithYou:

    def __init__(self, train_file_path, test_file_path):

        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.1f}".format

        self.train_df = pd.read_csv(filepath_or_buffer=train_file_path)
        self.test_df = pd.read_csv(filepath_or_buffer=test_file_path)
        self.model = None
        self.feature = None
        self.label = None

        scale_factor = 1000.0
        # scale train set's label
        self.train_df["median_house_value"] /= scale_factor
        # scale the test set's label
        self.test_df["median_house_value"] /= scale_factor


    def build_model(self, my_learning_rate):
        """ Create a simple linear regression model """

        # Create a simple sequential model 
        self.model = tf.keras.models.Sequential()

        # Add one linear layer to the model to yield  a simple linear regressor.
        self.model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

        # Compile the model topography into code that tensorFlow can efficiently 
        #   execute. Configure training to minimize the model's mean squared error.
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                            loss="mean_squared_error",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])


    def train_model(self, my_epochs, my_batch_size=None, my_validation_split=0.1, train_df=None):
        """ Feed a dataset into the model in order to train it. """
        if train_df is None:
           train_df = self.train_df

        history = self.model.fit(x=train_df[self.feature], y=train_df[self.label],
                                batch_size=my_batch_size, epochs=my_epochs,
                                validation_split=my_validation_split)

        # Gather the model's trained weight and bias
        trained_weight = self.model.get_weights()[0]
        trained_bias = self.model.get_weights()[1]

        # the list of epochs stored separately from the rest of history
        epochs = history.epoch

        # Get the root mean squared error for each epoch.
        hist = pd.DataFrame(history.history)
        rmse = hist["root_mean_squared_error"]
        return epochs, rmse, history.history


    def plot_the_loss_curve(self, epochs, mae_training, mae_validation):
        """ Plot the curve of loss vs epoch. """
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")
        plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
        plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
        plt.legend()

        # We are not going to plot the fist epoch, since the loss on the fist
        #   is often substantially greater than the loss for other epochs.
        merged_mae_lists = mae_training[1:] + mae_validation[1:]
        highest_loss = max(merged_mae_lists)
        lowest_loss = min(merged_mae_lists)
        delta = highest_loss - lowest_loss
        print(delta)

        top_of_y_axis = highest_loss + (delta * 0.05)
        bottom_of_y_axis = lowest_loss - (delta * 0.05)

        plt.ylim(bottom_of_y_axis, top_of_y_axis)
        plt.show()


    def machine_learning(self, learning_rate, epochs, batch_size, validation_split,
                            my_feature, my_label):
        print("Execute machine learning...\n")
        self.model = None
        self.feature = my_feature
        self.label = my_label

        # Invoke the functions to build and train the model
        self.build_model(learning_rate)

        shuffled_train_df = self.train_df.reindex(np.random.permutation(self.train_df.index))

        epochs, rmse, history = self.train_model(epochs, batch_size, validation_split, shuffled_train_df)

        self.plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                                    history["val_root_mean_squared_error"])

        # Use the test dataset and evaluate the model's performance.
        results = self.model.evaluate(self.test_df[self.feature], self.test_df[self.label], batch_size=batch_size)


##### Main function ######
jedi = ForceWithYou("california_housing_train.csv", "california_housing_test.csv")

jedi.machine_learning(learning_rate=0.08, epochs=30, batch_size=100, validation_split=0.2,
                        my_feature="median_income", my_label="median_house_value")
                    

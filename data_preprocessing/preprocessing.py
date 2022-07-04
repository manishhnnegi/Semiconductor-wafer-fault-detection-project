import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class Preprocessor:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object


    def remove_columns(self, data, columns):
        """
                        Method Name: remove_columns
                        Description: This method removes the given columns from a pandas dataframe.
                        Output: A pandas DataFrame after removing the specified columns.
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        try:
            useful_data = data.drop(labels=columns, axis=1)
            self.logger_object.log(self.file_object, 'removed column successfully')
            return useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()




    def separate_label_feature(self, data, label_column_name):
        """
                                Method Name: separate_label_feature
                                Description: This method separates the features and a Label Coulmns.
                                Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            X = data.drop(labels = label_column_name, axis=1)
            y = data[label_column_name]
            return X,y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self, data):
        """
                                               Method Name: is_null_present
                                               Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                               Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                               On Failure: Raise Exception

                                               Written By: iNeuron Intelligence
                                               Version: 1.0
                                               Revisions: None

                                """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        try:
            null_present = False
            null_counts = data.isna().sum()
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if (null_present):  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.isna().sum().index
                dataframe_with_null['missing values '] = data.isna().sum().values
                # dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()


    def impute_missing_values(self, data):
        """
                                                Method Name: impute_missing_values
                                                Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                                Output: A Dataframe which has all the missing values imputed.
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        try:
            kn = KNNImputer(n_neighbors=3, weights='uniform')
            new_array = kn.fit_transform(data)
            new_data = pd.DataFrame(new_array, columns=data.columns)
            return new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self, data):
        """
                                                        Method Name: get_columns_with_zero_std_deviation
                                                        Description: This method finds out the columns which have a standard deviation of zero.
                                                        Output: List of the columns with standard deviation of zero
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        col_to_drop = []
        try:
            for i in data.columns:
                if data[i].std() == 0:
                    col_to_drop.append(i)
            return col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

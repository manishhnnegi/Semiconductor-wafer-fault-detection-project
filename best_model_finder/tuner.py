from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder:

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object



    def get_best_params_for_random_forest(self, train_x, train_y):
        """
                                        Method Name: get_best_params_for_random_forest
                                        Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            param_grid = {"n_estimators": [10, 30], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 3, 1), "max_features": ['auto', 'log2']}
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, verbose=3)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, max_features=max_features)
            # training the mew model
            clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_xgboost(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_xgboost
                                                Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            param_grid_xgboost = {

                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5],
                'n_estimators': [10, 20]

            }
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic'), param_grid=param_grid_xgboost,
                                verbose=3, cv=5)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            return xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
                                                        Method Name: get_best_model
                                                        Description: Find out the Model which has the best AUC score.
                                                        Output: The best model name and the model object
                                                        On Failure: Raise Exception

                                                        Written By: iNeuron Intelligence
                                                        Version: 1.0
                                                        Revisions: None

                                                """

        # create best model for XGBoost
        try:
            # create best model for XGBoost
            #self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)

            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost)  # AUC for XGBoost
                # Log AUC

            # create best model for Random Forest
            random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            prediction_random_forest = random_forest.predict(test_x)  # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                random_forest_score = accuracy_score(test_y, prediction_random_forest)

            else:
                random_forest_score = roc_auc_score(test_y, prediction_random_forest)  # AUC for Random Forest

            # comparing the two models
            if (random_forest_score < xgboost_score):
                return 'XGBoost', xgboost
            else:
                return 'RandomForest', random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()



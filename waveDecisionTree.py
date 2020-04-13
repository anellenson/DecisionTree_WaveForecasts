from sklearn.ensemble import BaggingRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import io
import os
import random
import sys
import time
import Error
import scipy.io as sio
import numpy as np
import pandas as pd
import pickle

class waveDecisionTree():

    def load_data(self, season):
        self.season = season
        if 'summer' not in season and 'winter' not in season:
            print('Season options are winter or summer')

        else:
            all_data = sio.loadmat(season + '_allfeatures_allyears.mat')

        #The features are an array where the first three columns are hourly data points Hs, MWD, Tm output by WW3 at NDBC Buoy 46050.
        #and the second two columns are wind magnitude and wind direction output by GFS and input into WW3.

            train_features = all_data['train_features'][:]
            print("The input features are an array of shape " + str(train_features.shape)) #Note the shape MxN, where M = number of hours and N = number of features

            #The train target are the hourly corrections (hourly deviations) between the WW3 output
            #and the observed wave height at NDBC Buoy 46050
            train_target = all_data['train_target'][:]
            train_target = train_target.squeeze()

            train_obs = train_features[:,0] + train_target

            self.train_features = train_features
            self.train_obs = train_obs
            self.train_target = train_target
            self.all_data = all_data

    def train_tree(self, no_runs, depths, treenos):
        '''
        no_runs = number of times to run
        depths = list of integers which are the tree depths to try
        treenos = list of integers which are the forest size (number of trees) to try
        '''


        start = time.time()

        #All the possible train feature names are Hs_ww3, mwd_ww3, Tm_ww3, wndmag, and wnddir
        #Load the train features and choose which features you want as a list in the "train feature names"
        trainfeaturenames = ['Hs_ww3','mwd_ww3','Tm_ww3','wndmag','wnddir']



        test_features = self.all_data['test_features'][:]
        test_target = self.all_data['test_target'][:]
        test_target = test_target.squeeze()

        #possibility here to just choose the input features you're interested in
        test_features = test_features[:,0:len(trainfeaturenames)]

        ww3_hs_test = test_features[:,0] #The first column of the feature set is Hs.
        obs_hs_test = ww3_hs_test + test_target #Recover the observations by adding back the corrections

        #Number of folds for the cross-fold validation
        no_folds = 5

        #Set up the inputs for the GridSearchCV function
        #depths = [5,10,15,20,30,50,70]
        #treenos = [10,30,50,100,150,200,250,275,300,350,400,500]
        opt_params = {'base_estimator__max_depth':depths,'n_estimators':treenos}

        #Set up final data frames which will need to be saved
        #Importance values for each feature
        nandata = np.ones((no_runs,self.train_features.shape[1]))*np.nan
        feature_importance_matrix = pd.DataFrame(nandata, columns = trainfeaturenames)

        #Optimal parameters for each run
        nandata = np.ones((no_runs,2))*np.nan
        forest_parameter_matrix = pd.DataFrame(nandata,columns = ['depth','number'])

        #Training error
        nandata = np.ones((no_runs,5))*np.nan
        training_error_df = pd.DataFrame(nandata,columns = ['RMSE','bias','PE','SI','Corr_Coeff'])

        #Final corrections for each run. The mean of this matrix will be taken as the final corrections.
        DT_var_matrix = np.ones((no_runs,len(ww3_hs_test)))*np.nan

        #All ready to train!
        print("Training now")
        for runno in range(no_runs):
            t0_run = time.time()

            #Use gridsearch CV with "Bagging Regressor" - the ensemble technique and the "Decision Tree Regressor" - the base learner - nested inside.
            DT_cv = GridSearchCV(BaggingRegressor(base_estimator = tree.DecisionTreeRegressor()), opt_params, cv = 5, scoring = 'neg_mean_absolute_error', iid = False)

            #GridSearchCV has returned an object where the best parameters are automatically selected in DT_cv
            #Now we can fit the tree to the training data
            DT_cv.fit(self.train_features,self.train_target)


            #Save out forest parameters
            forest_parameter_matrix.iloc[runno]['depth'] = DT_cv.best_estimator_.base_estimator.max_depth
            forest_parameter_matrix.iloc[runno]['number'] = DT_cv.best_estimator_.n_estimators

            bestDT = DT_cv.best_estimator_

            #Record the importances
            importances = [x.feature_importances_ for x in bestDT.estimators_]
            importance_df = pd.DataFrame(importances, columns = trainfeaturenames)
            feature_importance_matrix.iloc[runno] = importance_df.describe().loc['mean'].values


            DT_corrections = DT_cv.predict(test_features)

            #make the ultimate prediction using decision tree corrections
            DT_var_matrix[runno,:] = DT_corrections
            t1_run = time.time()
            timedifference = t1_run - t0_run

            print("Run " + str(runno) + "took {0:0.2f}".format(timedifference) + "seconds featuring " + " ".join(trainfeaturenames))

        end = time.time()

        totaltimediff = end-start
        print("Total time is {0:0.2f}".format(totaltimediff))

        #Save the best bagged regression tree to explore later.
        best_DT = DT_cv.best_estimator_
        DT_pickle = open('./ExampleDT.pickle', 'wb')
        pickle.dump(best_DT, DT_pickle)
        DT_pickle.close()

        #Calculate the mean corrections from the runs for the final corrections.
        meancorrections = np.mean(DT_var_matrix)
        dt_hs = ww3_hs_test + meancorrections

        self.feature_importance_matrix = feature_importance_matrix
        self.forest_parameter_matrix = forest_parameter_matrix
        self.test_target = test_target

        return obs_hs_test, ww3_hs_test, dt_hs

class exploreTree():

    def __init__(self, bestDT, inputdata, target, treeno, trainfeaturenames):
        self.bestDT = bestDT
        self.inputdata = inputdata
        self.target = target
        self.estimator = self.bestDT.estimators_[treeno]
        self.trainfeaturenames = trainfeaturenames


    def top_populated_leaves(self, numleaves, filter_features = False):
        '''
        This function will return information about the top populated partitions.

        Arguments:
        _______________
        inputdata:  what input data do you want to have the tree predict on? Must be the same input features as what the tree was trained on
        numleaves:  the number top populated leaves

        Returns:
        _______________
        leafIDs:        The final node (leaf) numbers applied to each example. Size = [inputdata.shape[0], 1]
        topleafIDs:     The top populated leaf IDs. Size = [numleaves, 1]
        correctionVals: The correction values associated with each of the topleafIDs. Size = [numleaves,1]
        '''

        if filter_features:
            evaluate_features = self.filtered_features

        else:
            evaluate_features = self.inputdata

        #Find the most populated partitions and choose the top populated partitions
        all_leaf_ids = self.estimator.apply(evaluate_features)
        bins = np.unique(all_leaf_ids)
        numexs, leafids = np.histogram(all_leaf_ids, bins)
        numexs_idx = np.argsort(numexs)
        topleaf_ids = leafids[numexs_idx[::-1]] #Node numbers of the top populated leaves
        correction_vals = [np.float(np.round(self.estimator.tree_.value[ll][0][0],3)) for ll in topleaf_ids[:numleaves]] #Correction values for the top populated leaves

        self.correction_vals = correction_vals
        self.topleaf_ids = topleaf_ids
        self.all_leaf_ids = all_leaf_ids
        self.evaluate_features = evaluate_features


        return all_leaf_ids, topleaf_ids, correction_vals

    def filter_feature_vals(self, feature, feature_min, feature_max):
        '''
        This input features the input data so that you're only interetsed in a certain region of features and associated values
        feature:        string. Options are 'Hs', 'Tm', 'MWD', 'wndmag', 'wnddir'
        feature_min:    minimum value of that feature
        feature_max:    maximum value of that feature
        '''

        feature_column = {'Hs': 0, 'Tm': 2, 'MWD': 1, 'wndmag': 3, 'wnddir': 4}
        ff = feature_column[feature]
        feature_idx = np.where((self.inputdata[:, ff] > feature_min) & (self.inputdata[:, ff] < feature_max))[0]
        # Filter those feature nodes
        filtered_features = self.inputdata[feature_idx]

        self.filtered_features = filtered_features

        print("Filtered features")

        return filtered_features

    def return_hs(self, topleafno):

        leaf_members = np.where(self.all_leaf_ids == topleafno)[0]
        correction = np.float(np.round(self.estimator.tree_.value[topleafno][0][0],3)) #What was the correction applied to these members

        ww3_hs = self.evaluate_features[leaf_members,0]
        obs_hs = self.evaluate_features[leaf_members,0] + self.target[leaf_members]
        dt_hs = self.evaluate_features[leaf_members,0] + correction

        return ww3_hs, obs_hs, dt_hs

    def print_rules(self, topleafno):
        #Note that the following code is from scikit learn documentation

        leaf_members = np.where(self.all_leaf_ids == topleafno)[0]
        #Which region of model phase space are these examples from?
        n_nodes = self.estimator.tree_.node_count
        children_left = self.estimator.tree_.children_left
        children_right = self.estimator.tree_.children_right
        feature = self.estimator.tree_.feature
        threshold = self.estimator.tree_.threshold

        node_indicator = self.estimator.decision_path(self.inputdata)


        sample_id = leaf_members[0] #Choose a sample from the leaf
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]


        print('Input Feature Phase Space: ' % sample_id)
        for node_id in node_index:
            if self.all_leaf_ids[sample_id] == node_id:
                continue

            if (self.inputdata[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("%s %s %s)"
                  % (self.trainfeaturenames[feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))

import waveDecisionTree 
import pickle

waveDT = waveDecisionTree.waveDecisionTree()
waveDT.load_data('winter')
depths = [2,5]
treenos = [10, 20]
noruns = 2
waveDT.train_tree(noruns, depths, treenos)
print(waveDT.feature_importance_matrix)
DT_pickle = open('./ExampleDT.pickle','rb')
bestDT = pickle.load(DT_pickle)
DT_pickle.close()
inputdata = waveDT.train_features
target = waveDT.train_target
treeno = 0
numleaves = 5
trainfeaturenames = ['Hs', 'MWD', 'Tm', 'wndmag', 'wnddir']
exploreTree = waveDecisionTree.exploreTree(bestDT, inputdata, target, treeno, trainfeaturenames)
#if you want to filter a feature choose a feature and a value
feature = 'Tm'
feature_min = 6
feature_max = 9

ff = exploreTree.filter_feature_vals(feature, feature_min, feature_max)
leaf_ids, topleaf_ids, correction_vals = exploreTree.top_populated_leaves(numleaves, filter_features = True) #Number of leaves
topleafno = topleaf_ids[0] #This is the leaf number you want to explore, 0 is the most populated leaf number
ww3_hs, obs_hs, dt_hs = exploreTree.return_hs(topleafno)
exploreTree.print_rules(topleafno)


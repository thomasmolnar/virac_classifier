import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import class_weight

try:
    from sklearn.metrics import ConfusionMatrixDisplay
except:
    pass
try:
    import xgboost
    import lightgbm
except:
    pass

def classif_report(y_test, y_pred, classes, plot_name):
    
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred,output_dict=True)
    
    if plot_name:
        cm = confusion_matrix(y_test,y_pred,normalize=True)
        displ = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=classes)
        disp = displ.plot(include_values=True, cmap=plt.cm.Blues, 
                          ax=None, xticks_rotation='horizontal',
                          values_format=None)

        fig = disp.figure_
        ax = disp.ax_
        im = disp.im_
        fig.set_size_inches(15,15)
        font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 20}
        matplotlib.rc('font', **font)
        ax.set_xlabel('Predicted label',family='serif', fontsize=17.5, labelpad=10)
        ax.set_ylabel('True label',family='serif', fontsize=17.5, labelpad=2)
        #ax.tick_params(labelsize=15)
        ticks_font = matplotlib.font_manager.FontProperties(family='serif',
                                                            style='normal', size=20,
                                                            weight='normal', stretch='normal')
        for label in ax.get_xticklabels():
            label.set_fontproperties(ticks_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(ticks_font)
            
        fig.savefig(plot_name + '_cm.png')

    return cm, cr

def plot_imp(imp_dict, plot_name):
    sort_dict = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}
    feat_names = list(sort_dict.keys())
    feat_imp = np.array(list(sort_dict.values()))
    y_pos = np.arange(len(feat_names))
    
    fig = plt.figure( figsize=(10,10))
    plt.barh(np.arange(0, len(feat_imp)), feat_imp, color='grey', align='center')
    plt.yticks(np.arange(0, len(feat_imp)), feat_names, rotation=22.5, family='serif', fontsize=13)
    plt.xlabel('Relative Feature Importance', family='serif', fontsize=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    fig.savefig(plot_name + '_imp.png')
    return plt


class classification(object):
    
    def train_feat_clip(self, df, impute=True, Nsigma=10):
        """
        Outlier clip outside of 10 sigma (computed from percentiles)

        """
        
        for i in self.periodic_features:
            df[i + '_x'] = np.cos(df[i])
            df[i + '_y'] = np.sin(df[i])
            
        df[self.data_cols + self.target_cols] = df[self.data_cols + self.target_cols].replace([np.inf, -np.inf], np.nan)
        
        self.ptransformer = PowerTransformer()
        with np.errstate(invalid='ignore'):
            df[self.log_transform_cols] = self.ptransformer.fit_transform(df[self.log_transform_cols].astype(np.float64)).astype(np.float32)
        
        self.upper_lower_clips = {}
        fltr = [True]*len(df)

        ## Don't clip periodic features
        for i in list(set(self.data_cols) - 
                        set([s + '_x' for s in self.periodic_features]) - 
                          set([s + '_y' for s in self.periodic_features])):
            eff_sigma = .25*np.diff(np.nanpercentile(df[i], [5., 95.]))[0]
            eff_median = np.nanpercentile(df[i], 50.)
            bot, top = eff_median-Nsigma*eff_sigma, eff_median+Nsigma*eff_sigma
            fltr &= ~(df[i]<bot)
            if i not in self.no_upper_features:
                fltr &= ~(df[i]>top)
            self.upper_lower_clips[i] = [bot, top]
            print(i, np.count_nonzero(fltr))
            
        if impute:
            self.imputer = KNNImputer(n_neighbors=5)
#             self.imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10), verbose=2)
            df[self.data_cols] = self.imputer.fit_transform(df[self.data_cols].values)
        else:
            self.imputer = None
            
        print("{}% sources removed from clip.".format(round(len(df)/np.count_nonzero(fltr)-1, 4)*100))
        
        return df[fltr].reset_index(drop=True)
    
    
    def run(self, training_set, plot_name, nthreads, impute=True, xgb=False, cross_val=True):
        
        training_set = self.train_feat_clip(training_set, impute)
        
        print(training_set[['sourceid','var_class']].groupby('var_class').agg('count'))
        
        X_train, y_train = training_set[self.data_cols], training_set[self.target_cols].values.ravel()
        self.sc = RobustScaler()
        X_train = self.sc.fit_transform(X_train)
        
        if xgb:
#             self.model = lightgbm.LGBMClassifier(n_estimators=100, max_depth=8, learning_rate=0.15)
            self.model = xgboost.XGBClassifier(n_estimators=100,
                                               max_depth=8,
                                               learning_rate=0.15,
                                               min_child_weight=3, #5,
                                               gamma=2.,
                                               colsample_bytree=0.9,
                                               subsample=0.8,
                                               tree_method='hist', n_jobs=1)
            fit_params = {'sample_weight':class_weight.compute_sample_weight(class_weight='balanced',y=y_train)}
        
        else:
            self.model = RandomForestClassifier(n_estimators=100, min_samples_split=5, 
                                                min_samples_leaf=5, max_features='sqrt',
                                                max_depth=8,
                                                class_weight='balanced_subsample')
            fit_params = None
        
        if cross_val:
            split = KFold(n_splits=10, shuffle=True, random_state=42)
            cv = cross_validate(self.model, X_train, y_train, cv=split, return_estimator=True,
                                n_jobs=nthreads,fit_params=fit_params)

            class_, prob_, prob_const_ = np.zeros_like(y_train), np.zeros_like(y_train), np.zeros_like(y_train)
            split = KFold(n_splits=10, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(split.split(X_train, y_train)):
                class_[test_index] = cv['estimator'][i].predict(X_train[test_index])
                prob = cv['estimator'][i].predict_proba(X_train[test_index])
                class_dict = dict(zip(cv['estimator'][i].classes_, np.arange(len(cv['estimator'][i].classes_))))
                prob_[test_index] = prob[np.arange(len(test_index)), [class_dict[clss] for clss in class_[test_index]]]
                prob_const_[test_index] = prob[:, class_dict['CONST']]
            training_set['class'], training_set['prob'], training_set['prob_var'] = class_, prob_, 1-prob_const_

            self.cm, self.cr = classif_report(y_train, training_set['class'], 
                                              cv['estimator'][0].classes_, plot_name)
    #         print(self.cr['VAR']['f1-score'])

    #         for ss in list(set(np.unique(training_set['detailed_var_class']))-set(['CONST'])):
    #             print(ss,np.count_nonzero((training_set['class']=='CONST')&
    #                                       (training_set['detailed_var_class']==ss))/
    #                       np.count_nonzero((training_set['detailed_var_class']==ss)))

        self.model.fit(X_train, y_train, **fit_params)
        self.feature_importance = {c : self.model.feature_importances_[j]
                                    for j, c in enumerate(self.data_cols)}
        if plot_name:
            plot_imp(self.feature_importance, plot_name)
            
        return training_set
    
    def predict(self, y):
        
        yinp = y.copy()
        
        for i in self.periodic_features:
            yinp[i + '_x'] = np.cos(yinp[i])
            yinp[i + '_y'] = np.sin(yinp[i])
            
        yinp = yinp[self.data_cols].replace([np.inf, -np.inf], np.nan)
        with np.errstate(invalid='ignore'):
            yinp[self.log_transform_cols] = self.ptransformer.transform(yinp[self.log_transform_cols])
        if self.imputer is not None:
            yinp[self.data_cols] = self.imputer.transform(yinp[self.data_cols].values)
                
        yinp = self.sc.transform(yinp)
        
        y['class'] = self.model.predict(yinp)
        prob = self.model.predict_proba(yinp)
        class_dict = dict(zip(self.model.classes_, np.arange(len(self.model.classes_))))
        y['prob'] = prob[np.arange(len(y['class'])), [class_dict[clss] for clss in y['class']]]
    
        return y
    
    def __init__(self):
        pass

    
class binary_classification(classification):
    
    def __init__(self, training_set, plot_name=None):
        
        self.data_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
                           "ks_eta",#"ks_eta_e",
                           "ks_stetson_i","ks_stetson_j","ks_stetson_k",
                           "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25",
                           "ks_stdev_over_error","ks_mad_over_error",
                           "ks_p100_p0_over_error","ks_p99_p1_over_error",
                           "ks_p95_p5_over_error","ks_p84_p16_over_error","ks_p75_p25_over_error"
                         ]
        
        self.periodic_features = []
        self.no_upper_features = []
        
        self.target_cols = ['var_class']
        
        self.log_transform_cols = self.data_cols
        
        self.training_set=self.run(training_set, plot_name, nthreads=1)
        
        

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
try:
    from sklearn.metrics import ConfusionMatrixDisplay
except:
    pass

def feat_clip(df_inp, data_cols, target_cols, impute=False, qmin=0.5, qmax=99.5):
    """
    Outlier clip outside of 0.5th and 99.5th percentile
    
    """
    df = df_inp[data_cols + target_cols].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if impute:
        for i in data_cols:
            col_mean = df[i].mean()
            df[i].fillna(col, inplace=True)
    
    fltr = [True]*len(df)
    for i in data_cols:
        bot = np.nanpercentile(df[i], qmin)
        top = np.nanpercentile(df[i], qmax)
        fltr &= (df[i]>bot)&(df[i]<top)
    df = df[fltr].reset_index(drop=True)
    
    print("{}% sources removed from clip.".format(round(len(df_inp)-len(df), 4)*100))
    return df

def classif_report(y_test, y_pred, estimator, plot_name):
    
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)
    
    if plot_name:
        cm = confusion_matrix(y_test,y_pred,normalize=True)
        displ = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=estimator.classes_)
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
    
    def run(self, training_set, plot_name):
        
        print(training_set[[list(set(training_set.columns)-set(['class']))[0],
                            'class']].groupby('class').agg('count'))
        
        X_train, self.y_train = training_set[self.data_cols], training_set[self.target_cols].values.ravel()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=1000, min_samples_split=5, 
                                            min_samples_leaf=5, max_features='sqrt',
                                            max_depth=18, class_weight='balanced_subsample')
        
        split = KFold(n_splits=10, shuffle=True, random_state=42)
        cv = cross_validate(self.model, X_train, self.y_train, cv=split, return_estimator=True)
        
        self.ypred = np.zeros_like(self.y_train)
        split = KFold(n_splits=10, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(split.split(X_train, self.y_train)):
            self.ypred[test_index] = cv['estimator'][i].predict(X_train[test_index])
            
        self.cm, self.cr = classif_report(self.y_train, self.ypred, cv['estimator'][0], plot_name)
        
        self.model.fit(X_train, self.y_train)
        self.feature_importance = [{c : self.model.feature_importances_[j]
                                    for j, c in enumerate(self.data_cols)}]
        if plot_name:
            plot_imp(self.feature_importance, plot_name)
    
    def __init__(self):
        pass

    
class binary_classification(classification):
    
    def __init__(self, training_set, plot_name=None):
        
        self.data_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
#                            "ks_eta","ks_stetson_i","ks_stetson_j","ks_stetson_k",
                           "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25"]
        
        self.target_cols = ['class']
        
        ## Added mean imputation - may need to be rethink for sources
        ## with very limited data entries
        training_set = feat_clip(training_set, self.data_cols, self.target_cols, impute=False)
        
        self.run(training_set, plot_name)
        
        
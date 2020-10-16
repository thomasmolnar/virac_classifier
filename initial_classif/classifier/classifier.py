import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.ensembles import RandomForestClassifier

def feat_clip(df_inp, data_cols, qmin=0.1, qmax=99.9):
    """
    clip outside of 1rst and 99th percentile
    
    """
    df = df_inp[data_cols].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df_run = df.copy()
    for i in data_cols:
        values = np.array(df['{}'.format(i)])
        bot = np.percentile(values, qmin)
        top = np.percentile(values, qmax)
        
        df = df.drop(df[np.array(df['{}'.format(i)])<bot].index)
        df = df.drop(df[np.array(df['{}'.format(i)])>top].index)
    
    print("{} sources removed from clip.".format(len(df_inp)-len(df)))
    print("{} sources left".format(len(df)))
    return df


def select_df(df, _class, inf=False):
    df_use = df.copy()
    if inf:
        df_use.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_use.dropna(axis=0, how='any', inplace=True)

    df_use['class'] = pd.Series(len(df_use)*['{}'.format(_class)])
    print("{0} {1} train class".format(len(df_use), _class))

    return df_use

def classif_report(y_test, y_pred, estimator, plot_name):
    
    cm = metric.confusion_matrix(y_test,y_pred)
    cr = metric.classification_report(y_test,y_pred)
    
    if plot_name:
        cm = metric.confusion_matrix(y_test,y_pred,normalize=True)
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
        
        X_train, y_train = training_set[self.data_cols], training_set[self.target_cols]
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=1000, min_samples_split=5, 
                                            min_samples_leaf=5, max_features='sqrt',
                                            max_depth=18, class_weight='balanced_subsample')
        
        split = KFold(n_splits=10, shuffle=True, random_state=42)
        cv = cross_validate(self.model, X_train, y_train, cv=split, return_estimators=True)
        
        ypred = np.zeros_like(y_train)
        for i,train_index, test_index in enumerate(split(X_train, y_train)):
            ypred[test_index] = cv['estimator'][i].pred(X_train[test_index])

        self.cm, self.cr = classif_report(y, y_pred, cv['estimator'][0], plot_name)
        
        self.model.fit(X_train, y_train.values.ravel())
        self.feature_importance = [{c : self.model.feature_importances_[j]
                                    for j, c in enumerate(self.data_cols)}]
        if plot_name:
            plot_imp(self.feature_importance, plot_name)
    
    def __init__(self):
        pass

class variable_classification(classification):
    
    def __init__(self, data, plot_name=None):
        
        self.data_cols = XXX
        self.target_cols = ['class']
        
        ## Might be better to impute missing data?
        train_const = select_df(data, 'CONST', inf=True).dropna()
        train_all_var = select_df(data, 'VAR').dropna()
        
        training_set = feat_clip(pd.concat([train_const, train_all_var], axis=0), self.data_cols)
        
        run(training_set)
        
    
class binary_classification(classification):
    
    def __init__(self, data, plot_name=None):
        
        self.data_cols = ['skewness','kurtosis', 'stetson_i', 'eta',
                          'mags_stdev', 'mags_mad',
                          'mags_q100mq0', 'mags_q99mq1',
                          'mags_q95mq5','mags_q90mq10', 'mags_q75mq25'] 
        self.target_cols = ['class']
        
        ## Might be better to impute missing data?
        train_const = select_df(data, 'CONST', inf=True).dropna()
        train_all_var = select_df(data, 'VAR').dropna()
        
        training_set = feat_clip(pd.concat([train_const, train_all_var], axis=0), self.data_cols)
        
        run(training_set)
        
        
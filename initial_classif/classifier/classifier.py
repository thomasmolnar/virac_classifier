

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

class binary_classification(object):
    
    def __init__(self, data):
        self.data_cols = ['skewness','kurtosis', 'stetson_i', 'eta',
                         'mags_stdev', 'mags_mad','mags_q1', 'mags_q2', 'mags_q4', 'mags_q8', 'mags_q16',
                         'mags_q32','mags_q50', 'mags_q68', 'mags_q84', 'mags_q92',
                         'mags_q96', 'mags_q98','mags_q99', 'mags_q100mq0', 'mags_q99mq1',
                         'mags_q95mq5','mags_q90mq10', 'mags_q75mq25'] 
        self.target_cols = ['class']
        
        ## Might be better to impute missing data
        train_const = select_df(data, 'CONST', inf=True).dropna()
        train_all_var = select_df(data, 'VAR').dropna()
        
        training_set = feat_clip(pd.concat([train_const, train_all_var], axis=0), data_cols)
        
        X_train, X_test, y_train, y_test = train_test_split(training_set[self.data_cols], 
                                                            training_set[self.target_cols], 
                                                            test_size=0.25, random_state=42)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=1000, min_samples_split=5, 
                                            min_samples_leaf=5, max_features='sqrt',
                                            max_depth=18, class_weight='balanced_subsample')
        self.model.fit(X_train, y_train.values.ravel())
        y_pred = self.model.predict(X_test)
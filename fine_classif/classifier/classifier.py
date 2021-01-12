from initial_classif.classifier.classifier import classification

class variable_classification(classification):
    
    def __init__(self, training_set, config, plot_name=None):
        
        self.data_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
                           "ks_eta",#"ks_eta_e",
                           "ks_stetson_i","ks_stetson_j","ks_stetson_k",
                           "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25",
                           'amp_0', 'amp_1', 'amp_2', 'amp_3', 
                           'amplitude', 'beyondfrac', 'delta_loglik', 'ls_period', 'lsq_period',
                           'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
                           'phi0_phi1_x','phi0_phi2_x','phi0_phi3_x', 
                           'phi1_phi2_x', 'phi1_phi3_x', 'phi2_phi3_x',
                           'phi0_phi1_y','phi0_phi2_y','phi0_phi3_y', 
                           'phi1_phi2_y', 'phi1_phi3_y', 'phi2_phi3_y',
                           'a0_a1', 'a0_a2',
                           'a0_a3', 'a1_a2', 'a1_a3', 'a2_a3','JK_col','HK_col']
        
        self.periodic_features = ['phi0_phi1','phi0_phi2','phi0_phi3', 
                                   'phi1_phi2', 'phi1_phi3', 'phi2_phi3']
        
        self.target_cols = ['var_class']
        
        self.log_transform_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
                                   "ks_eta",#"ks_eta_e",
                                   "ks_stetson_i","ks_stetson_j","ks_stetson_k",
                                   "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25",
                                   'amp_0', 'amp_1', 'amp_2', 'amp_3', 
                                   'ls_period', 'lsq_period',
                                   'amplitude', 'beyondfrac',
                                   'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
                                   'a0_a1', 'a0_a2', 'a0_a3', 'a1_a2', 'a1_a3']
        
        self.training_set = self.run(training_set, plot_name, int(config['var_cores']))
        
        

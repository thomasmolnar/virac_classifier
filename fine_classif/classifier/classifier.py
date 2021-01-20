from initial_classif.classifier.classifier import classification

class variable_classification(classification):
    
    def __init__(self, training_set, config, plot_name=None):
        
        # ls_period is Lomb-Scargle period, lsq_period is from full Fourier fit using a list of trial LS periods
        
        self.data_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
                           "ks_eta",#"ks_eta_e",
                           "ks_stetson_i","ks_stetson_j","ks_stetson_k",
                           "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25",
                           "ks_stdev_over_error","ks_mad_over_error",
                           "ks_p100_p0_over_error","ks_p99_p1_over_error",
                           "ks_p95_p5_over_error","ks_p84_p16_over_error","ks_p75_p25_over_error",
                           'amp_0', 'amp_1', 'amp_2', 'amp_3', 
                           'amplitude', 'beyondfrac', 'delta_loglik', 'lsq_period',
                           'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
                           'phi0_phi1_x','phi0_phi2_x','phi0_phi3_x', 
                           'phi1_phi2_x', 'phi1_phi3_x', 'phi2_phi3_x',
                           'phi0_phi1_y','phi0_phi2_y','phi0_phi3_y', 
                           'phi1_phi2_y', 'phi1_phi3_y', 'phi2_phi3_y',
                           'phi0_phi1_double_x','phi0_phi2_double_x','phi0_phi3_double_x', 
                           'phi1_phi2_double_x', 'phi1_phi3_double_x', 'phi2_phi3_double_x',
                           'phi0_phi1_double_y','phi0_phi2_double_y','phi0_phi3_double_y', 
                           'phi1_phi2_double_y', 'phi1_phi3_double_y', 'phi2_phi3_double_y',
                           'a0_a1', 'a0_a2','a0_a3', 'a1_a2', 'a1_a3', 'a2_a3',
                           'a0_a1_double', 'a0_a2_double', 'a0_a3_double', 'a1_a2_double', 
                           'a1_a3_double', 'a2_a3_double',
                           'JK_col','HK_col', 'peak_ratio_model', 'peak_ratio_data']
        
        self.periodic_features = ['phi0_phi1','phi0_phi2','phi0_phi3', 
                                   'phi1_phi2', 'phi1_phi3', 'phi2_phi3',
                                  'phi0_phi1_double','phi0_phi2_double','phi0_phi3_double', 
                                   'phi1_phi2_double', 'phi1_phi3_double', 'phi2_phi3_double']
        
        self.target_cols = ['var_class']
        
        self.log_transform_cols = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
                                   "ks_eta",#"ks_eta_e",
                                   "ks_stetson_i","ks_stetson_j","ks_stetson_k",
                                   "ks_p100_p0","ks_p99_p1","ks_p95_p5","ks_p84_p16","ks_p75_p25",
                                   "ks_stdev_over_error","ks_mad_over_error",
                                   "ks_p100_p0_over_error","ks_p99_p1_over_error",
                                   "ks_p95_p5_over_error","ks_p84_p16_over_error","ks_p75_p25_over_error",
                                   'amp_0', 'amp_1', 'amp_2', 'amp_3', 
                                   'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
                                   'lsq_period',
                                   'amplitude', 'beyondfrac',
                                   'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
                                   'a0_a1', 'a0_a2', 'a0_a3', 'a1_a2', 'a1_a3', 'a2_a3',
                                   'a0_a1_double', 'a0_a2_double', 'a0_a3_double', 'a1_a2_double', 'a1_a3_double', 'a2_a3_double']
        
        self.training_set = self.run(training_set, plot_name, int(config['var_cores']))
        
        

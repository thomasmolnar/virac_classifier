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
                           'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
                           'amplitude', 'beyondfrac', 'normed_delta_loglik', 'lsq_period',
                           'max_pow', 'max_phase_lag', 'pow_mean_disp', 'phase_lag_mean', 'log10_fap',
                           'phi1_phi0_x','phi2_phi0_x','phi3_phi0_x', 
                           'phi2_phi1_x', 'phi3_phi1_x', 'phi3_phi2_x',
                           'phi1_phi0_y','phi2_phi0_y','phi3_phi0_y', 
                           'phi2_phi1_y', 'phi3_phi1_y', 'phi3_phi2_y',
                           'phi1_phi0_double_x','phi2_phi0_double_x','phi3_phi0_double_x', 
                           'phi2_phi1_double_x', 'phi3_phi1_double_x', 'phi3_phi2_double_x',
                           'phi1_phi0_double_y','phi2_phi0_double_y','phi3_phi0_double_y', 
                           'phi2_phi1_double_y', 'phi3_phi1_double_y', 'phi3_phi2_double_y',
                           'a0_a1', 'a0_a2','a0_a3', 'a1_a2', 'a1_a3', 'a2_a3',
                           'a0_a1_double', 'a0_a2_double', 'a0_a3_double', 'a1_a2_double', 
                           'a1_a3_double', 'a2_a3_double',
                           'JK_col','HK_col', 'peak_ratio_model', 'peak_ratio_data',
                           'Z_scale', 'Z_model',
                           'Y_scale', 'Y_model',
                           'J_scale', 'J_model',
                           'H_scale', 'H_model',
                           'model_amplitude'
#                            'log10_decaps_g_amp','log10_decaps_r_amp','log10_decaps_i_amp','log10_decaps_z_amp',
#                            'nsc2_rmsvar', 'nsc2_madvar', 'nsc2_iqrvar', 'nsc2_etavar', 'nsc2_jvar', 'nsc2_kvar',
#                            'nsc2_chivar', 'nsc2_romsvar', 'nsc2_nsigvar'
#                            'Z_contemp_std','Z_contemp_abs',
#                            'Y_contemp_std','Y_contemp_abs',
#                            'J_contemp_std','J_contemp_abs',
#                            'H_contemp_std','H_contemp_abs'
                         ]
        
        self.periodic_features = ['phi1_phi0','phi2_phi0','phi3_phi0', 
                                   'phi2_phi1', 'phi3_phi1', 'phi3_phi2',
                                  'phi1_phi0_double','phi2_phi0_double','phi3_phi0_double', 
                                   'phi2_phi1_double', 'phi3_phi1_double', 'phi3_phi2_double']
        
        self.no_upper_features = ['JK_col', 'HK_col', 'ks_stetson_i']
        
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
                                   'lsq_period', 'normed_delta_loglik',
                                   'amplitude', 'beyondfrac', 'log10_fap',
                                   'max_pow', 'max_phase_lag', 'pow_mean_disp', 'phase_lag_mean',
                                   'a0_a1', 'a0_a2', 'a0_a3', 'a1_a2', 'a1_a3', 'a2_a3',
                                   'a0_a1_double', 'a0_a2_double', 'a0_a3_double', 
                                   'a1_a2_double', 'a1_a3_double', 'a2_a3_double',
                                   'JK_col','HK_col',
                                   'Z_scale', 'Z_model',
                                   'Y_scale', 'Y_model',
                                   'J_scale', 'J_model',
                                   'H_scale', 'H_model',
                                   'model_amplitude'
#                                    'log10_decaps_g_amp','log10_decaps_r_amp','log10_decaps_i_amp','log10_decaps_z_amp',
#                                    'nsc2_rmsvar', 'nsc2_madvar', 'nsc2_iqrvar', 'nsc2_etavar', 'nsc2_jvar', 'nsc2_kvar',
#                                    'nsc2_chivar', 'nsc2_romsvar', 'nsc2_nsigvar'
#                                    'Z_contemp_std','Z_contemp_abs',
#                                    'Y_contemp_std','Y_contemp_abs',
#                                    'J_contemp_std','J_contemp_abs',
#                                    'H_contemp_std','H_contemp_abs'
                                  ]
        
        self.training_set = self.run(training_set, plot_name, int(config['var_cores']), xgb=True)
        
        

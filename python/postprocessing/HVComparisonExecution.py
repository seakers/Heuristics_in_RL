# -*- coding: utf-8 -*-
"""
Comparing HV evolution of the testing runs of the trained PPO agent and the Epsilon-MOEA runs

@author: roshan94
"""
from Utils.dataHandler import DataHandler
from Utils.normalizationHandler import NormalizationHandler
from Utils.mOORunStatistics import MOORunStatistics
from Utils.mOOCaseStatistics import MOOCaseStatistics
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt

results_dir = 'C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\python\\keras-based\\results'
problem_dir = 'EqStiff\\' # EqStiff, Artery

constr_names = ''
obj_names = ['TrueObjective1','TrueObjective2']
heuristic_names = ['P','N','O','I']
#obj_names = ['Normalized Stiffness', 'Normalized Volume Fraction']

#constr_names = ['FeasibilityViolation','ConnectivityViolation','StiffnessRatioViolation']
constr_names = ['ConnectivityViolation','StiffnessRatioViolation']

# Set parameters for DataHandler.get_objectives() method
objs_norm_num = [0, 0] 
objs_norm_den = [1.8162e6, 1] # Youngs modulus used to normalize stiffness
objs_max = [False, False] 

# To be set to true if negative of any objective is to be used to compute HV, 
# first objective (stiffness) is to be maximized and second objective (volume fraction/deviation) is to be minimized, however -normalized stiffness is stored in csv so -1 multiplication is not required
true_obj_names = [r'$C_{22}$',r'$v_f$']
if problem_dir == 'Artery\\':
    #obj_names = ['Normalized Stiffness', 'Normalized Deviation']
    constr_names = ['FeasibilityViolation','ConnectivityViolation']
    # Set parameters for DataHandler.get_objectives() method
    objs_norm_num = [2e5, 0]
    objs_norm_den = [1e6, 1]
    true_obj_names = [r'$\frac{C_{11}}{v_f}$',r'deviation']
    
n_runs = 20

case_bools = {} # Last integer signifies whether eps-MOEA, AOS or PPO-H is used (1, 2 or 3 respectively) 
eps_moea_bools = [False for i in range(len(heuristic_names))] # No heuristics enforced
eps_moea_bools.append(1)
case_bools['Eps. MOEA'] = eps_moea_bools
# aos_case_bools = [True for i in range(len(heuristic_names))]
# aos_case_bools.append(2)
# case_bools['AOS'] =  aos_case_bools # All heuristics enforced through AOS
ppo_moea_bools = [False for i in range(len(heuristic_names))]
ppo_moea_bools.append(3)
case_bools['PPO-H'] = ppo_moea_bools

plot_colors = {} # black, yellow, blue for Eps. MOEA, AOS & PPO-H respectively
plot_colors['Eps. MOEA'] = '#000000'
# plot_colors['AOS'] = '#E69F00'
plot_colors['PPO-H'] = '#56B4E9'

plot_markers = {}
plot_markers['Eps. MOEA'] = '*'
# plot_markers['AOS'] = 'o'
plot_markers['PPO-H'] = '+'

## Sample NFEs
nfe_samples = []
max_nfe = 6000 

# sample nfes are divided into increments of 50 and 100, computed such that the total number of samples is 100
n_sample_50 = int(1500/50)
n_sample_100 = int((max_nfe - 1500)/100)
for i in range(n_sample_50):
    nfe_samples.append(50*i)
for i in range(n_sample_100):
    nfe_samples.append((n_sample_50*50) + (100*i))
nfe_samples.append(max_nfe)

## Create dictionary of all solution objectives (and constraints if applicable) for all runs for each case (mainly collate internal runs for each coevolutionary run)
case_objs = {}
case_constrs = {}
case_nfes = {}
int_pop_found = False
print('Reading and storing solutions for all cases')

for case_key in list(case_bools.keys()):
    run_objs = {}
    run_constrs = {}
    run_nfes = {}
    heurs_incorporated = case_bools[case_key]
    print('Reading Case: ' + case_key)

    if problem_dir == 'EqStiff\\':
        filename_prob = 'eqstiff_'
    else:
        filename_prob = 'artery_'
    
    heurs_dir = ''
    for i in range(len(heurs_incorporated) - 1):
        if heurs_incorporated[i]:
            heurs_dir += heuristic_names[i]
    
    if not heurs_dir == '':
        heurs_dir += '\\'
            
    if heurs_incorporated[-1] == 1:
        alg_dir = 'Eps MOEA\\'
    elif heurs_incorporated[-1] == 2:
        alg_dir = 'AOS\\'
    else:
        alg_dir = 'PPO-H\\'
    
    for i in range(n_runs):
        nfes_current_run = []
        current_run_objs = []
        current_run_constrs = []
        print('Reading Run: ' + str(i))

        current_filepath = results_dir + problem_dir + alg_dir + heurs_dir + 'run ' + str(i) + '\\'
            
        if heurs_incorporated[-1] == 1:
            run_filename = 'EpsilonMOEA_' + str(i) + '_allSolutions.csv'
        elif heurs_incorporated[-1] == 2:
            run_filename = 'AOSMOEA_' + str(i) + '_allSolutions.csv'
        else:
            run_filename = 'RL_training_designs_ppo_' + filename_prob + str(i) + '.csv'
        
        runfile_datahandler = DataHandler(current_filepath + run_filename)
        runfile_cols = runfile_datahandler.read(ignore_nans=False)
        sorted_runfile_cols = runfile_datahandler.sort_by_nfe()
        
        nfes_current_run = sorted_runfile_cols.get('NFE')
        current_run_objs = runfile_datahandler.get_objectives(obj_names=obj_names, objs_max=objs_max, objs_norm_den=objs_norm_den, objs_norm_num=objs_norm_num)
        if not constr_names == '':
            current_run_constrs = runfile_datahandler.get_parameters(parameter_names=constr_names)
                    
        run_nfes['run'+str(i)] = nfes_current_run
        run_objs['run'+str(i)] = np.stack(current_run_objs, axis=0)
        if not constr_names == '':
            run_constrs['run'+str(i)] = np.stack(current_run_constrs, axis=0)
        
    case_nfes[case_key] = run_nfes
    case_objs[case_key] = run_objs
    if not constr_names == '':
        case_constrs[case_key] = run_constrs

## Create dictionary of Pareto Fronts at sample nfe for all runs for each case
print('Computing and storing Pareto Fronts')
dummy_caseStats = MOOCaseStatistics(hv_allcases={}, nfe_array=nfe_samples, case_names=list(case_bools.keys()))
case_pfs = {}
n_feas_cases = {}
for case_key in list(case_objs.keys()):
    print('Computing for Case: ' + case_key)
    objs_case = case_objs[case_key]
    nfes_case = case_nfes[case_key]
    if not constr_names == '':
        constrs_case = case_constrs[case_key]
    run_pfs = {}
    n_feas_runs = {}
    for run_key in list(objs_case.keys()):
        print('Computing for Run: ' + run_key)
        objs_run = objs_case[run_key]
        if not constr_names == '':
            constrs_run = constrs_case[run_key]
        
        nfes_run = nfes_case[run_key]
        
        pfs_run = {}
        n_feas_nfes = {}
        objs_fullsat = []
        for i in range(len(nfe_samples)):
            nfe_sample = nfe_samples[i]
            #if nfe_sample==1300:
                #print('Stop')
            print('Computing for NFE: ' + str(nfe_sample))
            if i != 0:
                closest_prev_nfe_idx = dummy_caseStats.find_closest_index(val=nfe_samples[i-1], search_list=nfes_run)
            else:
                closest_prev_nfe_idx = 0
                current_pf_objs = []
                current_pf_constrs = []
            closest_nfe_idx = dummy_caseStats.find_closest_index(val=nfe_sample, search_list=nfes_run)
            objs_nfe_sample = objs_run[closest_prev_nfe_idx:closest_nfe_idx,:]
            if len(current_pf_objs) != 0:
                objs_nfe_sample = np.append(objs_nfe_sample, current_pf_objs, axis=0)
            objs_current_unique, current_unique_idx = np.unique(objs_nfe_sample, return_index=True, axis=0)
        
            if not constr_names == '':
                constrs_nfe_sample = constrs_run[closest_prev_nfe_idx:closest_nfe_idx,:]
                if len(current_pf_constrs) != 0:
                    constrs_nfe_sample = np.append(constrs_nfe_sample, current_pf_constrs, axis=0)
                # Check and replace indices to account for designs with same objectives but different constraints (replacing with indices for fully feasible designs)
                new_unique_idx = list(current_unique_idx.copy())
                for obj_idx in current_unique_idx:
                    eq_obj_idx = [i for i in range(objs_nfe_sample.shape[0]) if (np.array_equal(objs_nfe_sample[i,:], objs_nfe_sample[obj_idx,:]) and i!=obj_idx)]
                    if len(eq_obj_idx) > 0:
                        for current_eq_obj_idx in eq_obj_idx:
                            if all(constrs_nfe_sample[current_eq_obj_idx,:] == 0) and any(constrs_nfe_sample[obj_idx,:] != 0):
                                if obj_idx in new_unique_idx:
                                    new_unique_idx.remove(obj_idx)
                                new_unique_idx.append(current_eq_obj_idx)
                                break # just need to add one instance of feasible design with same objectives
                
                objs_current_unique = objs_nfe_sample[new_unique_idx,:]
                constr_current_unique = constrs_nfe_sample[new_unique_idx,:]
                
                feas = [x for x in constr_current_unique if np.sum(x) == 0] # find number of fully feasible designs at different NFE
                
                # Add fully feasible objectives to objs_fullsat
                feas_idx = [idx for idx in range(len(constr_current_unique)) if all(constr_current_unique[idx] == 0)]
                for idx_val in feas_idx:
                    if not any(np.array_equal(f, objs_current_unique[idx_val,:]) for f in objs_fullsat): # check if design is already present in objs_fullsat
                        objs_fullsat.append(objs_current_unique[idx_val,:])
                    
                # Isolate the unique instances of objectives and constraints
                #objs_nfe_sample, unique_idx = np.unique(objs_nfe_sample, return_index=True, axis=0)
                #constrs_nfe_sample = constrs_nfe_sample[unique_idx]
                
                #aggr_constrs_nfe_sample = [np.mean(x) for x in constrs_nfe_sample]
                aggr_constrs_nfe_sample = [np.mean(x) for x in constr_current_unique]
                if len(aggr_constrs_nfe_sample) != 0: 
                    #current_mooRunStats = MOORunStatistics(objs_nfe_sample, aggr_constrs_nfe_sample)
                    current_mooRunStats = MOORunStatistics(objs_current_unique, aggr_constrs_nfe_sample)
                    pfs_idx_nfe = current_mooRunStats.compute_PF_idx_constrained(only_fullsat=True)
                else: # would happen if closest_prev_nfe_idx = closest_nfe_idx
                    pfs_idx_nfe = []
                n_feas_nfes['nfe:'+str(nfe_sample)] = len(objs_fullsat)
                #if i != 0:
                    #n_feas_nfes['nfe:'+str(nfe_sample)] = len(feas) + n_feas_nfes['nfe:'+str(nfe_samples[i-1])]
                #else:
                    #n_feas_nfes['nfe:'+str(nfe_sample)] = len(feas) 
            else:
                if len(objs_nfe_sample) != 0: 
                    # Isolate the unique instances of objectives 
                    #objs_nfe_sample = np.unique(objs_nfe_sample, axis=0)
                    objs_current_unique = np.unique(objs_nfe_sample, axis=0)
                    current_mooRunStats = MOORunStatistics(objs_current_unique)
                    pfs_idx_nfe = current_mooRunStats.compute_PF_idx_unconstrained()
                else: # would happen if closest_prev_nfe_idx = closest_nfe_idx
                    pfs_idx_nfe = []
                
            #current_pf_objs = objs_nfe_sample[pfs_idx_nfe]
            current_pf_objs = objs_current_unique[pfs_idx_nfe,:]
            if not constr_names == '':
                current_pf_constrs = constr_current_unique[pfs_idx_nfe,:]
            pfs_run['nfe:'+str(nfe_sample)] = current_pf_objs

        run_pfs[run_key] = pfs_run
        if not constr_names == '':
            n_feas_runs[run_key] = n_feas_nfes
    case_pfs[case_key] = run_pfs                
    if not constr_names == '':
        n_feas_cases[case_key] = n_feas_runs

## Plot Evolution of Pareto Fronts
n_nfes = 5
pf_nfe_markers = ["o", "s", "^", "v", "+"]

for case_key in list(case_objs.keys()):
    print('Plotting Pareto Fronts for Case: ' + case_key)
    pfs_current_case = case_pfs[case_key]
    nfes_current_case = case_nfes[case_key]
    for run_key in list(pfs_current_case.keys()):
        print('Plotting Pareto Fronts for ' + run_key)
        pfs_current_run = pfs_current_case[run_key]
        nfes_current_run = nfes_current_case[run_key]

        nfes_idx_plotting = np.linspace(start=0, end=len(nfes_current_run), num=5)
        idx_counter = 0

        for nfe_idx in nfes_idx_plotting:
            pfs_current_nfe = pfs_current_run['nfe:'+str(nfes_current_run[nfe_idx])]
            plt.figure()
            plt.scatter(pfs_current_nfe[0,:], pfs_current_nfe[1,:], marker=pf_nfe_markers[idx_counter], label='NFE:'+str(nfes_current_run[nfe_idx]))
            plt.xlabel(true_obj_names[0])
            plt.ylabel(true_obj_names[1])
            plt.title('Pareto Fronts')
            plt.legend(loc='best')
            plt.savefig(results_dir + problem_dir + alg_dir + heurs_dir + 'run ' + str(i) + '\\' + 'pareto_fronts.png', dpi=600)
            idx_counter += 1

## Normalize Pareto Fronts 
print('Normalizing Pareto Front objectives')
norm_handler = NormalizationHandler(objectives=case_pfs, n_objs=len(obj_names))
norm_objs = norm_handler.find_objs_normalization()
norm_handler.normalize_objs()
objs_norm = norm_handler.objectives

## Compute and store hypervolume at sampled nfe for all runs for each case
print('Computing Hypervolumes')
hvs_cases = {}
for case_key in list(objs_norm.keys()):
    pfs_norm_case = objs_norm[case_key]
    hvs_runs = {}
    for run_key in list(pfs_norm_case.keys()):
        pfs_norm_run = pfs_norm_case[run_key]
        hvs_nfes = {}
        for nfe_key in list(pfs_norm_run.keys()):
            pfs_norm_nfe = pfs_norm_run[nfe_key]
            nfe_runStats = MOORunStatistics(pfs_norm_nfe)
            nfe_runStats.update_norm_PF_objectives(norm_PF_objectives=pfs_norm_nfe)
            hv_nfe = nfe_runStats.compute_hv()
            hvs_nfes[nfe_key] = hv_nfe
        hvs_runs[run_key] = hvs_nfes
    hvs_cases[case_key] = hvs_runs
    
# Compute HV threshold
hv_max = 0
for case_key in list(hvs_cases.keys()):
    hv_runs = hvs_cases[case_key]
    for run_key in list(hv_runs.keys()):
        if np.max(list(hv_runs[run_key].values())) > hv_max:
            hv_max = np.max(list(hv_runs[run_key].values()))

hv_thresh = 0.8*hv_max

## Compute stats (Wilcoxon test, CDF for NFE and median & interquartile ranges for HVs)

#nfe_samples_stats = np.linspace(0, max_nfe, 11) # NFE values at which Wilcoxon tests will be conducted
nfe_samples_stats = nfe_samples

caseStats = MOOCaseStatistics(hv_allcases=hvs_cases, nfe_array=nfe_samples, case_names=list(case_bools.keys()))
U_test_cases = caseStats.compute_hypothesis_test_Uvals(nfe_samples=nfe_samples_stats, alternative='two-sided')
nfe_hv_attained = caseStats.compute_nfe_hypervolume_attained(hv_threshold=hv_thresh)
hv_med_cases, hv_1q_cases, hv_3q_cases = caseStats.compute_hypervolume_stats()

## Extract HV stats between two NFE for visualization
if  problem_dir == 'Artery\\':
    nfe_start = 10000   
    nfe_end = 13500

if problem_dir == 'Artery\\':
    nfe_start_idx = dummy_caseStats.find_closest_index(val=nfe_start, search_list=nfe_samples)
    nfe_end_idx = dummy_caseStats.find_closest_index(val=nfe_end, search_list=nfe_samples)
    nfe_samples_crop = nfe_samples[nfe_start_idx:nfe_end_idx+1]
    
    hv_med_cases_crop = {}
    hv_1q_cases_crop = {}
    hv_3q_cases_crop = {}
    for case_key in list(case_bools.keys()):
        hv_med_cases_crop[case_key] = hv_med_cases[case_key][nfe_start_idx:nfe_end_idx+1]
        hv_1q_cases_crop[case_key] = hv_1q_cases[case_key][nfe_start_idx:nfe_end_idx+1]
        hv_3q_cases_crop[case_key] = hv_3q_cases[case_key][nfe_start_idx:nfe_end_idx+1]

## Plotting 
# Plot hypervolume stats vs NFE for cases
fig1 = plt.figure()
for case_key in list(case_bools.keys()):
    plt.plot(nfe_samples, hv_med_cases[case_key], label=case_key, color=plot_colors[case_key])
    plt.fill_between(nfe_samples, hv_1q_cases[case_key], hv_3q_cases[case_key], color=plot_colors[case_key], alpha=0.5, edgecolor="none")
plt.xlabel(r'Number of Function Evaluations',fontsize=12)
plt.ylabel(r'Hypervolume',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})

# Plot hypervolume stats vs NFE between two NFE for cases 
if problem_dir == 'Artery\\':
    fig2 = plt.figure()
    for case_key in list(case_bools.keys()):
        plt.plot(nfe_samples_crop, hv_med_cases_crop[case_key], label=case_key, color=plot_colors[case_key])
        plt.fill_between(nfe_samples_crop, hv_1q_cases_crop[case_key], hv_3q_cases_crop[case_key], color=plot_colors[case_key], alpha=0.5, edgecolor="none")
    plt.xlabel(r'Number of Function Evaluations',fontsize=12)
    plt.ylabel(r'Hypervolume',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    

# Plot CDF for NFE vs NFE for cases
fig3 = plt.figure()
for case_key in list(case_bools.keys()):
    nfe_hv_attained_case = nfe_hv_attained[case_key]
    frac_hv_attained = np.zeros((len(nfe_samples)))
    for i in range(len(nfe_samples)):
        nfe_val = nfe_samples[i]
        idx_hv_attained = [x for x in list(nfe_hv_attained_case.values()) if x <= nfe_val]
        frac_hv_attained[i] = len(idx_hv_attained)/len(list(nfe_hv_attained_case))
    plt.plot(nfe_samples, frac_hv_attained, color=plot_colors[case_key], label=case_key)
plt.xlabel(r'Number of Function Evaluations',fontsize=12)
plt.ylabel(r'Frac. runs achieving 80% max. HV',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})

## Plot number of fully feasible designs vs NFE for cases (only for constrained optimization problems)
if not constr_names == '':
    fig4 = plt.figure()
    n_feas_med_allcases = {}
    n_feas_1q_allcases = {}
    n_feas_3q_allcases = {}
    for case_key in list(n_feas_cases.keys()):
        n_feas_case = n_feas_cases[case_key]
        n_feas_nfes = np.zeros((len(nfe_samples), len(list(n_feas_case.keys()))))
        run_counter = 0
        for run_key in list(n_feas_case.keys()):
            n_feas_run = n_feas_case[run_key]
            nfe_counter = 0
            for nfe_key in list(n_feas_run.keys()):
                n_feas_nfes[nfe_counter,run_counter] = n_feas_run[nfe_key]
                nfe_counter += 1
            run_counter += 1
            
        n_feas_med_case = np.zeros((len(nfe_samples)))
        n_feas_1q_case = np.zeros((len(nfe_samples)))
        n_feas_3q_case = np.zeros((len(nfe_samples)))
        for i in range(len(nfe_samples)):
            n_feas_med_case[i] = statistics.median(n_feas_nfes[i,:])
            n_feas_1q_case[i] = np.percentile(n_feas_nfes[i,:], 25)
            n_feas_3q_case[i] = np.percentile(n_feas_nfes[i,:], 75)
            
        plt.plot(nfe_samples, n_feas_med_case, color=plot_colors[case_key], label=case_key)
        plt.fill_between(nfe_samples, n_feas_1q_case, n_feas_3q_case, color=plot_colors[case_key], alpha=0.5)
    plt.xlabel(r'Number of Function Evaluations',fontsize=12)
    plt.ylabel(r'# fully feasible solutions',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from scipy.stats import fisher_exact
from scipy.special import loggamma, betaln, expit
from scipy import sparse
from scipy.optimize import minimize_scalar

def find_group_sizes(counts, max_scaling=None):
    """A utility function that reads a 2-column data frame with sample
    counts for each group (the index) for each of two data sets.
    The method uses optimization to find best scaling factor for downsampling
    one of the datasets and applies corrections for all groups for which there
    are not enough samples to satisfy the required proportions.
    """
    # since column 0 is to be resampled, make sure column 1 does not 
    # have more samples in any group
    count_values = np.vstack([counts[0], np.minimum(counts[0], counts[1])])
    
    if max_scaling is None:
        # a rough approximation of max scaling factor
        totals = count_values.sum(axis=1)
        max_scaling = totals[0] / totals[1]
        
    if max_scaling <= 0:
        raise ValueError('Scaling must be > 0.')
            
    def opt_counts(scale, ref_cost=2):
        """Cost function for a scaling factor.
        ref_cost - the cost of dropping a sample from the reference when it
          can be traded off for more samples in the main data set.
        """
        diff = count_values[1] * scale - count_values[0]
        cost = np.where(diff > 0, diff * ref_cost, diff)
        return np.abs(cost).sum()
    
    opt = minimize_scalar(opt_counts,
                          method='bounded',
                          bounds=(0, max_scaling))
    if not opt.success:
        raise ValueError(f'Cannot find scaling factor: {count_values}')

    # general scaling brings both data sets to similar sizes
    # once comparable, trim individual group sizes for exact matching
    general_scale = opt.x
    
    # desired group sizes for the sampled data
    sizes_0 = np.minimum(np.round(count_values[1] * general_scale), 
                         count_values[0])
    # desired group sizes for the reference data (if some need downsampling)
    sizes_1 = np.minimum(np.round(sizes_0 / general_scale), 
                         count_values[1])
    
    sizes = np.vstack([sizes_0, sizes_1]).astype('int')
    return pd.DataFrame(sizes.T, index=counts.index)



def stratified_sample(df, reference, factors, *, 
                      size_factor=None, 
                      df_scores=None):
    """Resamples df and reference data frames in such a way as to have the same 
    sample proportions with respect to a number of factors (columns) in both 
    data frames. A scaling factor can also be provided to limit the overall size
    of resampled data frame to a multiple (or fraction) of the reference.
    
    Parameters
    ----------
    df : the data frame to resample.
    reference : a data frame that defines the desired distribution of factors.
    factors : list of strings
      Column names or a column name in both df and reference data frames. 
    size_factor : float
      Size ratio of the resulting sub-sampling of df to the reference. Useful, 
      when the resampled df size needs to be related to the size of the 
      reference (like no more than 2x the size). The default is None, meaning 
      that only group proportions are maintained.
    df_scores : pd.Series, numeric, optional
      When provided, df records are removed in the order of decreasing scores
      instead of random down-sampling.
      
    Returns
    -------
    Concatenation of resampled df and reference data frames.
    """
    if df_scores is not None:
        common_ix = df.index.union(reference.index)
        df_scores = df_scores.reindex(common_ix)\
            .fillna(df_scores.min())\
            .sort_index()
        if df_scores.isna().all():
            raise ValueError('All scores cannot be NA')
        
    def select_samples(index, n):
        """Selects n samples to retain from index.
        """
        if n == len(index):
            return index
        if df_scores is None:
            return np.random.choice(index, n, replace=False)
        else:
            # n samples with lowest scores
            return df_scores.loc[index].sort_values().index[:n]
        
    df_groupby = df.groupby(factors, observed=True)
    ref_groupby = reference.groupby(factors, observed=True)
    
    counts = pd.concat([df_groupby.size(), ref_groupby.size()], 
                       keys=[0, 1], axis=1).fillna(0)
    
    # check if there's any data to be returned (expecting at least one
    # row to have non zero values everywhere).
    if (counts.product(axis=1) == 0).all():
        return pd.DataFrame(columns=df.columns)
    
    parts = []
    grp_sizes = find_group_sizes(counts, size_factor)
    for id, part_df, groupby in zip([0, 1], 
                                    [df, reference], 
                                    [df_groupby, ref_groupby]):
        ix_lst = []
        for group, ix in groupby.groups.items():
            n = grp_sizes.loc[group, id]
            if n > 0:
                ix_lst.append(select_samples(ix, n))
        parts.append(part_df.loc[np.sort(np.hstack(ix_lst))])
        
    return pd.concat(parts)






def gamma2pip(gamma_pve, p, burnin):


    gamma_counts = np.zeros(p)
    len_gamma = len(gamma_pve)
    burnin_start = int(burnin*len_gamma)
    indx = range(burnin_start, len_gamma)
    for j in indx:
        gamma = np.zeros(p)
        gamma[gamma_pve[j]] = 1
        gamma_counts += gamma

    gamma_prop = gamma_counts/len(indx)
    return gamma_prop



def get_log_lik_sparse(X_matrices, gamma):

    p_bar = int(np.sum(gamma!=0))
    n_labels = len(X_matrices)
    
    signature = np.random.randn(p_bar)
    
    vals_summaries = []
    for X_matrix in X_matrices:
        int_i = X_matrix[:,gamma!=0].dot(signature)

        vals_i, counts_i = np.unique(int_i, return_counts=True)
        vals_summaries += [pd.DataFrame(counts_i,index=vals_i)]

    summary_df = pd.concat(vals_summaries,axis=1)
    summary_df = summary_df.fillna(0) + 1
    
    gamma_vals = loggamma(summary_df).sum(axis=1)
    gamma_vals += (loggamma(n_labels) - loggamma(summary_df.sum(1) ))#+ n_labels))
    
    res = gamma_vals.sum()

    return res




class MCMC(object):
    
    def construct_X_y_mats(self, gamma_dash=None):
  
        Y_data = self.Y            
        Y_unique = np.unique(Y_data)
        X_matrices = []
        for y_val in Y_unique:
            X_matrices += [self.X_sparse[Y_data==y_val,:]]
            
        return X_matrices, Y_data
        
    
    def __init__(self, X_sparse, X_col_names, Y, out_dir, n_rep, 
                 burnin=0.2, save_freq=None, 
                 use_gibbs=False, r_gibbs=False,
                 marginalise_out_pi_i=False, expected_model_size=-1, pi_i=None, gamma_positive=False,
                verbose=False):
        
        self.verbose = verbose
        
                    
        
        if r_gibbs and not use_gibbs:
            raise ValueError('Must have use_gibbs=True to have random gibbs')
            
        
        
        if (save_freq is None) and (not use_gibbs):
            save_freq = np.max([int(n_rep/10),1])
            if verbose:
                print('\nSaving results every {} iterations'.format(save_freq))
        elif (save_freq is None) and (use_gibbs) and (not r_gibbs):
            save_freq = 10
            if verbose:
                print('\nSaving results every {} iterations'.format(save_freq))

        
        self.X_sparse = X_sparse
        self.p = X_sparse.shape[1]
        self.X_col_names = X_col_names
        self.Y = Y
        
        

        X_matrices, Y_data = self.construct_X_y_mats()
        self.X_matrices = X_matrices
        self.Y_data = Y_data
        

        self.n_rep = n_rep
        self.burnin = burnin
        self.marginalise_out_pi_i = marginalise_out_pi_i
        self.gamma_positive = gamma_positive
        
        assert(expected_model_size!=0)
        self.expected_model_size = expected_model_size
        
        if pi_i is None:
            self.explore_pi_i = True
        else:
            assert(self.marginalise_out_pi_i==False)
            self.explore_pi_i = False
            self.pi_i = pi_i
        
        self.gamma_pve = []
        self.score_collect = []
        self.pi_i_collect = np.zeros(n_rep)
        self.out_dir = out_dir
        self.save_freq = save_freq
        self.use_gibbs = use_gibbs
        self.r_gibbs = r_gibbs

        
        
    
    def log_score_gamma(self, gamma, X_matrices=None, Y_data=None):
        #assert(len(gamma)==len(X_b.columns))
        #return (get_target(X_b[X_b.columns[1==gamma]],Y_b))*3e4

        if X_matrices is None:
            X_matrices = self.X_matrices
        
        if Y_data is None:
            Y_data = self.Y_data
        
        abs_gamma = np.abs(gamma)
        if np.any(abs_gamma>0):
            ll1 = get_log_lik_sparse(X_matrices, gamma)
        elif not self.gamma_positive:
            Y_unique = np.unique(Y_data)
            n_obs = np.array([np.sum(Y_data==y_i) for y_i in Y_unique])
            ll1 = np.sum(loggamma(n_obs+1)) + loggamma(len(n_obs)) - loggamma(np.sum(n_obs)+len(n_obs))
        else:
            ll1 = -np.inf
            # print(n_obs, len(n_obs))
            
        if self.marginalise_out_pi_i:
            if self.expected_model_size>=0:
                m = self.expected_model_size
                k_i = abs_gamma.sum()
                p = gamma.shape[0]
                ll2 = loggamma(1+k_i)
                ll3 = loggamma(((p-m)/m)+p-k_i)  
            else:
                ll2 = 0
                ll3 = betaln(abs_gamma.sum()+1, (1-abs_gamma).sum()+1)
        else:
            ll2 = abs_gamma.sum()*np.log(self.pi_i)
            ll3 = (1-abs_gamma).sum()*np.log(1-self.pi_i)
            #raise NotImplemented('Need to implement not integrating out pi_i and passing pi_i around')

            
        return ll1 + ll2 + ll3





    def P1_kernel(self, gamma):

        indx = np.random.choice(len(gamma),1)

        gamma_prop = gamma.copy()
        gamma_sel = gamma_prop[indx]

        choices = [0.,1.]
        choices.remove(float(gamma_sel))
        gamma_prop[indx] = np.random.choice(choices)

        score_curr = self.log_score_gamma(gamma)

        n_gamma = gamma_prop.sum()

        score_prop = self.log_score_gamma(gamma_prop)

        if (np.log(np.random.rand(1))<=(score_prop-score_curr)):
            gamma_res = gamma_prop
            score_res = score_prop
        else:
            gamma_res = gamma
            score_res = score_curr


        return gamma_res, score_res




    def P2_kernel(self, gamma):

        gamma_prop = gamma.copy()

        score_curr = self.log_score_gamma(gamma)

        n_pve = np.sum(gamma>0)

        if n_pve>0:

            prop1 = np.random.choice(np.where(gamma==1)[0],1,replace=False)
            prop0 = np.random.choice(np.where(gamma==0)[0],1,replace=False)
    
            gamma_prop[prop1] = 0.
            gamma_prop[prop0] = 1
    
            n_gamma = gamma_prop.sum()
    
            score_prop = self.log_score_gamma(gamma_prop)
    
            if (np.log(np.random.rand(1))<=(score_prop-score_curr)):
                gamma_res = gamma_prop
                score_res = score_prop
            else:
                gamma_res = gamma
                score_res = score_curr
        else:
            gamma_res = gamma
            score_res = score_curr

        return gamma_res, score_res



    def Gibbs_kernel(self, gamma, random_choice=False, high_probability_indicators=None):

        scores = []

        iter_arr = range(len(gamma))

        for i in iter_arr:
            # if self.verbose and (i%100==0):
            #    print(i/len(iter_arr),end='\r',flush=True)
            gamma_plus = gamma.copy()
            gamma_plus[i] = 1
            gamma_zero = gamma.copy()
            gamma_zero[i] = 0

            score_prop_plus = self.log_score_gamma(gamma_plus)
            score_prop_zero = self.log_score_gamma(gamma_zero)

            score_arr = np.array([score_prop_zero, score_prop_plus])
            
            score_arr = score_arr-np.max(score_arr)
            exp_score_arr = np.exp(score_arr)
            w_arr = exp_score_arr/np.sum(exp_score_arr)

            r_select = np.random.choice(len(w_arr),1,p=w_arr)

            if r_select==0:
                gamma = gamma_zero
                score_res = score_prop_zero
            elif r_select==1:
                gamma = gamma_plus
                score_res = score_prop_plus
            else:
                raise ValueError('Invalid choice')
            scores += [score_res]
    
                
        return gamma, scores
    
    
    def calculate_pip(self):

        self.gamma_prop = gamma2pip(self.gamma_pve, self.p, self.burnin)
        return self.gamma_prop
        
    def run(self, pi_i=0.5, r_n_gamma0=None, print_every_iter=False, gamma_init=None):
        
        if (r_n_gamma0 is None) and (gamma_init is None):
            raise ValueError('Must specify initial condition')
        if (r_n_gamma0 is not None) and (gamma_init is not None):
            raise ValueError('Must specify only one initial condition')
            
        # BVS
        X_matrices = self.X_matrices  #qv_df.loc[pve_ids,:].values
        p = self.p

        gamma = np.zeros(p)
        
        other_val = 0

        if r_n_gamma0 is not None:
            choices = np.random.choice(p, r_n_gamma0, replace=False)
            gamma[choices] = 1.
        else:
            gamma = gamma_init.copy()
        

        if not self.use_gibbs:
            n_kernels = 2
        else:
            n_kernels = 1

        accs = 0
        tic = datetime.now()
        for i in range(self.n_rep):
            
            
            if print_every_iter:
                print('iter: %i, acc=%f' % (i,accs/(1+i)))
                                

            if self.explore_pi_i:
                abs_gamma = np.abs(gamma)
                self.pi_i = np.random.beta(1+np.sum(abs_gamma),1+len(abs_gamma)-np.sum(abs_gamma))
                
            if ((i+1) % self.save_freq)==0:
                print('iter: %i, acc=%f' % (i+1,accs/(1+i)))
                dt =( datetime.now()-tic).total_seconds()
                self.save_mcmc_output(i, dt)

            kernel_choice = np.random.choice(n_kernels,1)

            if not self.use_gibbs:
                if kernel_choice==0:
                    gamma_new, score = self.P1_kernel(gamma)
                elif kernel_choice==1:
                    gamma_new, score = self.P2_kernel(gamma)
                self.score_collect += [score]
            else:
                if self.r_gibbs:
                    gamma_new, score = self.Gibbs_kernel(gamma, random_choice=True)
                else:
                    gamma_new, score = self.Gibbs_kernel(gamma, random_choice=False )
                self.score_collect += [score]

                        
            if np.any(gamma_new!=gamma):
                accs+=1

            gamma = gamma_new.copy()
            
            self.gamma_pve += [np.where(gamma>0)[0]]

            self.pi_i_collect[i] = self.pi_i
            


        toc = datetime.now()

        print('time taken for %i iterations' % self.n_rep)
        print(toc-tic)
        
        if self.n_rep>0:

            dt =( datetime.now()-tic).total_seconds()
            self.save_mcmc_output(i, dt)
        
     

    def save_mcmc_output(self, i, dt):

        out_dir = self.out_dir
        

        gamma_prop = self.calculate_pip()

        pip_df = pd.DataFrame(gamma_prop, index=self.X_col_names).sort_values(0,ascending=False)
        pip_df.columns = ['PIP']

        self.pip_df = pip_df
        
        pip_out_fname = os.path.join(out_dir, 'pip.csv')
        pip_df.to_csv(pip_out_fname)
        print('written to ',pip_out_fname)

        pickle_out = {
            'mcmc' : self,
            'pip_df':pip_df,
            'score_collect': self.score_collect,
            'pi_i_collect': self.pi_i_collect[:i],
            'gamma_pve': self.gamma_pve,
            'n_iter':i,
            'n_rep': self.n_rep,
            'dt':dt
        }
        
        self.dt = dt

        out_fname = os.path.join(out_dir, 'out.pcl')
        pickle.dump(pickle_out, open( out_fname, 'wb' ))
        print('written to ',out_fname)
        
        

    def pred(self, X_test, return_counts=False, **kwargs):
        pip_min_freq=0.75

        if len(kwargs)>0:
            valid_kwargs = ['gamma_val']
            if not np.all([i in valid_kwargs for i in kwargs]):
                raise ValueError('Invalid kwarg')
        
        if 'gamma_val' in kwargs:
            gamma_val = kwargs['gamma_val']
            gamma_mask = gamma_val==1
        else:
            gamma_prop = self.calculate_pip()
            gamma_val = 1*(gamma_prop>=pip_min_freq)
            gamma_val += -1*((-gamma_prop)>=pip_min_freq)

            
        gamma_mask = gamma_val!=0
        p_bar = gamma_val[gamma_mask].shape[0]
        n_labels = len(self.X_matrices)

        signature = np.random.randn(p_bar)

        vals_summaries = []
        for X_matrix in self.X_matrices:
            int_i = X_matrix[:,gamma_val==1].dot(signature)
                
            vals_i, counts_i = np.unique(int_i, return_counts=True)
            vals_summaries += [pd.DataFrame(counts_i,index=vals_i)]

        summary_df = pd.concat(vals_summaries,axis=1)
        raw_counts = summary_df.copy()
        summary_df = summary_df.fillna(0) + 1
        # summary_df.index = summary_df.index.astype(int)
        summary_df.columns = range(len(summary_df.columns))
        

        hashes = X_test[:,gamma_val==1].dot(signature)
    

        #valid_new_hashes = list(set(new_hashes) & set(summary_df.index))

        hash_summary_lst = []
        for hash_val in hashes:
            if hash_val in summary_df.index:
                hash_summary_lst += [summary_df.loc[hash_val]]
            else:
                hash_summary_lst += [pd.Series(np.ones(n_labels))]


        alpha_df = pd.concat(hash_summary_lst,axis=1).T
        alpha_df = alpha_df.divide(alpha_df.sum(1), axis=0)

        if not return_counts:
            return alpha_df
        else:
            return alpha_df, raw_counts

        
        
        
class VB(object):


    def evaluate_bayes_uni(self):

        X = self.X_sparse
        Y = self.y
        
        alpha=1 
        beta=1
        alpha0=1 
        beta0=1
    
        alpha_1 = X[Y==1].sum(axis=0)
        beta_1 = X[Y==0].sum(axis=0)
        alpha_0 = (Y==1).sum()  - X[Y==1].sum(axis=0)
        beta_0 =  (Y==0).sum()  - X[Y==0].sum(axis=0)
        alpha_sum = (Y==1).sum()
        beta_sum = (Y==0).sum()

        
    
        logb1 = betaln(alpha_0 + alpha, beta_0 + beta) - betaln(alpha, beta) + betaln(alpha_1 + alpha, beta_1 + beta) - betaln(alpha, beta)
        logb0 = betaln(alpha_sum + alpha0, beta_sum + beta0) - betaln(alpha0, beta0)
        # calc1 = expit(logb1-logb2+np.log(0.1))
        self.logb1 = logb1.A1 + self.n_gamma_prior(1)
        self.logb0 = logb0 + self.n_gamma_prior(0)
        
    
    def __init__(self, X, X_cols, y, N=1, out_dir=None, verbose=False, alpha_param=1, beta_param=1, p_gamma=None, use_univariate=False, use_cache=False, beta_prior_param=1, alpha_prior_param=1, covariates=None):
        
        self.N = N
        self.verbose = verbose
        self.q_record = []
        self.q_conditional = np.zeros(len(X_cols))
        self.p_gamma = p_gamma
        
        X_sparse = X.tocsc()

        Y_unique = np.sort(np.unique(y))
        if len(Y_unique)>2:
            raise NotImplementedError('Need to handle multiclass output')

        self.X_matrices = []
        for y_val in Y_unique:
            self.X_matrices += [X_sparse[y==y_val,:]]


        self.covariates = covariates
        if covariates is not None:
            self.covariate_keys = []
            covariate_signature = np.random.randn(covariates.shape[1])
            for y_val in Y_unique:
                self.covariate_keys += [covariates[y==y_val,:].dot(covariate_signature)]
        
        self.X_sparse = X_sparse
        self.y = y

        
        self.X_cols = X_cols
        self.out_dir = out_dir
        self.t_start = datetime.now()
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        self.alpha_prior_param = alpha_prior_param
        self.beta_prior_param = beta_prior_param

        self.can_reuse_uni = False
        if (alpha_param==1) and (beta_param==1) and (covariates is None):
            self.can_reuse_uni = True
            self.use_univariate = use_univariate
            self.evaluate_bayes_uni()


        if (not self.can_reuse_uni) and use_univariate:
            raise ValueError('Cannot use univariate speed up')
        else:
            self.use_univariate = use_univariate

        self.use_cache = use_cache

        

    def n_gamma_prior(self, n_gamma):
        p0 = len(self.X_cols)
        
        if self.p_gamma is None:
            ll3 = betaln(n_gamma+self.alpha_prior_param, (p0-n_gamma)+self.beta_prior_param)
        else:
            ll3 = n_gamma*np.log(self.p_gamma) + (p0-n_gamma)*np.log(1-self.p_gamma)
        return ll3
        
    def get_log_lik_sparse(self, gamma):

        if gamma.sum()==1:
            if self.use_univariate:
                return self.logb1[np.where(gamma==1)[0][0]]

        
        if gamma.sum()==0:
            if self.use_univariate:
                return self.logb0

        X_matrices = self.X_matrices
        
        p_bar = np.sum(gamma)
        n_labels = len(X_matrices)

        signature = np.random.randn(np.sum(gamma!=0))

        vals_summaries = []
        for i, X_matrix in enumerate(X_matrices):

            int_i = X_matrix[:,gamma!=0].dot(signature)

            if self.covariates is not None:
                # print('using covariates')
                int_i += self.covariate_keys[i]
            # vals_i, counts_i = np.unique(int_i, return_counts=True)
            counted_i = pd.Series(int_i).groupby(int_i).size()
            vals_i, counts_i = counted_i.index, counted_i.values
            
            vals_summaries += [pd.DataFrame(counts_i,index=vals_i)]

        summary_df = pd.concat(vals_summaries,axis=1)
        summary_df = summary_df.fillna(0)
        
        alpha_beta_df = 0 * summary_df.copy()
        alpha_beta_df[alpha_beta_df.index!=0] += np.array([self.beta_param, self.alpha_param])
        alpha_beta_df[alpha_beta_df.index==0] += np.array([1, 1])
        
        gamma_vals = loggamma(summary_df + alpha_beta_df).sum(axis=1)
        gamma_vals -= loggamma(alpha_beta_df).sum(axis=1)
        gamma_vals += loggamma(alpha_beta_df.sum(axis=1))
        gamma_vals -= loggamma((summary_df + alpha_beta_df).sum(axis=1))


        res = gamma_vals.sum()

        n_gamma = gamma.sum()
        ll3 = self.n_gamma_prior(n_gamma)

        return res + ll3

    
    def run(self, n_rep):
        N = self.N
        
        function_cache = {}
        cache_key = np.random.randn(len(self.X_cols))

        self.tic = datetime.now()
        self.q_record = []
        for iter_i in range(n_rep):

            for i in list(range(len(self.X_cols))):
                if self.verbose:
                    if i % 100 ==0:
                        print(i/len(self.X_cols),end='\r',flush=True)

                q_pve = 0
                q_nve = 0
                for j in range(N):

                    gamma_val = 1*(np.random.rand(len(self.X_cols))<=self.q_conditional)
                    gamma_val0 = gamma_val.copy()
                    gamma_val0[i] = 0

                    gamma_val1 = gamma_val.copy()
                    gamma_val1[i] = 1


                    if self.use_cache:
                        signature0 = cache_key[gamma_val0==1].sum()
                        if signature0 in function_cache:
                            lp0 = function_cache[signature0]
                        else:
                            lp0 = self.get_log_lik_sparse(gamma_val0)
                            function_cache[signature0] = lp0
    
                        signature1 = cache_key[gamma_val1==1].sum()
                        if signature1 in function_cache:
                            lp1 = function_cache[signature1]
                        else:
                            lp1 = self.get_log_lik_sparse(gamma_val1)
                            function_cache[signature1] = lp1
                    else:
                        lp0 = self.get_log_lik_sparse(gamma_val0)
                        lp1 = self.get_log_lik_sparse(gamma_val1)
  
                    q_pve += lp1
                    q_nve += lp0

                self.q_conditional[i] = expit((q_pve-q_nve)/N)


            self.q_record += [self.q_conditional.copy()]
            
            # if self.out_dir is not None:
            self.save_results()

        dt = datetime.now()-self.tic
        print(dt)

    def save_results(self):
        
        qpip_df = pd.DataFrame(self.q_conditional, index=self.X_cols).sort_values(0,ascending=False)
        qpip_df.columns = ['qPIP']

        self.qpip_df = qpip_df
        self.dt = datetime.now()-self.t_start

        if self.out_dir is not None:
            pip_out_fname = os.path.join(self.out_dir, 'pip.csv')
            qpip_df.to_csv(pip_out_fname)
            print('written to ',pip_out_fname)

        
            pickle_out = {
                'q_record' : self.q_record,
                'dt' : self.dt 
            }
            out_fname = os.path.join(self.out_dir,'out.pcl')
            f = open( out_fname, 'wb' )
            pickle.dump(pickle_out, f)
            f.close()
            print('written to ',out_fname)
            
        



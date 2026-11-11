##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################


import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge

from utils.misc import prepare_inputs


class Model(nn.Module):
    def __init__(self, cfg):
        """
        OLS wrapper to simplify some of the OLS fitting. 
        args:
            dataset_train: training dataset, will be turned into training instances with the numpy sliding_window_view trick
            context_length: features that the linear model will see
            horizon: up until where the linear model will forecast
            instance_norm: switch on or off instance normalisation, which equates to subtracting the mean for a linear model
            individual: determines if separate model should be learned per channel or one shared across channels. Enabled only for 'weather' dataset as in DLinear.
            seed: for repeatability when using SVD solver
            max_train_N: set this if your dataset (N_samples * N_variables) is very large and you want to subsample
        """
        super(Model, self).__init__()
        self.cfg = cfg
        self.context_length = cfg.seq_len
        self.horizon = cfg.pred_len
        self.individual = cfg.individual
        self.instance_norm = cfg.instance_norm
        self.verbose = True
        #  Disable 'fit_intercept' in Ridge regresion when instance normalization is used. This adjustment is necessary
        #  because the bias (intercept) term is implicitly handled through normalization, specifically by appending the
        #  context standard deviation as a feature to each instance. Refer to Table 1 in paper for details on this setup.
        fit_intercept = False if self.instance_norm else True

        if self.individual:
            self.models = []
            for _ in range(dataset_train.shape[1]):
                self.models.append(Ridge(alpha=alpha,  # This has no appreciable impact for regularisation but is instead set for stability 
                                         fit_intercept=fit_intercept, 
                                         tol=0.00001, 
                                         copy_X=True, 
                                         max_iter=None, 
                                         solver='svd', 
                                         random_state=seed))
        else:
            if self.instance_norm:
                self.linear = nn.Linear(self.context_length + 1, self.horizon, bias=fit_intercept)
            else:
                self.linear = nn.Linear(self.context_length, self.horizon, bias=fit_intercept)
            self.alpha = cfg.alpha
    
    def fit_ols_solutions(self, train_loader):
        """
        Fit the OLS solutions for each series or in a global mode.
        self.dataset.shape = (D, V), where D is the length (in time) and V is the number of variables.
        """
        if self.verbose:
            print('Fitting')

        if self.individual:
            raise NotImplementedError("Individual OLS mode is not supported in this project runtime.")

        feature_dim = self.context_length + 1 if self.instance_norm else self.context_length
        device = self.linear.weight.device
        lhs = torch.zeros((feature_dim, feature_dim), dtype=torch.float64, device="cpu")
        rhs = torch.zeros((feature_dim, self.horizon), dtype=torch.float64, device="cpu")

        for inputs in train_loader:
            enc_window, _, dec_window, _ = prepare_inputs(inputs)
            dec_window = dec_window[:, -self.horizon:, :]

            if self.instance_norm:
                means = enc_window.mean(1, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
                enc_window = enc_window - means
                dec_window = dec_window - means
                enc_window = torch.concat([enc_window, stdev], dim=1)

            enc_window = enc_window.permute(0, 2, 1).reshape(-1, enc_window.shape[1]).detach().to(dtype=torch.float64, device="cpu")
            dec_window = dec_window.permute(0, 2, 1).reshape(-1, dec_window.shape[1]).detach().to(dtype=torch.float64, device="cpu")

            lhs += enc_window.t() @ enc_window
            rhs += enc_window.t() @ dec_window

        reg = self.alpha * torch.eye(feature_dim, dtype=torch.float64, device="cpu")
        weight = torch.linalg.solve(lhs + reg, rhs)
        self.linear.weight.data.copy_(weight.t().to(dtype=self.linear.weight.dtype, device=device))
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Using the pre-fitted models and context, x, predict to horizon
        """
        x_dec = x_dec[:, -self.horizon:, :]
        if self.instance_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = torch.concat([x_enc, stdev], dim=1)
        
        if self.individual:
            preds = []
            for series_idx in range(X.shape[1]):
                pred_i = self.models[series_idx].predict(X[:,series_idx])
                preds.append(pred_i[:,np.newaxis])
            preds = np.concatenate(preds, axis=1)
        else:
            x_enc = x_enc.permute(0, 2, 1)
            pred = self.linear(x_enc)
            preds = pred.permute(0, 2, 1)
        
        if self.instance_norm:
            return preds + means # Undo instance norm
        else:
            return preds

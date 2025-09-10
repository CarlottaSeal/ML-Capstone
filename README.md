# FinTSBridge
We provide experimental code in `./FinTSBridge_models/` folder, with the majority of model experiment run scripts located in the `./FinTSBridge_models/TSLib_baseline/` folder, the scripts for the PSformer model are in the `./FinTSBridge_models/PSformer_baseline/` folder.

To run the models, please refer to the corresponding `.sh` files in the corresponding scripts folder. Among these scripts, `Full_variates_forecasting` corresponds to the multivariate forecasting multivariate experiment task, `Univariate_forecasting` corresponds to the multivariate forecasting univariate experiment task, and `Partial_variates_forecasting` corresponds to the multivariate forecasting partial variates experiment task. Please configure the environment variables by referring to the `./requirements.txt`.

For example, to run the TimesNet model for the multivariate forecasting multivariate task, execute the following command: \
`sh ./FinTSBridge_models/TSLib_baseline/scripts/FinTS_script/Full_variates_forecasting/TimesNet_FBD.sh`

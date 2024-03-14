import pref_param_importance_script
import logging

from Scripts.PrefModel import pref_validation_script, pref_standardsim_script, pref_runsim_script, \
    pref_randomforest_script, pref_sensitivity_analysis_script

output_folder = '../../Dropbox/Results/PrefModel/'
output_folder_pref_forest = '../../Dropbox/Results/PrefModel/RandomForest'

logger = logging.getLogger('my_application')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(output_folder + 'log.txt')
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)

try:
    # Testing and Code Validation
    # pref_validation_script.main()

    # Standard simulations
    # pref_standardsim_script.main()
    # pref_runsim_script.main()

    # Random forest parameter tuning
    # pref_randomforest_validation_script.main()

    # Random forest model
    pref_randomforest_script.main()

    # Sensitivity Analysis
    # pref_sensitivity_analysis_script.main()

    # Parameter Importance
    # pref_param_importance_script.main()

except Exception as e:
    logger.exception(e)

# Release 0.3.0
- Allow user to insert information per operating point
- Retrieve operating points based on given criteria

# Release 0.2.0

## Features
- introduced a new plot function for displaying per-class PSD-ROC.
- added support for external axes to the plot_psd_roc function.
- added the duration_unit as class attribute. It is then returned by the
PSDSEval.psds function in the PSDS namedtuple.
- added a jupyter notebook that shows how the PSDS can be used to extract.
performance insights for the DCASE2020 Task 4 baseline system. It also explain
in more details few features of PSDS metric.

## Minor changes
- fixed a typo in README.md

# Release 0.1.0

## Features
- Added function for computing macro F-score. It is defined as the average of the
per class F-score metrics.

## Bug fixes
- Relaxed check of increasing monotonicity for eTPR (effective True Positive Rate).
This check is performed in the function that calculates the area under the PSD-ROC
curve and it is now disabled when the parameter `alpha_ct > 0`. In this case such
property can't be guaranteed.
- The confusion matrix (`counts`) was built with inverse axis which then caused
the cross-trigger rate to be assigned to the wrong class.

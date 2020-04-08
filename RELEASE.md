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

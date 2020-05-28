#!/usr/bin/env python3
"""
A script that calculates the PSDS using example data from DCASE 2019.
"""

import os
import numpy as np
import pandas as pd
from psds_eval import (PSDSEval, plot_psd_roc, plot_per_class_psd_roc)

if __name__ == "__main__":
    dtc_threshold = 0.5
    gtc_threshold = 0.5
    cttc_threshold = 0.3
    alpha_ct = 0.0
    alpha_st = 0.0
    max_efpr = 100

    # Load metadata and ground truth tables
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ground_truth_csv = os.path.join(data_dir, "dcase2019t4_gt.csv")
    metadata_csv = os.path.join(data_dir, "dcase2019t4_meta.csv")
    gt_table = pd.read_csv(ground_truth_csv, sep="\t")
    meta_table = pd.read_csv(metadata_csv, sep="\t")

    # Instantiate PSDSEval
    psds_eval = PSDSEval(dtc_threshold, gtc_threshold, cttc_threshold,
                         ground_truth=gt_table, metadata=meta_table)

    # Add the operating points, with the attached information
    for i, th in enumerate(np.arange(0.1, 1.1, 0.1)):
        csv_file = os.path.join(data_dir, f"baseline_{th:.1f}.csv")
        det_t = pd.read_csv(os.path.join(csv_file), sep="\t")
        info = {"name": f"Op {i + 1}", "threshold": th}
        psds_eval.add_operating_point(det_t, info=info)
        print(f"\rOperating point {i+1} added", end=" ")

    # Calculate the PSD-Score
    psds = psds_eval.psds(alpha_ct, alpha_st, max_efpr)
    print(f"\nPSD-Score: {psds.value:.5f}")

    # Plot the PSD-ROC
    plot_psd_roc(psds)

    # Plot per class tpr vs fpr/efpr/ctr
    tpr_vs_fpr, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=1.)
    plot_per_class_psd_roc(tpr_vs_fpr, psds_eval.class_names,
                           title="Per-class TPR-vs-FPR PSDROC",
                           xlabel="FPR")
    plot_per_class_psd_roc(tpr_vs_efpr, psds_eval.class_names,
                           title="Per-class TPR-vs-eFPR PSDROC",
                           xlabel="eFPR")

    # Recover some of the operating points for each class based on criteria
    # on TPR, FPR, eFPR and f-score
    class_constraints = list()
    # find the op. point with minimum eFPR and TPR >= 0.6 for the first class
    class_constraints.append({"class_name": psds_eval.class_names[0],
                              "constraint": "tpr",
                              "value": 0.6})
    # find the op. point with maximum TPR and FPR <= 50 for the second class
    class_constraints.append({"class_name": psds_eval.class_names[1],
                              "constraint": "fpr",
                              "value": 50})
    # find the op. point with maximum TPR and eFPR <= 50 for the third class
    class_constraints.append({"class_name": psds_eval.class_names[2],
                              "constraint": "efpr",
                              "value": 50})
    # find the op. point with maximum f1-score for the fourth class
    class_constraints.append({"class_name": psds_eval.class_names[3],
                              "constraint": "fscore",
                              "value": None})
    class_constraints_table = pd.DataFrame(class_constraints)
    selected_ops = psds_eval.select_operating_points_per_class(
        class_constraints_table, alpha_ct=1., beta=1.)

    for k in range(len(class_constraints)):
        print(f"For class {class_constraints_table.class_name[k]}, the best "
              f"op. point with {class_constraints_table.constraint[k]} ~ "
              f"{class_constraints_table.value[k]}:")
        print(f"\tProbability Threshold: {selected_ops.threshold[k]}, "
              f"TPR: {selected_ops.TPR[k]:.2f}, "
              f"FPR: {selected_ops.FPR[k]:.2f}, "
              f"eFPR: {selected_ops.eFPR[k]:.2f}, "
              f"F1-score: {selected_ops.Fscore[k]:.2f}")

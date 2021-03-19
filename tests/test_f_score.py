"""Test to verify that the f_score function is behaving correctly"""
from psds_eval import PSDSEval, PSDSEvalError
import pandas as pd
import numpy as np
from os.path import join, dirname, abspath
import pytest

DATADIR = join(dirname(abspath(__file__)), "data")


def read_gt_and_det():
    """Return detections and ground truth pandas tables"""
    det = pd.read_csv(join(DATADIR, "baseline_validation_AA_0.005.csv"),
                      sep="\t")
    gt = pd.read_csv(join(DATADIR, "baseline_validation_gt.csv"),
                     sep="\t")
    return det, gt


@pytest.fixture(scope="session")
def metadata():
    """A function that provides test audio metadata to each test"""
    return pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")


def test_compute_f_score(metadata):
    """Test the f_score is computed correctly"""
    det_t, gt_t = read_gt_and_det()
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt_t,
                         metadata=metadata)
    f_avg, per_class_f = psds_eval.compute_macro_f_score(det_t)
    expected_class_f = [0.7752161383285303, 0.7468354430379747,
                        0.548936170212766, 0.39943342776203966,
                        0.6548881036513545, 0.7663551401869159,
                        0.9405405405405406, 0.6978021978021978,
                        0.7105553512320706, 0.8427672955974843]
    assert f_avg == pytest.approx(0.7083329808351875), \
        "The average F-score was incorrect"
    for exp_f, class_f in zip(expected_class_f, per_class_f.values()):
        assert exp_f == pytest.approx(class_f), "Per-class F-score incorrect"


def test_compute_f_score_no_gt():
    """Test PSDSEvalError raised if gt is missing"""
    det_t, _ = read_gt_and_det()
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3)
    with pytest.raises(PSDSEvalError, match="Ground Truth must be provided"):
        psds_eval.compute_macro_f_score(det_t)


def test_compute_f_score_gt_later(metadata):
    """Test computation is correct when gt is not passed at init time"""
    det_t, gt_t = read_gt_and_det()
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3)
    psds_eval.set_ground_truth(gt_t, metadata)
    f_avg, per_class_f = psds_eval.compute_macro_f_score(det_t)
    expected_class_f = [0.7752161383285303, 0.7468354430379747,
                        0.548936170212766, 0.39943342776203966,
                        0.6548881036513545, 0.7663551401869159,
                        0.9405405405405406, 0.6978021978021978,
                        0.7105553512320706, 0.8427672955974843]
    assert f_avg == pytest.approx(0.7083329808351875), \
        "The average F-score was incorrect"
    for exp_f, class_f in zip(expected_class_f, per_class_f.values()):
        assert exp_f == pytest.approx(class_f), "Per-class F-score incorrect"


def test_compute_f_score_no_det(metadata):
    det_t, gt_t = read_gt_and_det()
    det_t = pd.DataFrame(columns=det_t.columns)
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt_t,
                         metadata=metadata)
    f_avg, per_class_f = psds_eval.compute_macro_f_score(det_t)
    per_class_f_array = np.fromiter(per_class_f.values(), dtype=float)
    assert np.isnan(f_avg),  "The average F-score was incorrect"
    assert np.all(np.isnan(per_class_f_array)), "Per-class F-score incorrect"

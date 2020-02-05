"""This test file runs full PSDS calculations using example data."""
import pytest
from os.path import (join, dirname, abspath)
import pandas as pd
import numpy as np
from psds_eval import PSDSEval


DATADIR = join(dirname(abspath(__file__)), "data")


@pytest.fixture(scope="session")
def metadata():
    """A function that provides test audio metadata to each test"""
    return pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")


def test_example_1_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    tpr = np.array([1., 1., 1.])
    fpr = np.array([12.857143, 12.857143, 0.])
    ctr = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [720., 0., 0.]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.9142857142857143), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_2_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_2.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_2.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    exp_counts = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ])
    tpr = np.array([0., 0., 1.])
    fpr = np.array([12.857143, 0., 12.857143])
    ctr = np.array([
        [0., 0., 240.],
        [144., 0., 144.],
        [0., 0., 0.]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.29047619047619044), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_3_paper_icassp(metadata):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_3.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_3.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    tpr = np.array([1., 1., 0.])
    fpr = np.array([12.857143, 0., 12.857143])
    ctr = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [600., 0., 0.]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.6238095238095237), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_example_4(metadata):
    """Run PSDSEval on some sample data and ensure the results are correct"""
    det = pd.read_csv(join(DATADIR, "test_4.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_4.gt"), sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [2, 0, 0, 1, 0],
        [0, 0, 0, 1, 3],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ])
    tpr = np.array([1., 0., 1., 0.])
    fpr = np.array([0, 38.57142857, 0, 25.71428571])
    ctr = np.array([
        [0, 0, 0, 87.80487805],
        [0, 0, 0, 300],
        [0, 156.52173913, 0, 0],
        [0, 0, 0, 0]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.5), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_det_on_file_no_gt():
    """Ensure that the psds metric is correct when there is no ground truth"""
    det = pd.DataFrame({"filename": ["test.wav"], "onset": [2.4],
                        "offset": [5.9], "event_label": ["c1"]})
    gt = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
    metadata = pd.DataFrame({"filename": ["test.wav"], "duration": [10.0]})
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(class_names=['c1'], ground_truth=gt,
                         metadata=metadata)
    exp_counts = np.array([
        [0, 1],
        [0, 0]
    ])
    tpr = np.array([0.])
    fpr = np.array([360.])
    ctr = np.array([[0.]])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_two_operating_points_one_with_no_detections():
    """Tests a case where the dtc and gtc df's are empty for the second op"""
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(ground_truth=gt, metadata=metadata)
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(join(DATADIR, "test_4.det"), sep="\t")
    psds_eval.add_operating_point(det)
    psds_eval.add_operating_point(det2)
    assert psds_eval.psds(0.0, 0.0, 100.0).value == \
        pytest.approx(0.9142857142857143), \
        "PSDS value was calculated incorrectly"


def test_two_operating_points_second_has_filtered_out_gtc():
    """Tests a case where the gt coverage df becomes empty for the second op"""
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(1, 1, 1, ground_truth=gt, metadata=metadata)
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(join(DATADIR, "test_1a.det"), sep="\t")
    psds_eval.add_operating_point(det)
    psds_eval.add_operating_point(det2)
    assert psds_eval.psds(0.0, 0.0, 100.0).value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"


def test_empty_det():
    """Run the PSDSEval class with tables that contain no detections"""
    gt = pd.DataFrame({"filename": ["test.wav"], "onset": [2.4],
                       "offset": [5.9], "event_label": ["c1"]})
    det = pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])
    metadata = pd.DataFrame({"filename": ["test.wav"], "duration": [10.0]})
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(class_names=['c1'], metadata=metadata,
                         ground_truth=gt)
    exp_counts = np.array([
        [0, 0],
        [0, 0]
    ])
    tpr = np.array([0.])
    fpr = np.array([0.])
    ctr = np.array([[0.]])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.0), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_files_from_dcase(metadata):
    """Run PSDSEval on some example data from DCASE"""
    det = pd.read_csv(join(DATADIR, "Y23R6_ppquxs_247.000_257000.det"),
                      sep="\t")
    gt = pd.read_csv(join(DATADIR, "Y23R6_ppquxs_247.000_257000.gt"),
                     sep="\t")
    # Record the checksums of the incoming data
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [1., 0., 1.],
        [1., 4., 0.],
        [0., 0., 0.]
    ])
    tpr = np.array([0.25, 1.])
    fpr = np.array([12.857143,  0.])
    ctr = np.array([
       [0., 0.],
       [600.40026684, 0.]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds1.value == pytest.approx(0.6089285714285714), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_full_dcase_validset():
    """Run PSDSEval on all the example data from DCASE"""
    det = pd.read_csv(join(DATADIR, "baseline_validation_AA_0.005.csv"),
                      sep="\t")
    gt = pd.read_csv(join(DATADIR, "baseline_validation_gt.csv"),
                     sep="\t")
    metadata = pd.read_csv(join(DATADIR, "baseline_validation_metadata.csv"),
                           sep="\t")
    # Record the checksums of the incoming data
    meta_hash = pd.util.hash_pandas_object(metadata).values
    gt_hash = pd.util.hash_pandas_object(gt).values
    det_hash = pd.util.hash_pandas_object(det).values

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    # matrix (n_class, n_class) last col/row is world (for FP)
    exp_counts = np.array([
        [269, 9, 63, 41, 120, 13, 7, 18, 128, 2, 302],
        [5, 59, 4, 45, 29, 31, 35, 46, 86, 58, 416],
        [54, 17, 129, 19, 105, 13, 14, 16, 82, 20, 585],
        [37, 43, 8, 164, 56, 9, 63, 63, 87, 7, 1100],
        [45, 10, 79, 73, 278, 7, 24, 51, 154, 22, 1480],
        [14, 22, 11, 24, 30, 41, 51, 26, 62, 43, 386],
        [3, 20, 12, 136, 96, 35, 87, 103, 97, 27, 840],
        [8, 41, 13, 119, 93, 48, 135, 127, 185, 32, 662],
        [89, 120, 74, 493, 825, 203, 403, 187, 966, 89, 1340],
        [0, 83, 1, 12, 58, 27, 46, 46, 120, 67, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    tpr = np.array([0.64047619, 0.61458333, 0.37829912, 0.28924162, 0.4877193,
                    0.63076923, 0.92553191, 0.53586498, 0.55074116,
                    0.72826087])
    fpr = np.array([93.08219178, 128.21917808, 180.30821918, 339.04109589,
                    456.16438356, 118.97260274, 258.90410959, 204.04109589,
                    413.01369863, 120.20547945])
    ctr = np.array([
        [0., 39.38051054, 275.66357376, 179.40010356, 525.07347382,
         56.88295966, 30.62928597, 78.76102107, 560.07837208, 8.75122456],
        [36.54377132, 0, 29.23501705,  328.89394185, 211.95387364,
         226.57138217, 255.80639922, 336.20269612, 628.55286666, 423.90774728],
        [412.06956854, 129.72560491, 0, 144.98744078, 801.24638326,
         99.20193317, 106.8328511, 122.09468697, 625.73527074, 152.61835872],
        [375.77227974, 436.70832511, 81.24806048, 0, 568.73642339, 91.40406805,
         639.82847632, 639.82847632, 883.57265777, 71.09205292],
        [201.60236547, 44.80052566, 353.92415271, 327.04383731, 0, 31.36036796,
         107.52126158, 228.48268086, 689.92809516, 98.56115645],
        [100.2921207, 157.60190396, 78.80095198, 171.92934977, 214.91168722, 0,
         365.34986827, 186.25679559, 444.15082025, 308.04008501],
        [13.91555321, 92.77035472, 55.66221283, 630.83841208, 445.29770265,
         162.34812076, 0, 477.7673268, 449.93622038, 125.23997887],
        [23.16073227, 118.69875286, 37.63618993, 344.51589244, 269.24351258,
         138.96439359, 390.83735697, 0, 535.59193363, 92.64292906],
        [122.13847545, 164.68109049, 101.55333914, 676.56481345, 1132.18249714,
         278.58551142,  553.05399557, 256.62803269, 0., 122.13847545],
        [0, 382.88155531, 4.61303079, 55.35636944, 267.55578564, 124.55183125,
         212.1994162, 212.1994162, 553.56369442, 0]
    ])
    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts)
    np.testing.assert_allclose(psds_eval.operating_points.tpr[0], tpr)
    np.testing.assert_allclose(psds_eval.operating_points.fpr[0], fpr)
    np.testing.assert_allclose(psds_eval.operating_points.ctr[0], ctr)
    psds1 = psds_eval.psds(0.0, 0.0, 100.0)
    # Check that all the psds metrics match
    assert psds1.value == pytest.approx(0.0044306914546640595), \
        "PSDS value was calculated incorrectly"
    # Check that the data has not been messed about with
    assert np.all(pd.util.hash_pandas_object(gt).values == gt_hash)
    assert np.all(pd.util.hash_pandas_object(metadata).values == meta_hash)
    assert np.all(pd.util.hash_pandas_object(det).values == det_hash)


def test_multi_ops_multiple_times_sequentially():
    """Run 3 times test_full_psds using delete_ops to reset the metric"""
    gt = pd.read_csv(join(DATADIR, "baseline_validation_gt.csv"),
                     sep="\t")
    metadata = pd.read_csv(join(DATADIR, "baseline_validation_metadata.csv"),
                           sep="\t")
    dets = []
    dets.append(pd.read_csv(join(DATADIR, "baseline_validation_AA_0.005.csv"),
                            sep="\t"))
    for k in range(5):
        dets.append(dets[0].sample(4500, random_state=7*k))
        print(dets[k+1])

    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    ref_psds_value = 0.07211376135412327
    for k in range(3):
        for det_t in dets:
            psds_eval.add_operating_point(det_t)
        psds = psds_eval.psds(0.0, 0.0, 100)
        assert psds.value == pytest.approx(ref_psds_value), \
            "PSDS was calculated incorrectly"
        psds_eval.clear_all_operating_points()
        assert psds_eval.num_operating_points() == 0
        assert len(psds_eval.operating_points) == 0


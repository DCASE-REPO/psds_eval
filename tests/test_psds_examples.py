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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    tpr = np.array([1., 1., 1.])
    fpr = np.array([12.857143, 12.857143, 0.])
    ctr = np.array([
        [0., 0., 720.],
        [0., 0.,   0.],
        [0., 0.,   0.]
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0]
    ])
    tpr = np.array([0., 0., 1.])
    fpr = np.array([12.857143, 0., 12.857143])
    ctr = np.array([
        [  0., 144., 0.],
        [  0.,   0., 0.],
        [240., 144., 0.]
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    tpr = np.array([1., 1., 0.])
    fpr = np.array([12.857143, 0., 12.857143])
    ctr = np.array([
        [0., 0., 600.],
        [0., 0.,   0.],
        [0., 0.,   0.]
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [2, 0, 0, 0, 0],
        [0, 0, 1, 0, 3],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 2],
        [0, 0, 0, 0, 0]
    ])
    tpr = np.array([1., 0., 1., 0.])
    fpr = np.array([0., 38.57142857, 0., 25.71428571])
    ctr = np.array([
        [0., 0., 0., 0.],
        [0., 0., 156.521739, 0.],
        [0., 0., 0., 0.],
        [87.804878, 300., 0., 0.]

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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [1., 1., 1.],
        [0., 4., 0.],
        [0., 0., 0.]
    ])
    tpr = np.array([0.25, 1.])
    fpr = np.array([12.857143,  0.])
    ctr = np.array([
       [0., 600.40026684],
       [0., 0.]
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
    # matrix (n_class, n_class): axis 0 = gt, axis 1 = det
    exp_counts = np.array([
        [269, 5, 54, 37, 45, 14, 3, 8, 89, 0, 302],
        [9, 59, 17, 43, 10, 22, 20, 41, 120, 83, 416],
        [63, 4, 129, 8, 79, 11, 12, 13, 74, 1, 585],
        [41, 45, 19, 164, 73, 24, 136, 119, 493, 12, 1100],
        [120, 29, 105, 56, 278, 30, 96, 93, 825, 58, 1480],
        [13, 31, 13, 9, 7, 41, 35, 48, 203, 27, 386],
        [7, 35, 14, 63, 24, 51, 87, 135, 403, 46, 840],
        [18, 46, 16, 63, 51, 26, 103, 127, 187, 46, 662],
        [128, 86, 82, 87, 154, 62, 97, 185, 966, 120, 1340],
        [2, 58, 20, 7, 22, 43, 27, 32, 89, 67, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    tpr = np.array([0.64047619, 0.61458333, 0.37829912, 0.28924162, 0.4877193,
                    0.63076923, 0.92553191, 0.53586498, 0.55074116,
                    0.72826087])
    fpr = np.array([93.08219178, 128.21917808, 180.30821918, 339.04109589,
                    456.16438356, 118.97260274, 258.90410959, 204.04109589,
                    413.01369863, 120.20547945])
    ctr = np.array([
        [0., 36.54377132, 412.06956854, 375.77227974, 201.60236547,
         100.2921207, 13.91555321, 23.16073227, 122.13847545, 0.],
        [39.38051054, 0., 129.72560491, 436.70832511, 44.80052566,
         157.60190396, 92.77035472, 118.69875286, 164.68109049, 382.88155531],
        [275.66357376, 29.23501705, 0., 81.24806048, 353.92415271,
         78.80095198, 55.66221283, 37.63618993, 101.55333914, 4.61303079],
        [179.40010356, 328.89394185, 144.98744078, 0., 327.04383731,
         171.92934977, 630.83841208, 344.51589244, 676.56481345, 55.35636944],
        [525.07347382, 211.95387364, 801.24638326, 568.73642339, 0.,
         214.91168722, 445.29770265, 269.24351258, 1132.18249714, 267.55578564],
        [56.88295966, 226.57138217, 99.20193317, 91.40406805, 31.36036796,
         0., 162.34812076, 138.96439359, 278.58551142, 124.55183125],
        [30.62928597, 255.80639922, 106.8328511, 639.82847632, 107.52126158,
         365.34986827, 0., 390.83735697, 553.05399557, 212.1994162],
        [78.76102107, 336.20269612, 122.09468697, 639.82847632, 228.48268086,
         186.25679559, 477.7673268, 0., 256.62803269, 212.1994162],
        [560.07837208, 628.55286666, 625.73527074, 883.57265777, 689.92809516,
         444.15082025, 449.93622038, 535.59193363, 0., 553.56369442],
        [8.75122456, 423.90774728, 152.61835872, 71.09205292, 98.56115645,
         308.04008501, 125.23997887, 92.64292906, 122.13847545, 0.]
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


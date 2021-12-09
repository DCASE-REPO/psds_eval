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


def create_repeated_indexes_in_dataframe(df):
    """Create indexes with repeated values in DataFrame"""
    df["index"] = np.arange(df.shape[0]).astype(int)
    df.loc[0, "index"] = df.shape[0]
    df.loc[1, "index"] = 2
    df = df.set_index("index")
    return df


@pytest.mark.parametrize("invalid_indexes", [False, True])
def test_example_1_paper_icassp(metadata, invalid_indexes):
    """Run PSDSEval on some sample data from the ICASSP paper"""
    det = pd.read_csv(join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(join(DATADIR, "test_1.gt"), sep="\t")
    # Record the checksums of the incoming data
    if invalid_indexes:
        det = create_repeated_indexes_in_dataframe(det)
        gt = create_repeated_indexes_in_dataframe(gt)
        metadata = create_repeated_indexes_in_dataframe(metadata)
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
        [np.nan, 0., 720.],
        [0., np.nan,   0.],
        [0., 0.,   np.nan]
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
        [np.nan, 144., 0.],
        [0., np.nan, 0.],
        [240., 144., np.nan]
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
        [np.nan, 0., 600.],
        [0., np.nan, 0.],
        [0., 0., np.nan]
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
        [np.nan, 0., 0., 0.],
        [0., np.nan, 156.521739, 0.],
        [0., 0., np.nan, 0.],
        [87.804878, 300., 0., np.nan]

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
    ctr = np.array([[np.nan]])
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
    ctr = np.array([[np.nan]])
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
       [np.nan, 600.40026684],
       [0., np.nan]
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
        [41, 45, 19, 141, 73, 24, 139, 119, 495, 12, 1103],
        [120, 29, 105, 56, 278, 30, 96, 93, 825, 58, 1480],
        [13, 31, 13, 9, 7, 41, 35, 48, 203, 27, 386],
        [7, 35, 14, 63, 24, 51, 87, 135, 403, 46, 840],
        [18, 46, 16, 63, 51, 26, 103, 127, 187, 46, 662],
        [128, 86, 82, 87, 154, 62, 97, 185, 966, 120, 1340],
        [2, 58, 20, 7, 22, 43, 27, 32, 89, 67, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    tpr = np.array([
        0.64047619, 0.62105263, 0.37829912, 0.25044405, 0.4877193, 0.63076923,
        0.92553191, 0.53586498, 0.55105533, 0.72826087
    ])
    fpr = np.array([
        93.08219178, 128.21917808, 180.30821918, 339.96575342, 456.16438356,
        118.97260274, 258.90410959, 204.04109589, 413.01369863, 120.20547945
    ])
    ctr = np.array([
        [np.nan, 36.63950594, 412.06956854, 378.4295175, 201.60236547,
         100.2921207, 13.91555321,   23.16073227, 122.18868771, 0.],
        [39.38051054, np.nan, 129.72560491, 439.79646629, 44.80052566,
         157.60190396,  92.77035472, 118.69875286, 164.74879241, 382.88155531],
        [ 275.66357376,  29.31160475,  np.nan,  81.82259838, 353.92415271,
          78.80095198,  55.66221283,  37.63618993, 101.59508866, 4.61303079],
        [179.40010356, 329.75555343, 144.98744078,  np.nan, 327.04383731,
         171.92934977, 644.75396529, 344.51589244, 679.58876871,  55.35636944],
        [525.07347382, 212.50913443, 801.24638326, 572.75818865, np.nan,
         214.91168722, 445.29770265, 269.24351258, 1132.64794785,
         267.55578564],
        [56.88295966, 227.16493681,  99.20193317,  92.05042318, 31.36036796,
         np.nan, 162.34812076, 138.96439359, 278.7000405, 124.55183125],
        [30.62928597, 256.47654156, 106.8328511 , 644.35296223,  107.52126158,
         365.34986827,  np.nan, 390.83735697, 553.28136119, 212.1994162 ],
        [78.76102107, 337.08345462, 122.09468697, 644.35296223,  228.48268086,
         186.25679559, 477.7673268 ,  np.nan, 256.73353485, 212.1994162 ],
        [560.07837208, 630.19950211, 625.73527074, 889.82075737,  689.92809516,
         444.15082025, 449.93622038, 535.59193363, np.nan, 553.56369442],
        [8.75122456, 425.01826886, 152.61835872, 71.59477358, 98.56115645,
         308.04008501,  125.23997887,   92.64292906, 122.18868771, np.nan]]
    )
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
    ref_psds_value = 0.07224283564515908
    for k in range(3):
        for det_t in dets:
            psds_eval.add_operating_point(det_t)
        psds = psds_eval.psds(0.0, 0.0, 100)
        assert psds.value == pytest.approx(ref_psds_value), \
            "PSDS was calculated incorrectly"
        psds_eval.clear_all_operating_points()
        assert psds_eval.num_operating_points() == 0
        assert len(psds_eval.operating_points) == 0

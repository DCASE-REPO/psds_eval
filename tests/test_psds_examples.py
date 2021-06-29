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
        [0., 144., 0.],
        [0., 0., 0.],
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
        [41, 45, 19, 141, 73, 24, 139, 119, 495, 12, 1103],
        [120, 29, 105, 56, 278, 30, 96, 93, 825, 58, 1480],
        [13, 31, 13, 9, 7, 41, 35, 48, 203, 27, 386],
        [7, 35, 14, 63, 24, 51, 87, 135, 403, 46, 840],
        [18, 46, 16, 63, 51, 26, 103, 127, 187, 46, 662],
        [128, 86, 82, 87, 154, 62, 97, 185, 966, 120, 1340],
        [2, 58, 20, 7, 22, 43, 27, 32, 89, 67, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    tpr = np.array([0.64047619, 0.62105263, 0.37829912, 0.25044405,
                    0.4877193, 0.63076923, 0.92553191, 0.53586498,
                    0.55105533, 0.72826087])
    fpr = np.array([93.08219178, 128.21917808, 180.30821918, 339.96575342,
                    456.16438356, 118.97260274, 258.90410959, 204.04109589,
                    413.01369863, 120.20547945])
    ctr = np.array([
        [0.000000000000000000e+00, 3.663950593661773070e+01,
         4.120695685351820998e+02, 3.784295175023650586e+02,
         2.016023654677545949e+02, 1.002921207007712923e+02,
         1.391555320766386217e+01, 2.316073226515178618e+01,
         1.221886877072230249e+02, 0.000000000000000000e+00],
        [3.938051053671746615e+01, 0.000000000000000000e+00,
         1.297256049092239891e+02, 4.397964662865323930e+02,
         4.480052565950101950e+01, 1.576019039583548818e+02,
         9.277035471775907638e+01, 1.186987528589028926e+02,
         1.647487924142332929e+02, 3.828815553089577293e+02],
        [2.756635737570222773e+02, 2.931160474929418669e+01,
         0.000000000000000000e+00, 8.182259837888975085e+01,
         3.539241527100580811e+02, 7.880095197917744088e+01,
         5.566221283065544867e+01, 3.763618993087165165e+01,
         1.015950886554438597e+02, 4.613030786854912080e+00],
        [1.794001035561573474e+02, 3.297555534295595976e+02,
         1.449874407808974013e+02, 0.000000000000000000e+00,
         3.270438373143574609e+02, 1.719293497727507827e+02,
         6.447539652884255474e+02, 3.445158924441328168e+02,
         6.795887687087123368e+02, 5.535636944225894496e+01],
        [5.250734738228995866e+02, 2.125091344323828366e+02,
         8.012463832628541240e+02, 5.727581886522282275e+02,
         0.000000000000000000e+00, 2.149116872159384570e+02,
         4.452977026452435894e+02, 2.692435125823894850e+02,
         1.132647947847853857e+03, 2.675557856375849042e+02],
        [5.688295966414744953e+01, 2.271649368070299317e+02,
         9.920193316587717902e+01, 9.205042317625095905e+01,
         3.136036796165071294e+01, 0.000000000000000000e+00,
         1.623481207560784014e+02, 1.389643935909107029e+02,
         2.787000405007446489e+02, 1.245518312450826386e+02],
        [3.062928597300247446e+01, 2.564765415563241504e+02,
         1.068328511017138851e+02, 6.443529622337567844e+02,
         1.075212615828024525e+02, 3.653498682670954167e+02,
         0.000000000000000000e+00, 3.908373569744363749e+02,
         5.532813611911334419e+02, 2.121994161953259663e+02],
        [7.876102107343493230e+01, 3.370834546168831594e+02,
         1.220946869733872973e+02, 6.443529622337567844e+02,
         2.284826808634552151e+02, 1.862567955871466836e+02,
         4.777673267964592583e+02, 0.000000000000000000e+00,
         2.567335348455135318e+02, 2.121994161953259663e+02],
        [5.600783720777594681e+02, 6.301995021098249481e+02,
         6.257352707386098700e+02, 8.898207573704260085e+02,
         6.899280951563157487e+02, 4.441508202462728150e+02,
         4.499362203811315339e+02, 5.355919336316350154e+02,
         0.000000000000000000e+00, 5.535636944225894922e+02],
        [8.751224563714991689e+00, 4.250182688647656732e+02,
         1.526183587167341216e+02, 7.159477358152852844e+01,
         9.856115645090224575e+01, 3.080400850095118130e+02,
         1.252399788689747595e+02, 9.264292906060714472e+01,
         1.221886877072230249e+02, 0.000000000000000000e+00]
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

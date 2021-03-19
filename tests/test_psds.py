"""Test the user facing functions in the PSDSEval module"""
import os
import numpy as np
import pytest
import pandas as pd
from psds_eval import PSDSEval, PSDSEvalError

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.mark.parametrize("x", [-1, 1.2, -5, float("-inf")])
def test_invalid_thresholds(x):
    """Ensure a PSDSEvalError is raised when thresholds are invalid"""
    with pytest.raises(PSDSEvalError, match="dtc_threshold"):
        PSDSEval(dtc_threshold=x)
    with pytest.raises(PSDSEvalError, match="cttc_threshold"):
        PSDSEval(cttc_threshold=x)
    with pytest.raises(PSDSEvalError, match="gtc_threshold"):
        PSDSEval(gtc_threshold=x)


@pytest.mark.parametrize("x", [1, 0, 0.0, 0.1, 0.6])
def test_valid_thresholds(x):
    """Test the PSDSEval with a range of valid threshold values"""
    assert PSDSEval(dtc_threshold=x)
    assert PSDSEval(cttc_threshold=x)
    assert PSDSEval(gtc_threshold=x)


def test_negative_alpha_st():
    """Ensure a PSDSEvalError is raised if alpha_st is negative"""
    with pytest.raises(PSDSEvalError, match="alpha_st can't be negative"):
        PSDSEval().psds(0.0, -1.0, 100)


def tests_num_operating_points_without_any_operating_points():
    """Ensures that the eval class has no operating points when initialised"""
    psds_eval = PSDSEval()
    assert psds_eval.num_operating_points() == 0


def test_eval_class_with_no_ground_truth():
    """Ensure that PSDSEval raises a PSDSEvalError when GT is None"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    with pytest.raises(PSDSEvalError,
                       match="The ground truth cannot be set without data"):
        PSDSEval(metadata=metadata, ground_truth=None)


def test_eval_class_with_no_metadata():
    """Ensure that PSDSEval raises a PSDSEvalError when metadata is None"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    with pytest.raises(PSDSEvalError,
                       match="Audio metadata is required"):
        PSDSEval(metadata=None, ground_truth=gt)


def test_set_ground_truth_with_no_ground_truth():
    """set_ground_truth() must raise a PSDSEvalError when GT is None"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError,
                       match="The ground truth cannot be set without data"):
        psds_eval.set_ground_truth(None, metadata)


def test_set_ground_truth_with_no_metadata():
    """set_ground_truth() must raise a PSDSEvalError with None metadata"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="Audio metadata is required"):
        psds_eval.set_ground_truth(gt, None)


def test_psds_with_empty_metadata():
    """Check that an error is raised when empty metadata is provided"""
    metadata = pd.read_csv(os.path.join(DATADIR, "empty.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    with pytest.raises(PSDSEvalError,
                       match="The metadata dataframe provided is empty"):
        _ = PSDSEval(ground_truth=gt, metadata=metadata)


def test_psds_with_empty_ground_truth():
    """Prove that empty ground truth does not raise an error"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "empty.gt"), sep="\t")
    psds_eval = PSDSEval(ground_truth=gt, metadata=metadata)
    assert isinstance(psds_eval.ground_truth, pd.DataFrame) is True
    assert psds_eval.ground_truth.empty is False


def test_setting_ground_truth_more_than_once():
    """Ensure that an error is raised when the ground truth is set twice"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)

    with pytest.raises(PSDSEvalError, match="You cannot set the ground truth "
                                            "more than once per evaluation"):
        psds_eval.set_ground_truth(gt_t=gt, meta_t=metadata)


BAD_GT_DATA = [[], (0.12, 8), float("-inf"), {"gt": [7, 2]}]


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_ground_truth(bad_data):
    """Setting the ground truth with invalid data must raise an error"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="The ground truth data must be "
                                            "provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(bad_data, metadata)


@pytest.mark.parametrize("bad_data", BAD_GT_DATA)
def test_set_ground_truth_with_bad_metadata(bad_data):
    """Setting the ground truth with invalid metadata must raise an error"""
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval()
    with pytest.raises(PSDSEvalError, match="The metadata data must be "
                                            "provided in a pandas.DataFrame"):
        psds_eval.set_ground_truth(gt, bad_data)


@pytest.mark.parametrize("table_name,raise_error",
                         [("test_1_overlap.gt", True),
                          ("test_1_nonoverlap.gt", False)])
def test_set_ground_truth_with_overlapping_events(table_name, raise_error):
    """Gronud truth with overlapping events must raise an error"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, table_name), sep="\t")
    psds_eval = PSDSEval()
    if raise_error:
        with pytest.raises(
                PSDSEvalError,
                match="The ground truth dataframe provided has intersecting "
                      "events/labels for the same class."):
            psds_eval.set_ground_truth(gt, metadata)
    else:
        psds_eval.set_ground_truth(gt, metadata)
        assert isinstance(psds_eval.ground_truth, pd.DataFrame) is True


def test_add_operating_point_with_no_metadata():
    """Ensure that add_operating_point raises an error when metadata is none"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    psds_eval = PSDSEval(metadata=None, ground_truth=None)
    with pytest.raises(PSDSEvalError,
                       match="Ground Truth must be provided "
                             "before adding the first operating point"):
        psds_eval.add_operating_point(det)


def test_add_operating_point_with_wrong_data_format():
    """Ensure add_operating_point raises an error when the input is not a
    pandas table"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t").to_numpy()
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    with pytest.raises(PSDSEvalError,
                       match="The detection data must be provided "
                             "in a pandas.DataFrame"):
        psds_eval.add_operating_point(det)


def test_add_operating_point_with_empty_dataframe():
    """Ensure add_operating_point raises an error when given an
    incorrect table"""
    det = pd.DataFrame()
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    with pytest.raises(PSDSEvalError,
                       match="The detection data columns need to "
                             "match the following"):
        psds_eval.add_operating_point(det)


def test_add_operating_point_with_zero_detections():
    """An error must not be raised when there are no detections"""
    det = pd.read_csv(os.path.join(DATADIR, "empty.det"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    psds_eval.add_operating_point(det)
    assert psds_eval.num_operating_points() == 1
    assert psds_eval.operating_points["id"][0] == \
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.mark.parametrize("table_name,raise_error",
                         [("test_1_overlap.det", True),
                          ("test_1_nonoverlap.det", False)])
def test_add_operating_points_with_overlapping_events(table_name, raise_error):
    """Detections with overlapping events must raise an error"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det = pd.read_csv(os.path.join(DATADIR, table_name), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    if raise_error:
        with pytest.raises(
                PSDSEvalError,
                match="The detection dataframe provided has intersecting "
                      "events/labels for the same class."):
            psds_eval.add_operating_point(det)
    else:
        psds_eval.add_operating_point(det)
        assert psds_eval.num_operating_points() == 1


def test_that_add_operating_point_added_a_point():
    """Ensure add_operating_point adds an operating point correctly"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    psds_eval.add_operating_point(det)
    assert psds_eval.num_operating_points() == 1
    assert psds_eval.operating_points["id"][0] == \
        "423089ce6d6554174881f69f9d0e57a8be9f5bc682dfce301462a8753aa6ec5f"


def test_adding_shuffled_operating_points():
    """Avoid the addition of the same operating point after shuffling"""
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(metadata=metadata, ground_truth=gt)
    psds_eval.add_operating_point(det)
    det_shuffled = det.copy(deep=True)
    det_shuffled = det_shuffled.sample(frac=1.).reset_index(drop=True)
    psds_eval.add_operating_point(det_shuffled)
    det_shuffled2 = det.copy(deep=True)
    det_shuffled2 = det_shuffled2[["onset", "event_label", "offset",
                                   "filename"]]
    psds_eval.add_operating_point(det_shuffled2)
    assert psds_eval.num_operating_points() == 1
    assert psds_eval.operating_points["id"][0] == \
        "423089ce6d6554174881f69f9d0e57a8be9f5bc682dfce301462a8753aa6ec5f"


def test_full_psds():
    """Run a full example of the PSDSEval and test the result"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
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

    psds_eval.add_operating_point(det)
    assert np.all(psds_eval.operating_points.counts[0] == exp_counts), \
        "Expected counts do not match"
    psds = psds_eval.psds(0.0, 0.0, 100.0)
    assert psds.value == pytest.approx(0.9142857142857143), \
        "PSDS was calculated incorrectly"


def test_delete_ops():
    """Perform deletion of ops"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    det_2 = pd.read_csv(os.path.join(DATADIR, "test_1a.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)

    assert psds_eval.operating_points.empty
    psds_eval.add_operating_point(det)
    psds_eval.add_operating_point(det_2)
    assert psds_eval.num_operating_points() == 2

    psds_eval.clear_all_operating_points()
    assert psds_eval.operating_points.empty


def test_add_operating_points_with_info():
    """Use info when adding operating points"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(os.path.join(DATADIR, "test_2.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"name": "test_1", "threshold1": 1}
    info2 = {"name": "test_2", "threshold2": 0}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    psds_eval.add_operating_point(det1, info=info1)
    psds_eval.add_operating_point(det2, info=info2)
    assert psds_eval.operating_points.name[0] == "test_1", \
        "The info name is not correctly reported."
    assert psds_eval.operating_points.name[1] == "test_2", \
        "The info name is not correctly reported."
    assert psds_eval.operating_points.threshold1[0] == 1, \
        "The info threshold1 is not correctly reported."
    assert psds_eval.operating_points.threshold2[1] == 0, \
        "The info threshold2 is not correctly reported."
    assert psds_eval.operating_points.threshold1.isna()[1], \
        "The info threshold1 is not correctly reported."
    assert psds_eval.operating_points.threshold2.isna()[0], \
        "The info threshold2 is not correctly reported."


def test_add_same_operating_point_with_different_info():
    """Check the use of conflicting info for the same operating point"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"name": "test_1", "threshold1": 1}
    info2 = {"name": "test_1_2", "threshold2": 0}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    psds_eval.add_operating_point(det1, info=info1)
    psds_eval.add_operating_point(det1, info=info2)
    assert psds_eval.num_operating_points() == 1
    assert psds_eval.operating_points.name[0] == "test_1", \
        "The info name is not correctly reported."
    assert psds_eval.operating_points.threshold1[0] == 1, \
        "The info threshold1 is not correctly reported."
    assert "threshold2" not in psds_eval.operating_points.columns, \
        "The info of ignored operating point modified the operating " \
        "points table."


def test_add_operating_point_with_info_using_column_names():
    """Check for non-permitted keys in the info"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"counts": 0, "threshold1": 1}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)

    with pytest.raises(PSDSEvalError,
                       match="the 'info' cannot contain the keys 'id', "
                             "'counts', 'tpr', 'fpr' or 'ctr'"):
        psds_eval.add_operating_point(det1, info=info1)


def test_retrieve_desired_operating_point():
    """Check if operating points can be found with requested constraints"""
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    det2 = pd.read_csv(os.path.join(DATADIR, "test_2.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"name": "test_1", "threshold1": 1}
    info2 = {"name": "test_2", "threshold2": 0}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    psds_eval.add_operating_point(det1, info=info1)
    psds_eval.add_operating_point(det2, info=info2)
    constraints = pd.DataFrame([
        {"class_name": "c1", "constraint": "tpr", "value": 1.},
        {"class_name": "c1", "constraint": "tpr", "value": 0.8},
        {"class_name": "c2", "constraint": "fpr", "value": 13.},
        {"class_name": "c3", "constraint": "efpr", "value": 240.},
        {"class_name": "c3", "constraint": "efpr", "value": 26.},
        {"class_name": "c1", "constraint": "fscore", "value": np.nan}])
    chosen_op_points = \
        psds_eval.select_operating_points_per_class(constraints, alpha_ct=1.,
                                                    beta=1.)
    assert chosen_op_points.name[0] == "test_1", \
        "Correct operating point is not chosen for tpr criteria with equality"
    assert chosen_op_points.name[1] == "test_1", \
        "Correct operating point is not chosen for tpr criteria with " \
        "inequality"
    assert chosen_op_points.name[2] == "test_1", \
        "Correct operating point is not chosen for fpr criteria with " \
        "inequality"
    assert chosen_op_points.name[3] == "test_1", \
        "Correct operating point is not chosen for efpr criteria with " \
        "equality"
    assert chosen_op_points.name[4] == "test_1", \
        "Correct operating point is not chosen for efpr criteria with " \
        "inequality"
    assert chosen_op_points.name[5] == "test_1", \
        "Correct operating point is not chosen for fscore criteria"
    assert chosen_op_points.Fscore[5] == pytest.approx(2./3.), \
        "Correct operating point is not chosen for fscore criteria"


def test_impossible_constraint_check():
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"name": "test_1", "threshold1": 1}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    psds_eval.add_operating_point(det1, info=info1)
    constraints = pd.DataFrame([
        {"class_name": "c2", "constraint": "fpr", "value": 11.},
        {"class_name": "c1", "constraint": "tpr", "value": 1.1}])
    chosen_op_points = \
        psds_eval.select_operating_points_per_class(constraints, alpha_ct=1.,
                                                    beta=1.)
    assert np.isnan(chosen_op_points.TPR[0]), \
        "NaN value is not returned for 0, 0 operating point"
    assert np.isnan(chosen_op_points.TPR[1]), \
        "NaN value is not returned for non-existing operating point"


def test_unknown_class_constraint_check():
    metadata = pd.read_csv(os.path.join(DATADIR, "test.metadata"), sep="\t")
    det1 = pd.read_csv(os.path.join(DATADIR, "test_1.det"), sep="\t")
    gt = pd.read_csv(os.path.join(DATADIR, "test_1.gt"), sep="\t")
    info1 = {"name": "test_1", "threshold1": 1}
    psds_eval = PSDSEval(dtc_threshold=0.5, gtc_threshold=0.5,
                         cttc_threshold=0.3, ground_truth=gt,
                         metadata=metadata)
    psds_eval.add_operating_point(det1, info=info1)
    constraints = pd.DataFrame([
        {"class_name": "class1", "constraint": "tpr", "value": 1.}])

    with pytest.raises(PSDSEvalError,
                       match="Unknown class: class1"):
        psds_eval.select_operating_points_per_class(constraints,
                                                    alpha_ct=1., beta=1.)

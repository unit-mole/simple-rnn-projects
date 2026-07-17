import numpy as np
from src.model_evaluation import classify_probabilities,evaluate_binary_classifier,select_threshold

def test_probability_thresholding():
    probabilities=np.array([.10,.25,.26,.90])
    assert classify_probabilities(probabilities,.255).tolist()==[0,0,1,1]

def test_metric_range():
    y_true=np.array([0,0,1,1])
    probabilities=np.array([.05,.2,.8,.95])
    metrics=evaluate_binary_classifier(y_true,probabilities,.5)
    assert metrics["accuracy"]==1.0
    assert 0<=metrics["pr_auc"]<=1

def test_validation_threshold_table():
    y_true=np.array([0,0,1,1])
    probabilities=np.array([.1,.3,.4,.8])
    threshold,table=select_threshold(y_true,probabilities,.2,.6,9)
    assert .2<=threshold<=.6
    assert len(table)==9

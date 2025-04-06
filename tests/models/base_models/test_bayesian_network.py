# Import necessary libraries
import pytest  # For writing and running tests
import numpy as np  # For array comparison
from pgmpy.models import DiscreteBayesianNetwork  # For creating the Bayesian Network
from pgmpy.factors.discrete import TabularCPD  # For defining Conditional Probability Distributions (CPDs)
from pgmpy.inference import VariableElimination  # For performing inference on the Bayesian Network
from models.base_models.bayesian_network import model, inference  # Import the model and inference object

# -------------------------------
# Test Suite for Bayesian Network
# -------------------------------

def test_bayesian_network_structure():
    """
    Test the structure of the Bayesian Network.

    Asserts:
        - The edges in the network match the expected structure.
    """
    expected_edges = [('Subject', 'Action'), ('Action', 'Location')]
    assert set(model.edges()) == set(expected_edges)

def test_bayesian_network_cpds():
    """
    Test the Conditional Probability Distributions (CPDs) in the Bayesian Network.

    Asserts:
        - The CPDs for 'Subject', 'Action', and 'Location' are correctly defined.
    """
    # Get the CPDs from the model
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}

    # Test the 'Subject' CPD
    assert cpds['Subject'].variable_card == 2
    assert np.array_equal(cpds['Subject'].values, [0.5, 0.5])  # Compare to a 1D array

    # Test the 'Action' CPD
    assert cpds['Action'].variable_card == 2
    assert np.array_equal(cpds['Action'].values, [[0.8, 0.3], [0.2, 0.7]])

    # Test the 'Location' CPD
    assert cpds['Location'].variable_card == 2
    assert np.array_equal(cpds['Location'].values, [[0.9, 0.1], [0.1, 0.9]])

def test_bayesian_network_inference():
    """
    Test inference on the Bayesian Network using Variable Elimination.

    Asserts:
        - The most probable 'Location' is correctly inferred given evidence for 'Subject'.
    """
    # Test inference for 'Subject' = 'Cat'
    result = inference.map_query(variables=['Location'], evidence={'Subject': 'Cat'})
    assert result['Location'] == 'On the Mat'

    # Test inference for 'Subject' = 'Dog'
    result = inference.map_query(variables=['Location'], evidence={'Subject': 'Dog'})
    assert result['Location'] == 'In the Park'

def test_bayesian_network_invalid_model():
    """
    Test the behavior when an invalid Bayesian Network is created.

    Asserts:
        - An exception is raised when the model is invalid.
    """
    # Create an invalid Bayesian Network (missing CPDs)
    invalid_model = DiscreteBayesianNetwork([('A', 'B')])

    with pytest.raises(ValueError, match="CPD associated with"):
        invalid_model.check_model()

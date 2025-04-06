# Import necessary libraries from pgmpy
from pgmpy.models import BayesianModel # For defining the structure of the Bayesian Network
from pgmpy.factors.discrete import TabularCPD  # For defining Conditional Probability Distributions (CPDs)
from pgmpy.inference import VariableElimination  # For performing inference on the Bayesian Network

# -------------------------------
# Define the Bayesian Network Structure
# -------------------------------

# Create a Bayesian Network with the following structure:
# 1. 'Subject' influences 'Action'
# 2. 'Action' influences 'Location'
# This structure represents the dependencies between the variables.
model = BayesianModel([
    ('Subject', 'Action'),  # Subject influences Action
    ('Action', 'Location')  # Action influences Location
])

# -------------------------------
# Define Conditional Probability Tables (CPTs)
# -------------------------------

# Define the CPD for the 'Subject' variable
# 'Subject' has two states: 'Cat' and 'Dog', each with an equal probability of 0.5.
cpd_subject = TabularCPD(
    variable='Subject',  # The variable being defined
    variable_card=2,  # Number of states for 'Subject'
    values=[[0.5], [0.5]],  # Probabilities for each state
    state_names={'Subject': ['Cat', 'Dog']}  # Names of the states
)

# Define the CPD for the 'Action' variable
# 'Action' depends on 'Subject' and has two states: 'Sits' and 'Runs'.
# The probabilities are defined as follows:
# - If 'Subject' is 'Cat', the probabilities are [0.8 (Sits), 0.2 (Runs)].
# - If 'Subject' is 'Dog', the probabilities are [0.3 (Sits), 0.7 (Runs)].
cpd_action = TabularCPD(
    variable='Action',  # The variable being defined
    variable_card=2,  # Number of states for 'Action'
    values=[
        [0.8, 0.3],  # Probabilities for 'Sits'
        [0.2, 0.7]   # Probabilities for 'Runs'
    ],
    evidence=['Subject'],  # The variable(s) that 'Action' depends on
    evidence_card=[2],  # Number of states for the evidence variable(s)
    state_names={
        'Action': ['Sits', 'Runs'],  # Names of the states for 'Action'
        'Subject': ['Cat', 'Dog']   # Names of the states for 'Subject'
    }
)

# Define the CPD for the 'Location' variable
# 'Location' depends on 'Action' and has two states: 'On the Mat' and 'In the Park'.
# The probabilities are defined as follows:
# - If 'Action' is 'Sits', the probabilities are [0.9 (On the Mat), 0.1 (In the Park)].
# - If 'Action' is 'Runs', the probabilities are [0.4 (On the Mat), 0.6 (In the Park)].
cpd_location = TabularCPD(
    variable='Location',  # The variable being defined
    variable_card=2,  # Number of states for 'Location'
    values=[
        [0.9, 0.1],  # Probabilities for 'On the Mat'
        [0.1, 0.9]   # Probabilities for 'In the Park'
    ],
    evidence=['Action'],  # The variable(s) that 'Location' depends on
    evidence_card=[2],  # Number of states for the evidence variable(s)
    state_names={
        'Location': ['On the Mat', 'In the Park'],  # Names of the states for 'Location'
        'Action': ['Sits', 'Runs']  # Names of the states for 'Action'
    }
)

# -------------------------------
# Add CPDs to the Model
# -------------------------------

# Add the defined CPDs to the Bayesian Network
model.add_cpds(cpd_subject, cpd_action, cpd_location)

# Check if the model is valid
# Ensures that the structure and CPDs are consistent and the model is well-defined.
assert model.check_model()

print("Bayesian Network successfully created!")

# -------------------------------
# Perform Inference
# -------------------------------

# Create an inference object using Variable Elimination
# This allows us to perform probabilistic queries on the Bayesian Network.
inference = VariableElimination(model)

# Predict the 'Location' given that the 'Subject' is 'Cat'
# The query specifies the evidence ('Subject' = 'Cat') and asks for the most probable 'Location'.
result = inference.map_query(variables=['Location'], evidence={'Subject': 'Cat'})

# Print the predicted location
print(f"Predicted Location: {result['Location']}")

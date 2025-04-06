# ![Wide SamHail](https://github.com/user-attachments/assets/ea274ca9-054a-43a3-adf4-ba0ce22700c1)

## Installation

For detailed installation steps, please refer to the [Installation Guide](./docs/INSTALLATION_GUIDE.md).

### **Project Objective: Samhail**

The **Samhail** project is a comprehensive framework for building, validating, and testing various probabilistic and machine learning models. The project focuses on implementing and evaluating models for inference, prediction, and natural language processing (NLP) tasks. It ensures robust testing, environment consistency, and reproducibility across development and deployment workflows.

### **Key Objectives**

1.  **Probabilistic Modeling**:
    
    *   Define and validate **Bayesian Networks** for probabilistic inference and decision-making.
        
    *   Implement **Markov Chains** for sequence modeling and text generation.
        
    *   Use **Markov Chain Monte Carlo (MCMC)** methods for probabilistic sampling and inference.
        
2.  **Deep Learning Models**:
    
    *   Implement and test **LSTM (Long Short-Term Memory)** networks for text prediction and sequence modeling.
        
    *   Use **BERT (Bidirectional Encoder Representations from Transformers)** for advanced NLP tasks like next-word prediction.
        
3.  **Natural Language Processing (NLP)**:
    
    *   Preprocess text data with tasks such as tokenization, lemmatization, stopword removal, and named entity recognition.
        
    *   Implement text augmentation, normalization, and vectorization techniques for NLP pipelines.
        
4.  **Testing and Validation**:
    
    *   Write unit tests using `pytest` to validate the correctness of all models and their outputs.
        
    *   Ensure proper error handling for invalid inputs or configurations.
        
5.  **Environment Consistency**:
    
    *   Maintain consistent Python environments across local development, Jupyter Notebooks, and CI pipelines.
        
    *   Use tools like `pyenv`, `ipykernel`, and `pip` to manage dependencies and ensure reproducibility.
        
6.  **Interactive Notebooks**:
    
    *   Provide Jupyter Notebooks for interactive exploration and demonstration of models and their functionalities.
        
    *   Ensure notebooks use the correct Python interpreter and dependencies.
        
7.  **CI/CD Integration**:
    
    *   Automate testing and validation workflows using GitHub Actions.
        
    *   Ensure dependency installation and environment setup are consistent in CI pipelines.
        

### **Models Implemented**

1.  **Bayesian Network**:
    
    *   Probabilistic graphical model representing dependencies between variables.
        
    *   Example: `'Subject' → 'Action' → 'Location'`.
        
    *   Supports inference using **Variable Elimination**.
        
2.  **Markov Chain**:
    
    *   Sequence modeling technique based on state transitions.
        
    *   Used for text generation and prediction tasks.
        
3.  **Markov Chain Monte Carlo (MCMC)**:
    
    *   Probabilistic sampling method for inference in complex models.
        
4.  **LSTM (Long Short-Term Memory)**:
    
    *   Recurrent neural network (RNN) architecture for sequence modeling.
        
    *   Used for next-word prediction and text generation.
        
5.  **BERT (Bidirectional Encoder Representations from Transformers)**:
    
    *   Transformer-based model for advanced NLP tasks.
        
    *   Used for next-word prediction and contextual understanding.
        
6.  **Text Preprocessor**:
    
    *   Comprehensive NLP pipeline for preprocessing text data.
        
    *   Tasks include tokenization, lemmatization, stopword removal, named entity recognition, and more.
        

### **Technologies and Libraries**

*   **Python**: Core programming language for the project.
    
*   **pgmpy**: Library for creating and working with Bayesian Networks.
    
*   **pomegranate**: Alternative library for probabilistic modeling (optional).
    
*   **spaCy**: NLP library for text preprocessing and linguistic analysis.
    
*   **TensorFlow/Keras**: Frameworks for implementing LSTM and BERT models.
    
*   **pytest**: Framework for writing and running automated tests.
    
*   **Jupyter Notebooks**: For interactive exploration and demonstration.
    
*   **VSCode**: Integrated development environment for coding and debugging.
    

### **Key Features**

1.  **Probabilistic Models**:
    
    *   Define Bayesian Networks, Markov Chains, and MCMC methods.
        
    *   Perform inference and sampling for probabilistic decision-making.
        
2.  **Deep Learning Models**:
    
    *   Implement LSTM and BERT for text prediction and sequence modeling.
        
    *   Evaluate model performance on test datasets.
        
3.  **NLP Pipelines**:
    
    *   Preprocess text data with tasks like tokenization, lemmatization, and stopword removal.
        
    *   Perform advanced NLP tasks like named entity recognition and vectorization.
        
4.  **Testing Framework**:
    
    *   Validate Bayesian Network structures, CPDs, and inference results.
        
    *   Test LSTM and BERT models for accuracy and robustness.
        
    *   Ensure proper error handling for invalid inputs or configurations.
        
5.  **Environment Management**:
    
    *   Align Python environments across local development, notebooks, and CI pipelines.
        
    *   Use `ipykernel` to ensure Jupyter Notebooks use the correct interpreter.
        

### **Challenges and Solutions**

1.  **Inconsistent Library Behavior**:
    
    *   `pgmpy` has inconsistencies across versions (e.g., `BayesianNetwork` vs. `DiscreteBayesianNetwork`).
        
    *   **Solution**: Pin the library version in `requirements.txt` and align code accordingly.
        
2.  **Environment Mismatch**:
    
    *   Different Python interpreters in VSCode, terminal, and notebooks.
        
    *   **Solution**: Use `ipykernel` to align Jupyter Notebook with the correct environment.
        
3.  **Missing Dependencies**:
    
    *   Errors like missing `spaCy models` (`en_core_web_sm`).
        
    *   **Solution**: Install required models and add them to the setup process.
        
4.  **Testing Failures**:
    
    *   Assertion errors or missing modules during `pytest` runs.
        
    *   **Solution**: Debug and fix test cases, ensure proper dependency installation, and validate models.
        

### **Future Scope**

1.  **Expand Model Complexity**:
    
    *   Add support for more complex Bayesian Networks and NLP tasks.
        
2.  **Integrate Alternative Libraries**:
    
    *   Explore `pomegranate` for performance optimization and simpler API usage.
        
3.  **Interactive Visualizations**:
    
    *   Use libraries like `networkx` or pygraphviz to visualize Bayesian Network structures.
        
4.  **Deployment**:
    
    *   Package the project as a library or API for broader use cases.
        

### **Conclusion**

The **Samhail** project provides a robust framework for probabilistic modeling, deep learning, and NLP tasks. By leveraging modern Python libraries and tools, it ensures consistency, reliability, and scalability for real-world applications in inference, prediction, and decision-making.

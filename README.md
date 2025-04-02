# Installation Guide

This guide provides detailed steps to set up and run the project successfully on 
**MacOS Sequoia** using **Anaconda Python Environment** and **Visual Studio Code**.

## Install Anaconda Python Environment
1. Download the latest version of **Anaconda** from the [Anaconda official website](https://www.anaconda.com/).

2. Follow the installation instructions for MacOS.

3. Verify the installation by running: `conda --version`
4. **Set Up a Conda Environment**
---------------------------------

4.1.  Create a new Conda environment for the project: `conda create --name project\_env python=3.9 -y`
    
4.2.  Activate the environment: `conda activate project\_env`
    

5. **Install Python Dependencies**
----------------------------------

Run the following commands to install all required Python libraries:

```bash
pip install numpy
pip install spacy 
pip install langdetect
pip install emoji
pip install pytest
pip install pytest-cov
pip install pytest-mock
pip install pgmpy
pip install tensorflow
pip install transformers 
pip install torch
```

5. **Install Visual Studio Code**
---------------------------------

1.  Download and install **Visual Studio Code** from the [VS Code official website](https://code.visualstudio.com/download).
    
2.  Verify the installation by running: `code --version`
    

6. **Configure Conda Python in Visual Studio Code**
---------------------------------------------------

1.  Open Visual Studio Code.
    
2.  Install the **Python Extension**:
    
    *   Go to the Extensions view (Ctrl+Shift+X or Cmd+Shift+X on Mac).
        
    *   Search for "Python" and install the extension by Microsoft.
        
3.  Configure the Python interpreter:
    
    *   Press Cmd+Shift+P to open the Command Palette.
        
    *   Type Python: Select Interpreter and select it.
        
    *   Choose the Conda environment (project\_env) you created earlier.
        

7. **Install Necessary VS Code Extensions**
-------------------------------------------

Install the following extensions in Visual Studio Code:

1.  **Python** (by Microsoft): For Python development.
    
2.  **GitHub Copilot**: For AI-powered code suggestions.
    
3.  **GitHub Copilot Chat**: For interactive AI assistance.
    
4.  **Pylance**: For Python language support.
    
5.  **Pytest**: For running Python tests.
    
6.  **Prettier**: For code formatting.
    

8. **Run Pytest for Testing**
-----------------------------

1.  Navigate to the project root directory
    
2.  Run the tests using pytest: ` pytest --cov=models/nlps --cov-report=html -v`
    
3.  Example
```python 
output:============================= test session starts =============================platform darwin -- Python 3.9.x, pytest-7.x.x, py-1.x.x, pluggy-1.x.xrootdir: /Users/apple/Documents/Projects/Samhailplugins: cov-3.x.xcollected 10 itemstests/models/base\_models/test\_bert.py ..........                        \[100%\]---------- coverage: platform darwin, python 3.9.x ----------Name                                      Stmts   Miss  Cover   Missing-----------------------------------------------------------------------models/base\_models/bert.py                  50      0   100%-----------------------------------------------------------------------============================== 10 passed in 2.34s ==============================
```    

9. **Additional Notes**
-----------------------

*   Ensure that your Conda environment (project\_env) is activated before running any Python commands.
    
*   If you encounter any issues with TensorFlow or PyTorch, ensure that your MacOS has the necessary hardware support (e.g., GPU drivers).

* If any issue faced for installation, create a GitHub Issue for this project, and providing accurate details.
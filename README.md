# Installation Guide

This guide provides detailed steps to set up and run the project successfully on 
*MacOS Sequoia** using **Python Virtual Environment** and **Visual Studio Code**.

## **Prerequisite**
1. Download the project into your local machine using SSH/HTTPS
2. Navigate to project root directory from terminal

## 1. **Prepare Dependent Library Installation via Homebrew**

```bash
# Install HomeBrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew install openssl readline sqlite3 xz zlib
```

## 2. **Set Proper paths for your `$SHELL`**

2.1.  Add the following to your `~/.zshrc`. MacOS Sequoia uses `zsh` by default. Hit the following commands.

```zsh
echo 'export PATH="/usr/local/opt/curl/bin:$PATH"' >> ~/.zshrc
echo 'export LDFLAGS="/usr/local/opt/curl/lib:$LDFLAGS"' >> ~/.zshrc
echo 'export CPPFLAGS="/usr/local/opt/curl/include:$CPPFLAGS"' >> ~/.zshrc
echo 'export PKG_CONFIG_PATH="/usr/local/opt/curl/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zshrc
```

2.2.  Reload the shell by `source ~/.zshrc`


## 3. **Install pyenv as Python package manager**

3.1.  Install `pyenv` from HomeBrew using `brew install pyenv`

3.2   Add the following to your `~/.zshrc`

```zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

3.3   Check `pyenv` installation using `pyenv --version`

## 4. **Install and set Python for the project**

4.1.  Check available Python versions to install from `pyenv`, confirm version `3.12.7` is available using `pyenv install --list`
    
4.2.  Install Python by `pyenv install 3.12.7`

4.3 Set Python for your project only, not affecting system Python using `pyenv local 3.12.7`

4.4 Check your Python version for the project using `python --version`, in case there is a version mismatch, it means the `pyenv` Python is conflicting with system python.

4.5 Rahash `pyenv` using `pyenv rehash`


5. **Install Python Dependencies for Anaconda Environment**
----------------------------------

5.1.  Update pip using `pip install --upgrade pip`

5.2.  Install dependencies using `pip`

```bash
pip install --no-cache-dir
```

6. **Install Visual Studio Code**
---------------------------------

6.1.  Download and install **Visual Studio Code** from the [VS Code official website](https://code.visualstudio.com/download).
    
6.2.  Verify the installation by running: `code --version`
    

7. **Configure Conda Python in Visual Studio Code**
---------------------------------------------------

7.1.  Open Visual Studio Code.
    
7.2.  Install the **Python Extension**:
    
    *   Go to the Extensions view (Ctrl+Shift+X or Cmd+Shift+X on Mac).
        
    *   Search for "Python" and install the extension by Microsoft.
        
7.3.  Configure the Python interpreter:
    
    *   Press Cmd+Shift+P to open the Command Palette.
        
    *   Type Python: Select Interpreter and select it.
        
    *   Choose the Pyenv environment (project\_env) you created earlier.
        

8. **Install Necessary VS Code Extensions**
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
    
*   If you encounter any issues with TensorFlow or PyTorch, ensure that your MacOS has the necessary hardware support (e.g., GPU drivers).

* If you encounter any installation issues, please create a GitHub Issue with detailed information.
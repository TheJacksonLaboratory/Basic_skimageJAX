# Background knowledge: what you need to know to get the most out of this workshop

This workshop is tailored toward researchers with a working knowledge of Python. If you are a total beginner, it would be a good idea to familiarize yourself with some basic programming concepts before you begin. The [Software Carpentries](http://swcarpentry.github.io/python-novice-gapminder/) website might be a good place to start. You will need to be able to understand and make use of the following Python concepts:

1. Data can be stored in **variables** through **assignment**.

    ```python
    >>> a = 4
    >>> b = 'test'
    >>> a
    4
    >>> b
    'test'
    ```

2. Variables hold data of different **types** (e.g., strings, integers, 64-bit floating-point numbers, etc.).

    ```python
    >>> type(a)
    <class 'int'>
    >>> type(b)
    <class 'str'>
    ```

3. Data can be stored as **lists** and these can also be assigned to variables.

    ```python
    >>> c = [1,2,3,4]
    >>> c
    [1, 2, 3, 4]
    >>> type(c)
    <class 'list'>
    ```

4. **Indexing** can be used to select specific positions in a list.

    ```python
    >>> c[0]
    1
    >>> c[3]
    4
    ```

5. **Functions** can take one or more **arguments**, do something with them, and return a value.

    ```python
    >>> def add(a, b):
    ...     sum = a + b
    ...     return sum
    ...
    >>> add(1 ,3)
    4
    >>> add(34, 92)
    126
    ```

6. Python's functionality can be extended through **packages** and **modules**.

    ```python
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> a = np.array([1,2,3]) #accessing the array function in the np (numpy) package to create an array object
    >>> plt.hist(a)
    (array([1., 0., 0., 0., 0., 1., 0., 0., 0., 1.]), array([1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4, 2.6, 2.8, 3. ]), <a list of 10 Patch objects>)
    ```

7. Data can be represented as **objects**, which have **attributes** as well as associated functions called **methods**.

    ```python
    >>> a.dtype #accessing the data type of object 'a', which is a numpy array (created in #6 above)
    dtype('int64')
    >>> a.max() #a function that finds the maximum value of 'a'
    3
    ```

8. Note that in Python, **dot notation** can be used to call an object method, e.g., `a.max()`, but can also be used to call a function that is part of a package, e.g., `np.array()`.

## Tools we will be using

You will need to use three software tools in addition to Python for this workshop: git, conda, and jupyter.

[git](https://git-scm.com) is only used in this workshop to copy the course notes and code to your local computer, although it is a very powerful tool for managing your projects!

[conda](https://conda.io/docs/) is a tool used for two purposes: to create [environments](https://conda.io/docs/user-guide/tasks/manage-environments.html) for your projects and to manage Python packages. It will be a very good idea to read up on conda before you begin this course, spefically how to use environments.

[jupyter](http://jupyter.org) is a tool for developing your code in "notebooks" (file extension \*.ipynb) in a web browser. The main utility of using notebooks for image analysis is that you can save all of your output on the same page as your code, which helps you and your collaborators to follow the steps of an analysis.

## Setting up for the class

1. Clone the git repository to your computer.

    ```bash
    git clone https://github.com/TheJacksonLaboratory/PythonImagingBasic.git

    ```

    **Windows Users**: You will first need to install [git](https://git-scm.com/download/win). Then you can use Git Bash to use the command shown above

    **As an alternative to using git**, you can also download the repository as a zip file from [Github](https://github.com/TheJacksonLaboratory/Basic_skimageJAX/archive/3.0.zip).

2. Install [Anaconda](https://www.anaconda.com/download/) (or [Miniconda](https://conda.io/miniconda.html))
    * This gives you access to the `conda` Python package management system
    * **Windows Users**: You may want to opt for the full install of Anaconda. This will give you "Anaconda Prompt" which will make it easier for you to follow along here.

3. Using conda in a terminal window (or Anaconda prompt in Windows), create a virtual environment called 'ImPyClass' that uses Python3:

    ```bash
    conda create -n ImPyClass python=3
    ```
  
4. Switch to the ImPyClass virtual environment

      macOS, Linux: `conda activate ImPyClass`

      Windows (Anaconda Prompt): `activate ImPyClass`

5. Install Scikit-Image, Numpy, Jupyter Lab, and NB_Conda:

    ```bash
    conda install scikit-image numpy nb_conda
    conda install -c conda-forge jupyterlab
    ```

6. Launch Jupyter Lab from the Anaconda launcher or command line.

```bash
jupyter lab
```

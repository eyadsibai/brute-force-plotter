Brute Force Plotter
===================
[Work in progress]
Tool to visualize data quickly with no brain usage for plot creation

Installation
------------
will be packaged soon


Example
-------
It was tested on python3 only (Python 3.12+ recommended)

.. code:: bash

	$ git clone https://github.com/eyadsibai/brute_force_plotter.git
	$ cd brute_force_plotter
	$ pip3 install -r requirements.txt
	$ python3 -m src example/titanic.csv example/titanic_dtypes.json example/output

Command Line Options
--------------------
- ``--skip-existing``: Skip generating plots that already exist (default: True)
- ``--theme``: Choose plot style theme (darkgrid, whitegrid, dark, white, ticks) (default: darkgrid)
- ``--n-workers``: Number of parallel workers for plot generation (default: 4)
- ``--export-stats``: Export statistical summary to CSV files

.. code:: bash

	$ python3 -m src example/titanic.csv example/titanic_dtypes.json example/output --theme whitegrid --n-workers 8 --export-stats

Arguments
---------
- json.dump({k:v.name for k,v in df.dtypes.to_dict().items()},open('dtypes.json','w'))  
- the first argument is the input file (csv file with data) `example/titanic.csv <https://github.com/eyadsibai/brute_force_plotter/example/titanic.csv>`_
- second argument is a json file with the data types of each columns (c for category, n for numeric, i for ignore) `example/titanic_dtypes.json <https://github.com/eyadsibai/brute_force_plotter/example/titanic_dtypes.json>`_

.. code:: json

	{
	"Survived": "c",
	"Pclass": "c",
	"Sex": "c",
	"Age": "n",
	"SibSp": "n",
	"Parch": "n",
	"Fare": "n",
	"Embarked": "c",
	"PassengerId": "i",
	"Ticket": "i",
	"Cabin": "i",
	"Name": "i"
	}	

- third argument is the output directory
- c stands for category, i stands for ignore, n for numeric

Features
--------
The tool automatically generates:

**Distribution Plots:**

- Histogram with KDE for numeric variables
- Violin plots for numeric variables
- Bar plots for categorical variables
- Correlation matrices (Pearson and Spearman)
- Missing values heatmap

**2D Interaction Plots:**

- Scatter plots for numeric vs numeric
- Heatmaps for categorical vs categorical
- Bar/Box/Violin/Strip plots for categorical vs numeric

**Statistical Summaries (with --export-stats):**

- Numeric statistics (mean, std, min, max, quartiles)
- Category value counts
- Missing values analysis

Example Plots
-------------.. image:: https://raw.githubusercontent.com/eyadsibai/brute_force_plotter/master/example/output/distributions/Age-dist-plot.png
    :alt: Age Distribution (Histogram with Kernel Density Estimation, Violin Plot)
    :width: 260
    :height: 300
    :align: center
    
.. image:: https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Pclass-Sex-heatmap.png
    :alt: Heatmap for Sex and Pclass
    :width: 260
    :height: 300
    :align: center

.. image:: https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Pclass-Survived-bar-plot.png
    :alt: Pclass vs Survived
    :width: 260
    :height: 300
    :align: center    
    
.. image:: https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Survived-Age-plot.png
    :alt: Survived vs Age
    :width: 260
    :height: 300
    :align: center
    
.. image:: https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Age-Fare-scatter-plot.png
    :alt: Age vs Fare
    :width: 260
    :height: 300
    :align: center

TODO
----
- target variable support
- Tests?
- Support 3 variables (contour plots/ etc)
- Fallback for large datasets
- Figure out the data type or suggest some
- Map visualization (if geocoordinates)
- Minimize the number of plots
- Support for Time Series

Recent Updates (2025)
---------------------
✅ Updated all dependencies to latest stable versions
✅ Added correlation matrix plots (Pearson and Spearman)
✅ Added missing values visualization
✅ Added statistical summary export
✅ Added configurable plot themes
✅ Added parallel processing controls
✅ Added skip-existing-plots option
✅ Improved logging and progress indicators
✅ Code cleanup and better error handling

Brute Force Plotter
===================
[Work in progress]
Tool to visualize data quickly with no brain usage for plot creation

Installation
------------
will be packaged soon


Example
-------
It was tested on python3 only

.. code:: bash

	$ git clone https://github.com/eyadsibai/brute_force_plotter.git
	$ cd brute_force_plotter
	$ pip3 install -r requirements.txt
	$ python3 brute_force_plotter.py example/titanic.csv example/titanic_dtypes.json example/output
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


.. image:: https://raw.githubusercontent.com/eyadsibai/brute_force_plotter/master/example/output/distributions/Age-dist-plot.png
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
- Clean up part of the code
- More documentation
- Tests?
- Support 3 variables (contour plots/ etc)
- Fallback for large datasets
- Figure out the data type or suggest some
- Map visualization (if geocoordinates)
- Minimize the number of plots
- Support for Time Series

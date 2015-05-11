Brute Force Plotter
===================
[Work in progress]
Tool to visualize data quickly with no brain usage for plots creation

Installation
------------
Not yet tested


To Try for Now
--------------
It was tested on python3 only

.. code:: bash

	$ git clone https://github.com/eyadsibai/brute_force_plotter.git
	$ cd brute_force_plotter
	$ pip3 install -r requirements.txt
	$ PYTHONPATH=brute_force_plotter python3 brute_force_plotter/brute_force_plotter.py example/titanic.csv example/titanic_dtypes.json example/output

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

TODO
----
- Clean up part of the code
- More documentation
- Tests?
- Support 3 variables (contour plots/ etc)
- Fallback for large datasets
- Figure out the data type or suggest some
- Map visualization (if geocoordinates)
- Support for Time Series

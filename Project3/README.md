Environment setup:

1. Install python2;
2. Make sure tkinter is installed;
3. Create virtual environment;
4. Activate virtual and run code.

To create the virtual env, run:

$ virtualenv --python=/usr/bin/python2.7 venv

To activate it:

$ source venv/bin/activate

To install tkinter:

$ sudo apt-get install python-tk

To run:

$ python pacman.py -p ApproximateQAgent -a extractor=FeatureExtractor -x 10 -n 15 -l smallClassic
$ python pacman.py -p ApproximateQAgent -a extractor=FeatureExtractor -x 10 -n 15 -l mediumClassic
$ python pacman.py -p ApproximateQAgent -a extractor=FeatureExtractor -x 10 -n 15 -l originalClassic

The relevant code is at agents.py and featureExtractor.py

The code at unused_agents was used to understand the code base.

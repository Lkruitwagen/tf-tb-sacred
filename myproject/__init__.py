import os

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('demo')

ex.observers.append(FileStorageObserver(os.path.join('experiments','sacred')))
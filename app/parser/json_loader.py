import json
from .loader import AbstractLoader


class JsonLoader(AbstractLoader):
    '''
    This is a class for loading data from a json file.
    '''

    def __init__(self, file):
        '''
        Constructor for the JsonLoader class.

        :param self: Instance of the JsonLoader object
        :param file: json file from which we want to load data
        '''
        self.__file = file

    def read_data(self):
        '''
        Access the json data from the given file and 'load' it into 
        a dictionary.

        :param self: Instance of the JsonLoader object
        :return data: Dictionary containing json data
        '''
        # load in json data
        with open(self.__file, 'r') as fp:
            data = json.load(fp)

        return data

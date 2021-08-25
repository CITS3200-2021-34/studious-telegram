import json
from .loader import AbstractLoader


class JsonLoader(AbstractLoader):

    def __init__(self, file):
        self.__file = file

    def read_data(self):

        # load in json data
        with open(self.__file, 'r') as fp:
            data = json.load(fp)

        return data

from .dataformatter import AbstractDataFormater


class DataParser(AbstractDataFormater):
    '''
    Class for parsing the data from a text file into individual post/thread strings.
    '''

    def __init__(self, file):
        '''
        Constructor for the DataParser class.

        :param self: Instance of the DataParser object
        :param file: A text file to be parsed
        '''
        self.file = file

    def parse_text(self) -> list[str]:
        '''
        Arranges the content of a file into a list of separate question (or answer) 
        strings based on their date line.

        :param self: Instance of the DataParser object
        :return threads: The list of strings, where each string is an individual
            'post'/'thread', ie. question or answer
        '''
        with open(self.file, 'r') as f:
            line = f.readline()
            line = f.readline()  # Not sure best way to do this, needed to skip the first line
            str = ''
            threads = []

            while line:
                if(line[:4] == 'Date' and len(str) > 0):
                    str = str.rstrip('\n')
                    threads.append(str)
                    str = ''
                str += line
                line = f.readline()
            threads.append(str)
        return threads

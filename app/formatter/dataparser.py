from .dataformatter import AbstractDataFormater


class DataParser(AbstractDataFormater):

    def __init__(self, file):
        self.file = file

    def parse_text(self) -> list[str]:
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

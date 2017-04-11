SUPPORTED_ENGINES = ['hoomd', 'lammps']


class UnsupportedEngine(Exception):
    def __init__(self, engine):
        message = 'Unsupported engine: "{0}". Supported engines are: {1}'.format(
            engine, ', '.join(SUPPORTED_ENGINES))
        super(UnsupportedEngine, self).__init__(message)


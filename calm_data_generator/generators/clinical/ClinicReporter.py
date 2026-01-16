from ..Synthetic.StreamReporter import StreamReporter


class ClinicReporter(StreamReporter):
    """
    Reporter specialized for Clinical data blocks.
    Inherits all functionality from StreamReporter but allows for future specializations.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.logger.name = "ClinicReporter"

import logging
import warnings
from icecream import ic


def setup_logging(file_name: str) -> None:

    logging.basicConfig(
        filename=file_name,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    def icecream_logger(*args):
        logging.info(', '.join(str(arg) for arg in args))

    ic.enable()
    ic.configureOutput(outputFunction=icecream_logger)

    def warn_with_log(
        message, category, filename, lineno, file=None, line=None
    ):
        logging.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

    warnings.filterwarnings("default")
    warnings.showwarning = warn_with_log

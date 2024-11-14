import logging
import sys

from colorama import Fore

__all__ = ["logger"]


class CustomFormatter(logging.Formatter):
    fmt_str = "[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: f"{Fore.LIGHTMAGENTA_EX}{fmt_str}{Fore.RESET}",
        logging.INFO: f"{Fore.WHITE}{fmt_str}{Fore.RESET}",
        logging.WARNING: f"{Fore.YELLOW}{fmt_str}{Fore.RESET}",
        logging.ERROR: f"{Fore.LIGHTRED_EX}{fmt_str}{Fore.RESET}",
        logging.CRITICAL: f"{Fore.RED}{fmt_str}{Fore.RESET}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger with 'spam_application'
logger = logging.getLogger("cfdmod")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

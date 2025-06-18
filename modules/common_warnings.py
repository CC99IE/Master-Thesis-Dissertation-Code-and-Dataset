import warnings
import logging
from PyPDF2.errors import PdfReadWarning

# Suppress PyPDF2 PdfReadWarning
warnings.filterwarnings("ignore", category=PdfReadWarning)

# Force PyPDF2 logger to only show ERROR or above
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
import logging
import _pytest.logging

# 2022-09-22: This modification allows live and captured logs to be 
# formatted in the expected way when running tests using pytest. 
# This is because _pytest.LiveLoggingStreamHandler is a subclass of 
# logging.StreamHandler and is used instead of geomeTRIC's own loggers
# during test runs.
# It took a long time to figure out and I'm glad it only took one line to fix it!

logging.StreamHandler.terminator = ""

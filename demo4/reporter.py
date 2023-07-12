import logging

def reporter_loader(loggerLevel, train_name):
    reporter = logging.getLogger("Logger")

    if loggerLevel == "info":
        reporter.setLevel(logging.INFO)
    elif loggerLevel == "debug":
        reporter.setLevel(logging.DEBUG)
    else:
        reporter.setLevel(logging.WARNING)

    stream_hander = logging.StreamHandler()			
    formatter = logging.Formatter('[%(levelname)s] %(message)s')	
    stream_hander.setFormatter(formatter)			        
    reporter.addHandler(stream_hander)				

    file_handler = logging.FileHandler(f"logs/{train_name}.log")
    formatter = logging.Formatter('"%(asctime)s [%(levelname)s] %(message)s')	
    file_handler.setFormatter(formatter)
    reporter.addHandler(file_handler)		

    reporter.info("Reporter Initialized!")
    reporter.info(f"Reporter LEVEL: {loggerLevel}")			

    return reporter
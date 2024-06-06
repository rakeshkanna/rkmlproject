import sys
from logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exec_info = error_detail.exc_info()
    file_Name = exec_info.tb_frame.f_code.co_filename
    line_number = exec_info.tb_lineno

    error_message = "Error Occured in Python Script [{0}] at Line Number [{1}] error Message [{2}]".format(
        file_Name,line_number,str(error) 
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)
    def __str__(self) -> str:
        return self.error_message

if __name__== "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero Error")
        raise CustomException(e,sys)


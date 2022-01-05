import os
import time
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time
import threading
#import re
from bs4 import BeautifulSoup
#import pandas as pd

#from selenium.webdriver.common.keys import Keys



def get_today() : #오늘 가져오는 애
    now = time.localtime()
    s ="%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
    return s




def make_folder(): #폴더 만드는 애
    root_dir = "D:/testImage"
    today = get_today()
    work_dir = root_dir + "/" + today
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

        for i in range(0,24):
            c = str(i)
            if i < 10:
                os.mkdir(work_dir + "/0" + c)
            else:
                os.mkdir(work_dir + "/" + c)



schedule.every().day.at("03:14").do(make_folder)

while True:
    schedule.run_pending()
    time.sleep(1)
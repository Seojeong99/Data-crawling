from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time
import datetime
import cv2
#import re
from bs4 import BeautifulSoup
#import pandas as pd

#from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 3)
URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010273&cctvname=%25EC%259A%25A9%25EC%2582%25B0%25EA%25B0%2580%25EC%25A1%25B1%25EA%25B3%25B5%25EC%259B%2590&kind=Seoul&cctvip=null&cctvch=52&id=524&cctvpasswd=null&cctvport=null&minX=126.93214900832989&minY=37.51694865289079&maxX=127.02387136440655&maxY=37.536838850082916"
driver.get(URL1)
time.sleep(5)



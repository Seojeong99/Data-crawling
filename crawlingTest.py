from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time
from bs4 import BeautifulSoup
from pprint import pprint
import requests
from urllib.request import urlretrieve

import re
from bs4 import BeautifulSoup
import pandas as pd

from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 2)

URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04096&cctvname=%25EC%259B%2594%25EA%25B3%25841%25EA%25B5%2590-%25EB%2585%25B9%25EC%25B2%259C%25EA%25B5%2590&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=932&minX=127.03624212566622&minY=37.6356911150544&maxX=127.08718166625532&maxY=37.65900425441203"
driver.get(URL1)
time.sleep(3)

img_folder = './img'

 index, link in enumerate(img_url):
    #     start = link.rfind('.')
    #     end = link.rfind('&')
    #     filetype = link[start:end]
    urlretrieve(link, f'./img/{index}.jpg')


a = driver.find_element_by_xpath('/html/body/div/div/object/div')\
    #.click()
img_folder = './img'
urlretrieve(a, 'C:/please/a.jpg')

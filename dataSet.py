from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import time
import re
from bs4 import BeautifulSoup
import pandas as pd

from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 2)

URL1 = "https://land.naver.com/"
driver.get(URL1)
time.sleep(3)
driver.find_element_by_xpath('//*[@id="queryInputHeader"]').click()
driver.find_element_by_xpath('//*[@id="queryInputHeader"]').send_keys("안녕")
time.sleep(3)

'''
driver.find_element_by_xpath('/html/body/div[3]/div[3]/a/span').click()
time.sleep(3)
driver.find_element_by_xpath('//*[@id="queryInputHeader"]').click()
time.sleep(3)

driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div[1]/div[1]/div[1]/div[2]/a').click()
time.sleep(3)
driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div[1]/div[1]/div[1]/div[2]/div/ul/li[6]/a').click()

time.sleep(7)
soup = BeautifulSoup(driver.page_source, "html.parser")
result = soup.find('p', class_='desc_translation').text
print(result)
'''
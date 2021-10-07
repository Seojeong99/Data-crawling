from threading import Thread
import os
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time
import clipboard


driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 3)
URL1 = 'http://www.utic.go.kr/main/main.do'
driver.get(URL1)
driver.implicitly_wait(10)

#print(driver.window_handles)

time.sleep(1)
driver.switch_to.window(driver.window_handles[1])
driver.close()

driver.switch_to.window(driver.window_handles[0])


driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[2]/div/a[6]').click()
driver.find_element_by_xpath('//*[@id="searchText"]').send_keys("서울시립노원청소년센터")
driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[5]/div[1]/img').click()
driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[5]/div[2]/ul/li[1]/a').click()

'''
time.sleep(3)
pyautogui.click(x=488, y=591)
time.sleep(3)
pyautogui.doubleClick(x=552, y=547)
pyautogui.hotkey('ctrl', 'c')
tmp = clipboard.paste()
print("노원시각장애인복지관~마들지하차도남측"+tmp)

'''
time.sleep(3)
pyautogui.click(x=488, y=591)
#상대 좌표
#
time.sleep(3)
pyautogui.doubleClick(x=552, y=547)
pyautogui.hotkey('ctrl', 'c')
tmp = clipboard.paste()
print("학야여울청구아파트~녹천교앞교차로 "+tmp)

'''
time.sleep(5)
pyautogui.click(x=488, y=591)
time.sleep(3)
pyautogui.doubleClick(x=552, y=547)
pyautogui.hotkey('ctrl', 'c')
tmp = clipboard.paste()
print("학야여울청구아파트~녹천교앞교차로"+tmp)
'''


from threading import Thread
import os
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time
import clipboard
import ctypes

pyautogui.FAILSAFE = False

def screensize():
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    #print(screensize[0],screensize[1]) #1280 720

def popcnt():
    time.sleep(1)
    tmp = driver.window_handles
    print(tmp)

    #driver.switch_to.window(driver.window_handles[1])
    #driver.close()
    #driver.switch_to.window(driver.window_handles[0])

def apartment():
    driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[2]/div/a[6]').click()
    driver.find_element_by_xpath('//*[@id="searchText"]').send_keys("상계마들아파트")
    driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[5]/div[1]/img').click()
    driver.find_element_by_xpath('/html/body/div/div[2]/div/div[1]/div[5]/div[2]/ul/li[1]/a').click()
    time.sleep(4)
    pyautogui.moveTo(1324, 378)
    pyautogui.dragTo(1324, 344, button='left')#확대
    time.sleep(4)
    pyautogui.doubleClick(x=172, y=951)
    time.sleep(4)
    apartmentDrag()


def apartmentDrag():
    pyautogui.moveTo(0, 870)
    time.sleep(2)
    pyautogui.dragTo(287, 913, 1)
    pyautogui.hotkey('ctrl', 'c')
    tmp = clipboard.paste()
    print(tmp)



if __name__=='__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    URL1 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL1)
    driver.implicitly_wait(10)

    URL2 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL2)
    driver.implicitly_wait(10)

    URL3 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL3)
    driver.implicitly_wait(10)
    popcnt()
    apartment()

'''
time.sleep(3)
x = 448*65535/screensize[0]
y = 591*65535/screensize[1]
print(x, y)
pyautogui.click(x,y)

#pyautogui.click(x=488*(1920/screensize[0]), y=591*(1080/screensize[1]))
#상대 좌표
#
time.sleep(3)

#pyautogui.doubleClick(x=552*(1920/screensize[0]), y=547*(1080/screensize[1]))
pyautogui.hotkey('ctrl', 'c')
tmp = clipboard.paste()
print("학야여울청구아파트~녹천교앞교차로 "+tmp)



'''

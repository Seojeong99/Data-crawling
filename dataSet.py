from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time
import threading
#import re
from bs4 import BeautifulSoup

#from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 3)
URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010099&cctvname=%25EC%25A2%2585%25EB%25A1%259C2%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=122&cctvpasswd=null&cctvport=null&minX=126.96795976708209&minY=37.55907499804575&maxX=127.0188451664223&maxY=37.582412013189526"
driver.get(URL1)
time.sleep(5)

for i in range(0,20):
    c = str(i)
    im1 = pyautogui.screenshot('D:/testImage/test' + c + '.jpg', region=(713,199,420,280))
    time.sleep(1)

#날짜, 년,월,일,시,분,초 시간대별로이름
#일별로 폴더 만들어서 업데이트
#시간마다 폴더 만들어서 업데이트
#while 무제한으로.
#프로그램 시작하면 폴더부터 만들어. 210930 1시부터 24시까지 폴더 만들기 각 날짜와 시간에 맞춰서 들어가기
#밤 12시가되면 또 폴더 만들기
#계속 캡쳐하는 쓰레드
#계속 시간측정하고 쓰레드 정각이되면 폴더 만들기

#날짜, 년,월,일,시,분,초 시간대별로 이름


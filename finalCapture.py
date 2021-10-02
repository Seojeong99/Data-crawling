from threading import Thread
import os
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time


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

def test():
    print('1')

def capture(): #캡쳐하는애
    im1 = pyautogui.screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                               + get_today()+'-'+time.strftime('%M')
                               + '-' + time.strftime('%S') + '.jpg',
                               region=(713, 199, 420, 280))
    #time.sleep(1)


def checkAndMake():
    schedule.every().day.at("00:00").do(make_folder)


if __name__ == '__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010099&cctvname=%25EC%25A2%2585%25EB%25A1%259C2%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=122&cctvpasswd=null&cctvport=null&minX=126.96795976708209&minY=37.55907499804575&maxX=127.0188451664223&maxY=37.582412013189526"
    driver.get(URL1)
    time.sleep(5)
    make_folder()

    while (1):
        schedule.run_pending()
        thread1 = Thread(target=checkAndMake)
        thread2 = Thread(target=capture)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()





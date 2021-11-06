from threading import Thread
import os
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
import time


def click():
    driver.find_element_by_xpath('/html/body/div/div/object/div/div/div[4]/button[2]/span[1]').click()
    time.sleep(31)

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


def capture(): #캡쳐하는애
        im1 = pyautogui.screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                               + get_today()+'-' + time.strftime('%H')
                               + '-' + time.strftime('%M')
                               + '-' + time.strftime('%S') + '.jpg',
                               region=(242, 683, 850, 400))
                               #region=(816, 199, 420, 280))
                               #region=(241, 115, 420, 280))

        time.sleep(1)




def checkAndMake():
    schedule.every().day.at("00:00").do(make_folder)


if __name__ == '__main__':

    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)

    URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010098&cctvname=%25EC%25A2%2585%25EB%25A1%259C1%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=121&cctvpasswd=null&cctvport=null&minX=126.97260204133853&minY=37.55362493982439&maxX=126.99498168111192&maxY=37.57894357098292"
    driver.get(URL1)
    time.sleep(5)
    #driver.find_element_by_xpath('/html/body/div/div/object/div/div/div[4]/button[2]/span[1]').click()
    make_folder()

    while (1):

        schedule.run_pending()
        thread1 = Thread(target=capture)
        #thread2 = Thread(target=click)
        thread1.start()
        #thread2.start()
        #thread1.sleep(31)
        thread1.join()
        #thread2.join()





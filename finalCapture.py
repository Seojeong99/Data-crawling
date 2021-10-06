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


def capture(): #캡쳐하는애
    im1 = pyautogui.screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                               + get_today()+'-' + time.strftime('%H')
                               + '-' + time.strftime('%M')
                               + '-' + time.strftime('%S') + '.jpg',
                               region=(713, 199, 420, 280))
                               #region=(816, 199, 420, 280))
                               #region=(241, 115, 420, 280))

    #time.sleep(1)


def checkAndMake():
    schedule.every().day.at("00:00").do(make_folder)


if __name__ == '__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    #URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010099&cctvname=%25EC%25A2%2585%25EB%25A1%259C2%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=122&cctvpasswd=null&cctvport=null&minX=126.96795976708209&minY=37.55907499804575&maxX=127.0188451664223&maxY=37.582412013189526
    #URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04121&cctvname=%25EC%25A2%2585%25EC%2595%2594JC-%25ED%2595%2598%25EC%259B%2594%25EA%25B3%25A1&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=439&minX=127.02432898991978&minY=37.59403300413159&maxX=127.05478445249852&maxY=37.61069347720573"
    #URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04097&cctvname=%25EC%259B%2594%25EA%25B3%25A1%25EB%259E%25A8%25ED%2594%2584&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=227&minX=127.0294806973648&minY=37.59255398619633&maxX=127.05993621314144&maxY=37.60921356802172"
    URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04096&cctvname=%25EC%259B%2594%25EA%25B3%25841%25EA%25B5%2590-%25EB%2585%25B9%25EC%25B2%259C%25EA%25B5%2590&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=932&minX=127.03624212566622&minY=37.6356911150544&maxX=127.08718166625532&maxY=37.65900425441203"
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





import threading
import time
import pyautogui
import os
import schedule
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

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

# TimerThread
class TimerThread(threading.Thread):
    def __init__(self):
        self.currentTime = 0
        threading.Thread.__init__(self, name='Timer Thread')

    # TimerThread가 실행하는 함수
    def run(self):
        # 10초마다
        while True:
            # 10초 기다린다
            driver.find_element_by_xpath('/html/body/div/div/object/div/div/div[4]/button[2]/span[1]').click()
            time.sleep(31)

            #self.currentTime += 31
            #print("프로그램을 실행한 시간(초): " + str(self.currentTime))


if __name__ == '__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)

    URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010098&cctvname=%25EC%25A2%2585%25EB%25A1%259C1%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=121&cctvpasswd=null&cctvport=null&minX=126.97260204133853&minY=37.55362493982439&maxX=126.99498168111192&maxY=37.57894357098292"
    driver.get(URL1)
    time.sleep(5)
    # driver.find_element_by_xpath('/html/body/div/div/object/div/div/div[4]/button[2]/span[1]').click()
    make_folder()

    timer = TimerThread()
    # Daemon Thread로 설정하지 않음, 기본값임
    timer.setDaemon(False)
    # 타이머용 Thread 실행
    timer.start()


    while True:
            im1 = pyautogui.screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                                       + get_today() + '-' + time.strftime('%H')
                                       + '-' + time.strftime('%M')
                                       + '-' + time.strftime('%S') + '.jpg',
                                       region=(242, 683, 850, 400))
            # region=(816, 199, 420, 280))
            # region=(241, 115, 420, 280))

            time.sleep(1)

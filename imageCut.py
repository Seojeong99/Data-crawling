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
            time.sleep(30)

            #self.currentTime += 31
            #print("프로그램을 실행한 시간(초): " + str(self.currentTime))


if __name__ == '__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010144&cctvname=%25EC%2598%25AC%25EB%25A6%25BC%25ED%2594%25BD%25EA%25B3%25B5%25EC%259B%2590&kind=Seoul&cctvip=null&cctvch=52&id=167&cctvpasswd=null&cctvport=null&minX=127.1006466052482&minY=37.50181730698737&maxX=127.13109817174727&maxY=37.527114302340955"
    #URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010214&cctvname=%25EC%2598%25AC%25EB%25A6%25BC%25ED%2594%25BD%25EB%258C%2580%25EA%25B5%2590%25EB%2582%25A8%25EB%258B%25A8&kind=Seoul&cctvip=null&cctvch=52&id=270&cctvpasswd=null&cctvport=null&minX=127.09912824468147&minY=37.50792746385415&maxX=127.15004245416765&maxY=37.54851834572314"
    #URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010193&cctvname=%25EC%25B0%25BD%25EB%258D%2595%25EA%25B6%2581R&kind=Seoul&cctvip=null&cctvch=52&id=249&cctvpasswd=null&cctvport=null&minX=126.97656150723212&minY=37.560806567436394&maxX=127.007003856218&maxY=37.58612490336802"
    #URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=L010098&cctvname=%25EC%25A2%2585%25EB%25A1%259C1%25EA%25B0%2580&kind=Seoul&cctvip=null&cctvch=52&id=121&cctvpasswd=null&cctvport=null&minX=126.97260204133853&minY=37.55362493982439&maxX=126.99498168111192&maxY=37.57894357098292"
    driver.get(URL1)
    time.sleep(5)
    # driver.find_element_by_xpath('/html/body/div/div/object/div/div/div[4]/button[2]/span[1]').click()
    make_folder()

    timer = TimerThread()
    # Daemon Thread로 설정하지 않음,  기본값임
    timer.setDaemon(False)
    # 타이머용 Thread 실행
    timer.start()


    while True:
            im1 = pyautogui.screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                                       + get_today() + '-' + time.strftime('%H')
                                       + '-' + time.strftime('%M')
                                       + '-' + time.strftime('%S') + '.jpg',
                                       region=(620, 571, 300, 350))
                                       #region=(873, 536, 250, 380))#앞이 폭 뒤가 높이
                                       #region=(242, 683, 850, 400))
                                       #region=(460,644,400,230))
            # region=(816, 199, 420, 280))
            # region=(241, 115, 420, 280))

            time.sleep(1)

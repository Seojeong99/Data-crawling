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

    im1 = driver.find_element_by_xpath('/html/body/div/div/object/div').screenshot('D:/testImage/' + get_today() + '/' + time.strftime('%H') + '/'
                               + get_today()+'-' + time.strftime('%H')
                               + '-' + time.strftime('%M')
                               + '-' + time.strftime('%S') + '.png')

                               #,region=(713, 199, 420, 280))
                               #region=(816, 199, 420, 280))
                               #region=(241, 115, 420, 280))

    #time.sleep(1)


def checkAndMake():
    schedule.every().day.at("00:00").do(make_folder)


def screensize():
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    #print(screensize[0],screensize[1]) #1280 720

def popcnt():
    driver.switch_to.window(driver.window_handles[1])
    driver.close()
    driver.switch_to.window(driver.window_handles[0])

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

'''
def windowNum():
    tabs = driver.window_handles
    driver.switch_to.window(tabs[0])
    driver.get('http://www.naver.com/')
    driver.switch_to.window(tabs[1])
    driver.get('http://www.google.com/')
    driver.switch_to.window(tabs[2])
    driver.get('https://heodolf.tistory.com/')
'''

def driverCctv():
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    URL1 = "http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04096&cctvname=%25EC%259B%2594%25EA%25B3%25841%25EA%25B5%2590-%25EB%2585%25B9%25EC%25B2%259C%25EA%25B5%2590&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=932&minX=127.03624212566622&minY=37.6356911150544&maxX=127.08718166625532&maxY=37.65900425441203"
    driver.get(URL1)
    time.sleep(5)
    make_folder()

def driverMap1():
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    URL2 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL2)
    time.sleep(5)
    #tmp = driver.window_handles
    #print(tmp[0])
    #driver.switch_to.window(tmp[0])
    #apartment()

def driverMap2():
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    URL2 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL2)
    time.sleep(5)

def driverMap3():
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    URL2 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL2)
    time.sleep(5)


def driverMap4():
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    URL2 = 'http://www.utic.go.kr/main/main.do'
    driver.get(URL2)
    time.sleep(5)



if __name__=='__main__':
    driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
    driver.wait = WebDriverWait(driver, 3)
    driver.implicitly_wait(5)
    schedule.run_pending()
    # thread2 = Thread(target=capture)
    thread3 = Thread(target=driverMap1)
    thread4 = Thread(target=driverMap2)
    thread5 = Thread(target=driverMap3)
    thread6 = Thread(target=driverMap4)
    # thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    # thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    '''
    while (1):
        
        schedule.run_pending()
        #thread1 = Thread(target=checkAndMake)
        #thread1.start()        
        #thread1.join()
        
        '''


'''
thread로 6개 동시에 열고(cctv까지)
4개 동시에 검색
팝업 다 없애기
cctv틀기
창각각에서 확대
---------
창각각에서 가져오기
cctv캡쳐하기
엑셀에 쓰기
닫기
'''
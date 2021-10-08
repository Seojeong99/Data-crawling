from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 3)
driver.implicitly_wait(5)
URL1="http://www.utic.go.kr/view/map/cctvStream.jsp?cctvid=E04096&cctvname=%25EC%259B%2594%25EA%25B3%25841%25EA%25B5%2590-%25EB%2585%25B9%25EC%25B2%259C%25EA%25B5%2590&kind=A&cctvip=176&cctvch=null&id=null&cctvpasswd=null&cctvport=932&minX=127.03624212566622&minY=37.6356911150544&maxX=127.08718166625532&maxY=37.65900425441203"
driver.get(URL1)



# 브라우저 위치 조정하기
driver.set_window_position(0,0)


# 브라우저 화면 크기 변경하기
driver.set_window_size(1920, 1080)


# 브라우저 화면 크기 변경하기
#driver.maximize_window()

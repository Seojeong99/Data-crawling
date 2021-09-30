import pyautogui
import time

for i in range(0,10):
    c = str(i)
    im1 = pyautogui.screenshot('D:/testImage/test' + c + '.jpg',region=(100,200,300,400))
    time.sleep(1)

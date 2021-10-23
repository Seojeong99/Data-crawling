import pyautogui
import time

while(1):
    x, y = pyautogui.position()
    print('x={0}, y={1}'.format(x, y))
    time.sleep(3)

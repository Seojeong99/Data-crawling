import clipboard
import pyautogui
#clipboard.copy()
import time

time.sleep(2)
pyautogui.doubleClick(626,659)
pyautogui.hotkey('ctrl', 'c')
#pyautogui.hotkey('ctrl', 'v')
tmp = clipboard.paste()
print(tmp)

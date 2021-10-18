import pyautogui
import time
import clipboard

time.sleep(3)
pyautogui.moveTo(0, 866)
time.sleep(2)
pyautogui.dragTo(287, 913, 1)
pyautogui.hotkey('ctrl', 'c')
tmp = clipboard.paste()
print(tmp)
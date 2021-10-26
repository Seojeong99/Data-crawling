import threading
import pyautogui
import time
def execute(number):
    pyautogui.moveTo(0,100)
    pyautogui.dragTo(100,200)

def execute2(number):
   #time.sleep(5)
   pyautogui.moveTo(100,100)
   pyautogui.dragTo(50,30)

if __name__=='__main__':
    for i in range(1,8):
        my_thread = threading.Thread(target=execute,args=(i,))
        my_thread2 = threading.Thread(target=execute2, args=(i,))
        my_thread.start()
        my_thread2.start()
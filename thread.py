from threading import Thread
import time

def my_thread(val):
          print("Hi")


## 인스턴스 만들기
## 첫번째 argu는 스레드 함수 이름, 두번째 argu는 매개변수를 튜플 형태로 전달한 것.
t1 = Thread(target = my_thread, args=(1,))

## 스레드 시작.
t1.start()
t1.join()

## main
for i in range(3):
      print("I'm main Thread")
print("----the end----")
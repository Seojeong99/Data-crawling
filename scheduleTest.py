import schedule
import time


def job():  # TO DO ... # 10초에 한번씩 실행 schedule.every(10).second.do(job)
    print(1)


schedule.every(1).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)

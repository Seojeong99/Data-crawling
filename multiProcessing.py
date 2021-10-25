from multiprocessing import Process

def job():
    print(1)

def job2():
    print(2)

if __name__== '__main__':
    while(1):
        process1 = Process(target=job)
        process2 = Process(target=job2)
        process1.start()
        process2.start()
        process1.join()
        process2.join()
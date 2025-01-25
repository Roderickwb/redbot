import time
import threading

def threadA():
    while True:
        time.sleep(1)

def threadB():
    while True:
        time.sleep(1)

if __name__ == "__main__":
    # Start 2 threads
    tA = threading.Thread(target=threadA, daemon=True)
    tB = threading.Thread(target=threadB, daemon=True)
    tA.start()
    tB.start()

    while True:
        time.sleep(2)

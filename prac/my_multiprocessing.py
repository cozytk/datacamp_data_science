import multiprocessing
import time

num_list = ['p1', 'p2', 'p3', 'p4']
start = time.time()


def count(name):
    for i in range(0, 100000000):
        a = 1 + 2
    print("finish : ", name)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(count, num_list)
    pool.close()
    pool.join()

print("time :", time.time() - start)
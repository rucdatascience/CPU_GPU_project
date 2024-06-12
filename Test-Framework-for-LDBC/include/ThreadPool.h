/*this is from https://github.com/progschj/ThreadPool 

an explanation: https://www.cnblogs.com/chenleideblog/p/12915534.html
*/


#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t); // 线程池的构造函数
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>; // 将任务添加到线程池的任务队列中
    ~ThreadPool(); // 线程池的析构函数
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers; // 用于存放线程的数组，用vector容器保存
    // the task queue
    std::queue< std::function<void()> > tasks; // 用于存放任务的队列，用queue队列进行保存。任务类型为std::function<void()>。因为std::function是通用多态函数封装器，本质上任务队列中存放的是一个个函数
    
    // synchronization
    std::mutex queue_mutex; // 一个访问任务队列的互斥锁，在插入任务或者线程取出任务都需要借助互斥锁进行安全访问
    std::condition_variable condition; // 一个用于通知线程任务队列状态的条件变量，若有任务则通知线程可以执行，否则进入wait状态
    bool stop; // 标识线程池的状态，用于构造与析构中对线程池状态的了解
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) // 构造函数定义为inline。接收参数threads表示线程池中要创建多少个线程。
    :   stop(false) // stop初始为false，即表示线程池启动着。
{
    for(size_t i = 0;i<threads;++i) // 进入for循环，依次创建threads个线程，并放入线程数组workers中。
        workers.emplace_back( 
            /*
            在vector中，emplace_back()成员函数的作用是在容器尾部插入一个对象，作用效果与push_back()一样，但是两者有略微差异，
            即emplace_back(args)中放入的对象的参数，而push_back(OBJ(args))中放入的是对象。即emplace_back()直接在容器中以传入的参数直接调用对象的构造函数构造新的对象，
            而push_back()中先调用对象的构造函数构造一个临时对象，再将临时对象拷贝到容器内存中。
            
            lambda表达式的格式为：
            [ 捕获 ] ( 形参 ) 说明符(可选) 异常说明 attr -> 返回类型 { 函数体 }
            所以下述lambda表达式为 [ 捕获 ] { 函数体 } 类型。传入的lambda表达式就是创建线程的执行函数.       
            */
            [this] // 该lambda表达式捕获线程池指针this用于在函数体中使用（调用线程池成员变量stop、tasks等）    
            {
                for(;;) // for(;;)为一个死循环，表示每个线程都会反复这样执行，这其实每个线程池中的线程都会这样。
                {
                    std::function<void()> task; // 在循环中，先创建一个封装void()函数的std::function对象task，用于接收后续从任务队列中弹出的真实任务。

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex); // 该{}内，queue_mutex处于锁状态

                        /*即其表示若线程池已停止或者任务队列中不为空，则不会进入到wait状态。
                          由于刚开始创建线程池，线程池表示未停止，且任务队列为空，所以每个线程都会进入到wait状态。
                          若后续条件变量来了通知，线程就会继续往下进行
                        */
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });

                        /*若线程池已经停止且任务队列为空，则return，即所有线程跳出死循环、被抛出线程池；在此之前，每个线程要么在wait状态，要么在执行下面的task*/
                        if(this->stop && this->tasks.empty())
                            return;

                        /*
                        将任务队列中的第一个任务用task标记，然后将任务队列中该任务弹出。（此处线程实在获得了任务队列中的互斥锁的情况下进行的，在条件标量唤醒线程后，
                        线程在wait周期内得到了任务队列的互斥锁才会继续往下执行。所以最终只会有一个线程拿到任务，不会发生惊群效应）
                        在退出了{ }，我们队任务队列的所加的锁也释放了，然后我们的线程就可以执行我们拿到的任务task了，执行完毕之后，线程又进入了死循环。
                        */
                        task = std::move(this->tasks.front());
                        this->tasks.pop(); // task不会堆积在queue中
                    }

                    task();
                }
            }
        );
}

// add new work item to the pool
/*
equeue是一个模板函数，其类型形参为F与Args。其中class... Args表示多个类型形参。
auto用于自动推导出equeue的返回类型，函数的形参为(F&& f, Args&&... args)，其中&&表示右值引用。表示接受一个F类型的f，与若干个Args类型的args。

typename std::result_of<F(Args...)>::type   //获得以Args为参数的F的函数类型的返回类型
std::future<typename std::result_of<F(Args...)>::type> //std::future用来访问异步操作的结果
最终返回的是放在std::future中的F(Args…)返回类型的异步执行结果。
*/
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> // 表示返回类型，与lambda表达式中的表示方法一样。
{
    using return_type = typename std::result_of<F(Args...)>::type; // 获得以Args为参数的F的函数类型的返回类型

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future(); // res中保存了类型为return_type的变量，有task异步执行完毕才可以将值保存进去
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one(); // //任务加入任务队列后，需要去唤醒一个线程
    return res; // //待线程执行完毕，将执行的结果返回
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    /*
    在析构函数中，先对任务队列中加锁，将停止标记设置为true，这样后续即使有新的插入任务操作也会执行失败
    */
    {
        std::unique_lock<std::mutex> lock(queue_mutex); // 该{}内，queue_mutex处于锁状态
        stop = true;
    }
    /*使用条件变量唤醒所有线程，所有线程都会往下执行*/
    condition.notify_all();

    /*在stop设置为true且任务队列中为空时，对应的线程进而跳出循环结束
    将每个线程设置为join，等到每个线程结束完毕后，主线程再退出。*/
    for(std::thread &worker: workers)
        worker.join();
}

#endif











/*Example:

#include <iostream>
#include <tool_functions/ThreadPool.h>
using namespace std;

class example_class
{
public:
    int a;
    double b;
};
example_class example_function(example_class x)
{
    return x;
}

void ThreadPool_example()
{
    ThreadPool pool(4);	// 创建一个线程池，池中线程为4
    std::vector<std::future<example_class>> results; // return typename: example_class; 保存多线程执行结果
    for (int i = 0; i < 10; ++i)
    { // 创建10个任务
        int j = i + 10;
        results.emplace_back(  // 保存每个异步结果
            pool.enqueue([j] { // 将每个任务插入到任务队列中，lambda表达式： pass const type value j to thread; [] can be empty
                example_class a;
                a.a = j;
                return example_function(a); // return to results; the return type must be the same with results
            }));
    }
    for (auto &&result : results)			// 一次取出保存在results中的结果
        std::cout << result.get().a << ' '; // result.get() makes sure this thread has been finished here;
    results.clear(); // future get value can only be called once. get后results里面的future就不能用了；clear之后results才能被再次使用
    std::cout << std::endl;
    // if result.get() is pair<int, string>, then you cannot use result.get().first = result.get().second
}

int main()
{
    ThreadPool_example();
}

*/



/*Example: multi_thread write a vector (use std lock mechanism)  


#include <iostream>
#include <tool_functions/ThreadPool.h>
#include <mutex>
using namespace std;

int vector_size = 3;
vector<vector<int>> vectors(vector_size);
vector<std::mutex> mtx(vector_size); // 保护vectors
void thread_function(int ID, int value)
{
    mtx[ID].lock(); // only one thread can lock mtx[ID] here, until mtx[ID] is unlocked
    vectors[ID].push_back(value);
    mtx[ID].unlock();
}
void ThreadPool_example()
{
    ThreadPool pool(5);	// use 5 threads
    std::vector<std::future<int>> results; // return typename: xxx
    for (int i = 0; i < vector_size; ++i)
    {
        for (int j = 0; j < 1e1; j++)
        {
            results.emplace_back(
                pool.enqueue([i, j] { // pass const type value j to thread; [] can be empty
                    thread_function(i, j);
                    return 1; // return to results; the return type must be the same with results
                }));
        }
    }
    for (auto &&result : results)
        result.get(); // all threads finish here
    for (int i = 0; i < vector_size; ++i)
    {
        cout << "vectors[" << i << "].size(): " << vectors[i].size() << endl;
        for (int x = 0; x < vectors[i].size(); x++)
        {
            std::cout << vectors[i][x] << ' ';
        }
        std::cout << std::endl;
    }
}
int main(){ThreadPool_example();}


*/

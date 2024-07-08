import random
import numpy as np
from gtoa import GTOA
import inspect

class Test:

    def run(self, dim=10, n=1000, lower_bound=-100, upper_bound=100, f=None, max_iter = 1000, optimal = 0):
        gtoa = GTOA(dim=dim, n=n,lower_bound=lower_bound, upper_bound=upper_bound, f=f)
        r = gtoa.run(max_iter, optimal)
        print("\n-------------------------")
        print(inspect.getsource(f))
        print("params: ", r[0], "optimal_output: ", r[1], "optimal: ", optimal, "t: ", r[2])

    def Ufun(x,a,k,m):
        return k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a))

    def f1(self):
        def f(x):
            return np.sum(x**2)
        self.run(dim=10, n=50,lower_bound=-100, upper_bound=100, f=f, max_iter=1000)

    def f2(self):
        def f(x):
            D = 10
            mul = 1
            for i in range(0, D):
                mul *= np.abs(x[i])
            return np.sum(np.abs(x)) + mul
        self.run(dim=10, n=50,lower_bound=-10, upper_bound=10, f=f, max_iter=1000)
        
    def f3(self):
        def f(x):
            D = 10
            sum = 0
            for i in range(0, D):
                sum2 = 0
                for j in range (0, i):
                    sum2 += x[j]
                sum += sum2**2
            return sum
        self.run(dim=10, n=50,lower_bound=-100, upper_bound=100, f=f, max_iter=1000)

    def f4(self):
        def f(x):
            D = 10
            return np.max(np.abs(x))
        self.run(dim=10, n=50,lower_bound=-100, upper_bound=100, f=f, max_iter=1000)

    def f5(self):
        def f(x):
            return np.max(np.abs(x))
        self.run(dim=10, n=50,lower_bound=-100, upper_bound=100, f=f, max_iter=1000)

    def f6(self):
        def f(x):
            return np.sum(np.ceil(x + 0.5)**2)
        self.run(dim=10, n=50,lower_bound=-100, upper_bound=100, f=f, max_iter=1000)

    def f7(self):
        def f(x):
            D = 10
            sum = 0
            for i in range(0, D):
                sum += i * x[i]**4 + np.random.uniform(0, 1)
            return sum
        self.run(dim=10, n=50,lower_bound=-128, upper_bound=128, f=f, max_iter=1000)

    def f8(self):
        def f(x):
            return np.sum(-x*np.sin(np.sqrt(np.abs(x)))) 
        self.run(dim=10, n=50,lower_bound=-500, upper_bound=500, f=f, max_iter=100, optimal=-418.9829*10)

    def f9(self):
        def f(x):
            D = 10
            return np.sum(x**2-10*np.cos(2*np.pi*x) + 10)
        self.run(dim=10, n=50,lower_bound=-5.12, upper_bound=5.12, f=f, max_iter=1000, optimal=0)

    def f10(self):
        def f(x):
            D = 10
            return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/D))-np.exp(np.sum(np.cos(2*np.pi*x))/D)+20+np.exp(1)
        self.run(dim=10, n=50,lower_bound=-32, upper_bound=32, f=f, max_iter=1000, optimal=0)

    def f11(self):
        def f(x):
            D = 10
            a = 1+ 1/4000*np.sum(x**2)
            b = 1
            for i in range(0, D):
                b *= np.cos(x[i]/np.sqrt(i + 1))
            return a - b
        self.run(dim=10, n=50,lower_bound=-600, upper_bound=600, f=f, max_iter=1000, optimal=0)

    def f14(self):
        def f(x):
            D = 10
            a=np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
               [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
            o = 0
            for j in range(0, 25):
                o += 1/ (j + 1 + (x[0] - a[0, j]) + (x[1] - a[1, j]) )
            return 1 / 500 + o**-1
        self.run(dim=2, n=50,lower_bound=-65, upper_bound=65, f=f, max_iter=1000, optimal=1)

        
    def f15(self):
        def f(x):
            aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
            bK = np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
            bK=1.0/bK
            return np.sum(aK - ((x[0] * (bK**2 + bK * x[1]) ) / (bK**2 + bK*x[2] + x[3]) ))
        self.run(dim=4, n=50,lower_bound=-5, upper_bound=5, f=f, max_iter=1000, optimal=0.00030)

    def f16(self):
        def f(x):
            return 4 * x[0]**2 - 2.1*x[0]**4 + 1/3 * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4
        self.run(dim=2, n=50,lower_bound=-5, upper_bound=5, f=f, max_iter=1000, optimal=-1.0316)

    def f17(self):
        def f(x):
            return (x[1] - (5.1/(4*np.pi))*x[0]**2 + (5/np.pi)*x[0] - 6)**2 + 10 * (1 - (1/(8*np.pi))) * np.cos(x[0]) + 10
        self.run(dim=2, n=50,lower_bound=-5, upper_bound=5, f=f, max_iter=1000, optimal=0.398)

    def f18(self):
        def f(x):
            return (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2 ) ) * (30 + (20*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
        self.run(dim=2, n=50,lower_bound=-2, upper_bound=2, f=f, max_iter=1000, optimal=3)

    def f19(self):
        def f(x):
            aH=np.array([[3, 10, 30],[.1, 10, 35],[3, 10, 30],[.1, 10, 35]])
            cH=np.array([1, 1.2, 3, 3.2])
            pH=np.array([[.3689, .117, .2673],[.4699, .4387, .747],[.1091, .8732, .5547],[.03815, .5743, .8828]])
            o = 0
            for i in range(0, 4):
                p = 0
                for j in range(0, 3):
                    p +=  aH[i, j] * (x[j] - pH[i, j])**2
                o += cH[i] * np.exp(-1*p)
            return -1 * o
        self.run(dim=3, n=50,lower_bound=-1, upper_bound=2, f=f, max_iter=1000, optimal=-3.86)

    def f20(self):
        def f(x):
            aH=np.array([[10, 3, 17, 3.5, 1.7, 8],[.05, 10, 17, .1, 8, 14],[3, 3.5, 1.7, 10, 17, 8],[17, 8, .05, 10, .1, 14]])
            cH=np.array([1, 1.2, 3, 3.2])
            pH=np.array([[.1312, .1696, .5569, .0124, .8283, .5886],[.2329, .4135, .8307, .3736, .1004, .9991],
                [.2348, .1415, .3522, .2883, .3047, .6650],[.4047, .8828, .8732, .5743, .1091, .0381]])
            o = 0
            for i in range(0, 4):
                p = 0
                for j in range(0, 6):
                    p +=  aH[i, j] * (x[j] - pH[i, j])**2
                o += cH[i] * np.exp(-1*p)
            return -1 * o
        self.run(dim=6, n=50,lower_bound=0, upper_bound=1, f=f, max_iter=1000, optimal=-32)

    def f21(self):
        def f(x):
            aSH=np.array([[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]])
            cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            o = 0
            for i in range(0, 4):
                o=o-((x-aSH[i,:])*(x-aSH[i,:]).transpose()+cSH[i])**(-1)
            return o.sum()
        self.run(dim=4, n=50,lower_bound=0, upper_bound=1, f=f, max_iter=1000, optimal=-10.1532)

    def f22(self):
        def f(x):
            aSH=np.array([[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]])
            cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            o = 0
            for i in range(0, 7):
                o=o-((x-aSH[i,:])*(x-aSH[i,:]).transpose()+cSH[i])**(-1)
            return o.sum() * 4
        self.run(dim=4, n=50,lower_bound=0, upper_bound=1, f=f, max_iter=1000, optimal=-10.4028)
    
    def f23(self):
        def f(x):
            aSH=np.array([[4, 4, 4, 4],[1, 1, 1, 1],[8, 8, 8, 8],[6, 6, 6, 6],[3, 7, 3, 7],[2, 9, 2, 9],[5, 5, 3, 3],[8, 1, 8, 1],[6, 2, 6, 2],[7, 3.6, 7, 3.6]])
            cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            o = 0
            for i in range(0, 10):
                o=o-((x-aSH[i,:])*(x-aSH[i,:]).transpose()+cSH[i])**(-1)
            return o.sum() * 4
        self.run(dim=4, n=50,lower_bound=0, upper_bound=1, f=f, max_iter=1000, optimal=-10.5363)


if __name__ == "__main__":
    test = Test().f16()
    
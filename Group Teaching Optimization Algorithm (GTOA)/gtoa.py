import random
import numpy as np
import math
import matplotlib.pyplot as plt
class GTOA:
    def __init__(self, dim, n,lower_bound, upper_bound, f):
        self.dim = dim
        self.n = n
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.f = f
        self.e = random.uniform(0, 1)
        self.g = random.uniform(0, 1)
        self.a = random.uniform(0, 1)
        self.b = random.uniform(0, 1)
        self.c = 1 - self.b
        self.d = random.uniform(0, 1)
        self.k = random.uniform(0, 1)
        self.F = random.choice([1, 2])
        
    def run(self, t_max, optimal):
        # Initialize population
        t_current = self.n
        t = 0
        population = self.initialize_population()

        best_solutions = list()
        best_solution_0 = self.find_best_solution(population=population)
        best_solutions.append(best_solution_0)

        if self.f(best_solution_0) == optimal:
            return best_solution_0, self.f(best_solution_0), t

        
        # Main loop
        while t < t_max:
            # Initialize groups
            outstandig_group, average_group  = self.ability_group_phase(population)


            teacher = self.teacher_allocation_phase(population)
            x_teacher_o = self.teacher_phase_i(outstandig_group, teacher)
            x_student_o = self.student_phase(x_teacher_o, outstandig_group)

            x_teacher_a = self.teacher_phase_ii(average_group, teacher)
            x_student_a = self.student_phase(x_teacher_a, average_group)

            population = self.construct_new_population(x_student_o, x_student_a)
            best_solution_t = self.find_best_solution(population)
            best_solutions.append(best_solution_t)

            if self.f(best_solution_0) == optimal:
                return best_solution, self.f(best_solution), t

            t_current += 2 * self.n + 1
            t += 1

        self.show_plot(np.arange(0, t + 1), self.all_f(np.array(best_solutions)))

        best_solution = self.find_best_solution(np.array(best_solutions))
        return best_solution, self.f(best_solution), t
    

    def show_plot(self, x, y):
        plt.plot(x, y)
        plt.xlabel('number of iterations') # X-Label
        plt.ylabel('optimal solution') # Y-Label
        plt.show()

    def initialize_population(self):
        population = np.zeros((self.n, self.dim))
        for i in range(self.n):
            population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for i in range(0, self.n):
            for j in range(0, self.dim):
                population[i, j] = population[i, :].min() + (population[i, :].max() - population[i, :].min()) * self.k
        return population
    
    def ability_group_phase(self, population):
        # print("population", population)
        population_f = self.all_f(population)
        # print("population_f", population_f)
        population_f_index = population_f.argsort()
        # print("population_f_index", population_f_index)
        outstanding_group = population[population_f_index[:int(self.n/2)],:]
        # print("outstanding_group", population_f_index[:int(self.n/2)])
        average_group = population[population_f_index[int(self.n/2):],:]
        # print("average_group", population_f_index[int(self.n/2):])
        return outstanding_group, average_group
    
    def construct_new_population(self, x, y):
        return np.concatenate((x, y) , axis=0)

    def all_f(self, x):
        x_f = np.zeros((x.shape[0], ))
        for i in range(0, x.shape[0]):
            x_f[i] = self.f(x[i, :])
        return x_f
    
    def teacher_allocation_phase(self, x):
        f_x = self.all_f(x).argsort()
        x_3 = x[f_x[0:3]].sum(axis=0) / 3
        if self.f(x_3) < self.f(x[f_x[0]]):
            return x_3
        return x[f_x[0]]
    
    def teacher_phase_i(self, x, teacher):
        M = x.mean()
        sh = x.shape
        x_teacher = np.zeros(sh)
        for i in range(0, sh[0]):
            x_teacher[i, :] = x[i, :] + self.a * (teacher - self.F * (self.b * M + self.c * x[i, :])) 
            if self.f(x_teacher[i, :]) >= self.f(x[i, :]):
                x_teacher[i, :] = x[i, :]
        x_teacher = np.minimum(x_teacher, self.upper_bound)
        x_teacher = np.maximum(x_teacher, self.lower_bound)
        return x_teacher

    def student_phase(self, x_teacher, x):
        sh = x.shape
        x_student = np.zeros(sh)
        for i in range(0, sh[0]):
             for j in range(0, sh[0]):
                if self.f(x_teacher[i, :]) < self.f(x_teacher[j, :]):
                    x_student[i, :] = x_teacher[i, :] + self.e * (x_teacher[i, :] - x_teacher[j, :]) + self.g * (x_teacher[i, :] - x[i, :])
                else:
                    x_student[i, :] = x_teacher[i, :] - self.e * (x_teacher[i, :] - x_teacher[j, :]) + self.g * (x_teacher[i, :] - x[i, :])
                if self.f(x_teacher[i, :]) < self.f(x_student[i, :]):
                    x_student[i, :] = x_teacher[i, :]
        x_student = np.minimum(x_student, self.upper_bound)
        x_student = np.maximum(x_student, self.lower_bound)
        return x_student

    def teacher_phase_ii(self, x, teacher):
        sh = x.shape
        x_teacher = np.zeros(sh)
        for i in range(0, sh[0]):
            x_teacher[i, :] = x[i, :] + 2 * self.d * (teacher - x[i, :])
            if self.f(x_teacher[i, :]) < self.f(x[i, :]):
                x_teacher[i, :] = x[i, :]
        x_teacher = np.minimum(x_teacher, self.upper_bound)
        x_teacher = np.maximum(x_teacher, self.lower_bound)
        return x_teacher
    
    def find_best_solution(self, population):
        return population[self.all_f(population).argsort()[0], :]
import random
import statistics

class LCG:
    def __init__(self, seed=1, a=1103515245, c=12345, m=2 ** 31):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m


def analyze(generator, n=100_000):
    xs = [generator() for _ in range(n)]
    mean = statistics.mean(xs)
    var_sample = statistics.variance(xs)
    var_population = statistics.pvariance(xs)
    return mean, var_sample, var_population


def main():
    N = 100_000
    seed = 123456

    lcg = LCG(seed=seed)
    mean_lcg, var_s_lcg, var_p_lcg = analyze(lcg.rand, N)

    random.seed(seed)
    mean_py, var_s_py, var_p_py = analyze(random.random, N)

    theor_mean = 0.5
    theor_var = 1.0 / 12.0

    print("Lab 4 — Random sensor comparison (N={})".format(N))
    print("Theoretical: mean={:.6f}, variance(pop)={:.6f}".format(theor_mean, theor_var))
    print()

    print("LCG generator:")
    print("  sample mean = {:.6f} (diff = {:+.6e})".format(mean_lcg, mean_lcg - theor_mean))
    print("  sample variance (unbiased) = {:.6f} (theor pop = {:.6f}, diff = {:+.6e})".format(var_s_lcg, theor_var, var_s_lcg - theor_var))
    print("  population variance = {:.6f}".format(var_p_lcg))
    print()

    print("Python built-in random:")
    print("  sample mean = {:.6f} (diff = {:+.6e})".format(mean_py, mean_py - theor_mean))
    print("  sample variance (unbiased) = {:.6f} (theor pop = {:.6f}, diff = {:+.6e})".format(var_s_py, theor_var, var_s_py - theor_var))
    print("  population variance = {:.6f}".format(var_p_py))


if __name__ == '__main__':
    main()

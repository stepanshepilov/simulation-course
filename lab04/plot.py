import matplotlib.pyplot as plt
import random

def generate_samples(gen, n):
    return [gen() for _ in range(n)]

class LCG:
    def __init__(self, seed=1, a=1103515245, c=12345, m=2 ** 31):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

def main():
    N = 100_000
    seed = 123456

    lcg = LCG(seed=seed)
    random.seed(seed)

    xs_lcg = generate_samples(lcg.rand, N)
    xs_py = generate_samples(random.random, N)

    plt.figure(figsize=(8, 4))
    bins = 100
    plt.hist(xs_py, bins=bins, alpha=0.5, label='python.random', density=True)
    plt.hist(xs_lcg, bins=bins, alpha=0.5, label='LCG', density=True)
    plt.title('Distribution comparison (N={})'.format(N))
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    out = r'/Users/sepilovstepansergeevic/Desktop/Другое/ВУЗ/sim/simulation-course/lab04/output.png'
    plt.tight_layout()
    plt.savefig(out)
    print('Saved histogram to', out)


if __name__ == '__main__':
    main()

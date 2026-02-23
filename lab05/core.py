import argparse
import random
from collections import Counter

YES_NO = ["Yes", "No"]

MAGIC_8_BALL = [
    "It is certain.",
    "It is decidedly so.",
    "Without a doubt.",
    "Yes — definitely.",
    "You may rely on it.",
    
    "As I see it, yes.",
    "Most likely.",
    "Outlook good.",
    "Yes.",
    "Signs point to yes.",
    
    "Reply hazy, try again.",
    "Ask again later.",
    "Better not tell you now.",
    "Cannot predict now.",
    "Concentrate and ask again.",
    
    "Don't count on it.",
    "My reply is no.",
    "My sources say no.",
    "Outlook not so good.",
    "Very doubtful.",
]


def yes_no_once():
    return random.choice(YES_NO)


def magic8_once():
    return random.choice(MAGIC_8_BALL)


def simulate(generator, count):
    answers = [generator() for _ in range(count)]
    freqs = Counter(answers)
    return answers, freqs


def main():
    parser = argparse.ArgumentParser(description='Lab 5: random-event apps')
    parser.add_argument('--part', type=int, choices=(1, 2), help='Which app: 1=yes/no, 2=magic 8-ball')
    parser.add_argument('--count', type=int, default=1, help='How many answers to generate (default 1)')
    args = parser.parse_args()

    if args.part is None:
        print('Choose app:')
        print('  1) Say "Yes" or "No"')
        print('  2) Magic 8-Ball')
        
        try:
            choice = int(input('Enter 1 or 2: ').strip() or '1')
        except Exception:
            choice = 1
            
    else:
        choice = args.part

    if choice == 1:
        gen = yes_no_once
        title = 'Yes/No'
    else:
        gen = magic8_once
        title = 'Magic 8-Ball'

    if args.count == 1:
        print(f'[{title}] ->', gen())
    else:
        answers, freqs = simulate(gen, args.count)
        print(f'[{title}] {args.count} draws')
        for ans, c in freqs.most_common():
            print(f'  {c:6d}  {ans}')


if __name__ == '__main__':
    main()

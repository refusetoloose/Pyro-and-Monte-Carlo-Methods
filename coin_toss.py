# %%
from collections import Counter

import pyro
import torch
from tqdm import tqdm


# %%
def coin_toss(n):
    faces = Counter()
    total = 0
    for _ in tqdm(range(n)):

        # create a sample of Bernoulli distribution for fair (50/50) coin
        # The size of the sample is 1 by default for pyro.sample
        outcome = pyro.sample("coin", pyro.distributions.Bernoulli(0.5))

        # convert the Bernoulli distribution into meaning
        # what does 1/0 stands for ?
        if outcome.item() == 1:
            result = "head"
        else:
            result = "tail"

        # gain 2 if head is tossed, otherwise lose 2
        reward = {"head": 2, "tail": -2}[result]

        # update the faces Counter
        faces[result] += 1
        total += reward

    return faces, total


def coin_toss_tensor(n):

    # create a sample of Bernoulli distribution for fair (50/50) coin
    # The size of the sample is n
    outcome = pyro.sample("coin", pyro.distributions.Bernoulli(0.5).expand([n]))

    # return Counter object to summarize the counts of head/tail, and total rewards
    heads = torch.sum(outcome).item()
    tails = n - heads
    total_reward = 2 * heads - 2 * tails
    return Counter({"head": heads, "tail": tails}), total_reward


def simulation(n, simulation_func):
    faces, total = simulation_func(n)

    print(f"\nRan {n} simulation{'s' if n >1 else ''}")
    print(f"Total Reward = {total}")
    print(faces)


# %%
for n in [1, 1000, 1000000]:
    simulation(n, coin_toss)


# %%
for n in [1, 1000, 1000000]:
    simulation(n, coin_toss_tensor)

# %%

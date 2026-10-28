"""Prompt-lookup acceptance on token files (phase 8b, D3), model-free.

    python3 -I tests/mellum_prompt_lookup.py FILE.tokens ...

A prompt-lookup draft at position i takes the last n ids before i (n from NGRAM_MAX down to NGRAM_MIN), finds
their most recent earlier occurrence, and proposes the k ids that followed it.  Here the "model" is the file:
a draft id is accepted when it equals the file's own next id.  So this measures how often the text repeats
its own context, teacher-forced; it is not the model's greedy agreement, which needs the model.

Two summaries per file, for each (NGRAM_MIN, NGRAM_MAX):
  depth 1, every position: how many positions have a draft at all, and how many of those drafts are right
          (the case a two-row verify, B0, serves);
  draft k, walked: from the first position, each step drafts up to k ids when it can, verifies them in one
          (k + 1)-row step, keeps the agreeing prefix plus one, and moves on; without a draft it is an ordinary
          one-row step.  Reported as ids per step and the share of steps that verified a draft.
"""

import sys


def read_tokens(path):
    ids = []
    for line in open(path):
        if not line.startswith("#"):
            ids.extend(int(x) for x in line.split())
    return ids


class Lookup:
    """For each n, the most recent start of every n-gram seen so far."""

    def __init__(self, ids, nmin, nmax):
        self.ids, self.nmin, self.nmax = ids, nmin, nmax
        self.last = {n: {} for n in range(nmin, nmax + 1)}
        self.upto = 0

    def advance(self, i):
        # index every n-gram that ends before position i - 1, so a match is always strictly earlier
        while self.upto < i - 1:
            end = self.upto + 1
            for n in range(self.nmin, self.nmax + 1):
                if end >= n:
                    self.last[n][tuple(self.ids[end - n:end])] = end
            self.upto += 1

    def draft(self, i, k):
        self.advance(i)
        for n in range(self.nmax, self.nmin - 1, -1):
            if i < n:
                continue
            j = self.last[n].get(tuple(self.ids[i - n:i]))
            if j is not None:
                return self.ids[j:min(j + k, i)]
        return []


def depth1(ids, nmin, nmax):
    lk = Lookup(ids, nmin, nmax)
    drafted = right = 0
    for i in range(1, len(ids)):
        d = lk.draft(i, 1)
        if d:
            drafted += 1
            right += d[0] == ids[i]
    return len(ids) - 1, drafted, right


def walk(ids, nmin, nmax, k):
    lk = Lookup(ids, nmin, nmax)
    i, steps, verify_steps = 1, 0, 0
    while i < len(ids):
        d = lk.draft(i, k)
        steps += 1
        if d:
            verify_steps += 1
            m = 0
            while m < len(d) and i + m < len(ids) and d[m] == ids[i + m]:
                m += 1
            i += m + 1
        else:
            i += 1
    return len(ids) - 1, steps, verify_steps


def main(paths):
    for path in paths:
        ids = read_tokens(path)
        print("%s: %d ids" % (path, len(ids)))
        for nmin, nmax in ((1, 3), (2, 3), (3, 3)):
            n, drafted, right = depth1(ids, nmin, nmax)
            print("  ngram %d-%d  depth 1: %d positions, %d drafted (%.1f%%), %d right (%.1f%% of drafts, "
                  "%.1f%% of positions)" % (nmin, nmax, n, drafted, 100 * drafted / n, right,
                                            100 * right / drafted if drafted else 0, 100 * right / n))
            for k in (1, 2, 4, 8):
                n, steps, vsteps = walk(ids, nmin, nmax, k)
                print("               draft %d: %d ids in %d steps, %.3f ids per step, %d steps verified "
                      "a draft (%.1f%%)" % (k, n, steps, n / steps, vsteps, 100 * vsteps / steps))


if __name__ == "__main__":
    main(sys.argv[1:])

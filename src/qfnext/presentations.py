"""Finite presentations, not a general fundamental-group recognition engine."""
from dataclasses import dataclass


def free_reduce(word):
    out = []
    for x in word:
        x = int(x)
        if not x:
            raise ValueError('zero is not a generator')
        if out and out[-1] == -x:
            out.pop()
        else:
            out.append(x)
    return tuple(out)


def inverse(word):
    return tuple(-x for x in reversed(word))


def cyclic_reduce(word):
    w = free_reduce(word)
    i, j = 0, len(w)
    while j - i >= 2 and w[i] == -w[j - 1]:
        i += 1
        j -= 1
    return w[i:j]


def substitute(word, generator, replacement, max_length):
    out = []
    for x in word:
        part = replacement if x == generator else inverse(replacement) if x == -generator else (x,)
        if len(out) + len(part) > max_length:
            raise RuntimeError('Tietze word-length limit reached')
        for y in part:
            if out and out[-1] == -y:
                out.pop()
            else:
                out.append(y)
    return tuple(out)


@dataclass(frozen=True)
class GroupPresentation:
    root: int
    edge_generators: tuple
    relators: tuple
    tree_edges: tuple
    relator_cells: tuple

    def simplify(self, *, max_word_length=100000, max_eliminations=10000):
        """Tietze eliminations of generators that occur once in a relator.

        Returns original generator indices, relators and a replayable elimination
        log. The result need not be minimal or canonical.
        """
        if max_word_length < 1 or max_eliminations < 0:
            raise ValueError('invalid simplification limit')
        generators = list(range(1, len(self.edge_generators) + 1))
        relations = [cyclic_reduce(w) for w in self.relators if cyclic_reduce(w)]
        log = []
        while len(log) < max_eliminations:
            candidates = []
            for i, w in enumerate(relations):
                for g in sorted(set(map(abs, w))):
                    if sum(abs(x) == g for x in w) == 1:
                        candidates.append((len(w), i, g))
            if not candidates:
                break
            _, i, g = min(candidates)
            word = relations[i]
            j = next(j for j, x in enumerate(word) if abs(x) == g)
            rotated = word[j:] + word[:j]
            replacement = inverse(rotated[1:]) if rotated[0] > 0 else rotated[1:]
            new = [cyclic_reduce(substitute(w, g, replacement, max_word_length))
                   for k, w in enumerate(relations) if k != i]
            relations = [w for w in new if w]
            generators.remove(g)
            log.append((g, replacement, word))
        return {'generators': tuple(generators), 'relators': tuple(relations),
                'eliminations': tuple(log), 'complete_single_occurrence_pass':
                not any(sum(abs(x) == g for x in w) == 1
                        for w in relations for g in set(map(abs, w)))}

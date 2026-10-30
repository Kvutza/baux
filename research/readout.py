#!/usr/bin/env python3
"""Smooth hierarchical readout VJP experiment; no GPU or full-model parity claim.

The current fbt_wave.metal projects h by sqrt(8) Q, then uses six levels
of eight-way softmax. Every internal node is sum(leaves)/sqrt(leaf count).
Here radix and depth are variable so exhaustive derivative checks are cheap.
FP16 casts are omitted: this differentiates the smooth precursor, not rounding.
"""

from dataclasses import dataclass
from math import exp, fsum, isclose, log, sqrt
from random import Random


@dataclass(frozen=True)
class Shape:
    radix: int
    levels: int
    width: int = 3
    query: int = 2

    @property
    def vocabulary(self):
        return self.radix**self.levels

    @property
    def embedding(self):
        return self.width + self.width * self.query

    @property
    def size(self):
        return self.embedding + self.query * self.vocabulary


def project(values, shape):
    return [
        sqrt(8)
        * fsum(
            values[row] * values[shape.width + row * shape.query + dim]
            for row in range(shape.width)
        )
        for dim in range(shape.query)
    ]


def node(values, shape, first, count):
    return [
        fsum(
            values[shape.embedding + dim * shape.vocabulary + token]
            for token in range(first, first + count)
        )
        / sqrt(count)
        for dim in range(shape.query)
    ]


def dot(left, right):
    return fsum(a * b for a, b in zip(left, right, strict=True))


def siblings(values, shape, target, level):
    count = shape.radix ** (shape.levels - level - 1)
    parent = target // (shape.radix * count)
    first = parent * shape.radix * count
    children = [
        node(values, shape, first + child * count, count)
        for child in range(shape.radix)
    ]
    return children, first, count


def softmax(logits):
    peak = max(logits)
    weights = [exp(value - peak) for value in logits]
    total = fsum(weights)
    return [weight / total for weight in weights], log(total) + peak


def loss(values, shape, target):
    """Direct range sums: every centroid is recomputed after perturbation."""
    query = project(values, shape)
    result = 0.0
    for level in range(shape.levels):
        children, _, count = siblings(values, shape, target, level)
        logits = [dot(query, child) for child in children]
        _, normalizer = softmax(logits)
        result += normalizer - logits[target // count % shape.radix]
    return result


def scatter(delta, first, count, coefficient, query):
    """Range adjoint: distribute a tied centroid gradient to all its leaves."""
    for dim, value in enumerate(query):
        contribution = coefficient * value / sqrt(count)
        delta[dim][first] += contribution
        delta[dim][first + count] -= contribution


def gradients(values, shape, target):
    query = project(values, shape)
    query_grad = [0.0] * shape.query
    delta = [[0.0] * (shape.vocabulary + 1) for _ in range(shape.query)]
    for level in range(shape.levels):
        children, first, count = siblings(values, shape, target, level)
        probabilities, _ = softmax([dot(query, child) for child in children])
        selected = target // count % shape.radix
        for child, probability in enumerate(probabilities):
            coefficient = probability - (child == selected)
            for dim in range(shape.query):
                query_grad[dim] += coefficient * children[child][dim]
            scatter(delta, first + child * count, count, coefficient, query)
    return pullback(values, shape, query_grad, delta)


def pullback(values, shape, query_grad, delta):
    result = [0.0] * shape.size
    for row in range(shape.width):
        for dim in range(shape.query):
            index = shape.width + row * shape.query + dim
            result[row] += sqrt(8) * values[index] * query_grad[dim]
            result[index] = sqrt(8) * values[row] * query_grad[dim]
    for dim in range(shape.query):
        running = 0.0
        for token in range(shape.vocabulary):
            running += delta[dim][token]
            result[shape.embedding + dim * shape.vocabulary + token] = running
    return result


def close(actual, expected, tolerance=2e-7):
    if not isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual:.12g} != {expected:.12g}")


def shifted(values, direction, step):
    return [
        value + step * change for value, change in zip(values, direction, strict=True)
    ]


def difference(values, shape, target, direction, step):
    return (
        loss(shifted(values, direction, step), shape, target)
        - loss(shifted(values, direction, -step), shape, target)
    ) / (2 * step)


def directional(values, shape, target, direction):
    analytic = dot(gradients(values, shape, target), direction)
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        estimate = difference(values, shape, target, direction, step)
        errors.append(abs(estimate - analytic))
    close(estimate, analytic)
    if errors[0] > 1e-8 and errors[1] > errors[0] * 0.1:
        raise AssertionError(f"central differences did not converge: {errors}")
    return errors[-1]


def fixture(shape, seed):
    rng = Random(seed)
    return [rng.uniform(-0.2, 0.2) for _ in range(shape.size)]


def coordinates(shape):
    values = fixture(shape, 73)
    worst = 0.0
    checks = 0
    for target in (0, shape.vocabulary // 2, shape.vocabulary - 1):
        for index in range(shape.size):
            direction = [0.0] * shape.size
            direction[index] = 1.0
            worst = max(worst, directional(values, shape, target, direction))
            checks += 1
    return checks, worst


def family_checks(shape):
    values = fixture(shape, 19)
    rng = Random(41)
    boundaries = (0, shape.width, shape.embedding, shape.size)
    worst = 0.0
    for first, end in zip(boundaries, boundaries[1:]):
        direction = [0.0] * shape.size
        for index in range(first, end):
            direction[index] = rng.uniform(-1, 1) / sqrt(end - first)
        worst = max(worst, directional(values, shape, shape.vocabulary - 1, direction))
    return worst


def invariants():
    shape = Shape(2, 3)
    values = fixture(shape, 7)
    close(fsum(exp(-loss(values, shape, t)) for t in range(shape.vocabulary)), 1)
    for index in range(shape.width, shape.embedding):
        values[index] = 0.0
    close(loss(values, shape, 0), log(shape.vocabulary))
    gradient = gradients(values, shape, 0)
    for value in gradient[shape.embedding :]:
        close(value, 0)
    if max(abs(value) for value in gradient[shape.width : shape.embedding]) < 1e-6:
        raise AssertionError("zero query must still propagate to projection parameters")


def centroid_check():
    shape = Shape(2, 3, width=1, query=1)
    values = [1.0, 0.5] + [0.1 * token for token in range(shape.vocabulary)]
    # This leaf is outside the target's final siblings. Its influence comes
    # exclusively from an internal centroid; a detached-centroid VJP gives zero.
    index = shape.embedding + shape.vocabulary - 1
    direction = [0.0] * shape.size
    direction[index] = 1.0
    derivative = gradients(values, shape, 0)[index]
    if abs(derivative) < 1e-3:
        raise AssertionError("fixture did not expose the tied-centroid derivative")
    directional(values, shape, 0, direction)
    return derivative


def run():
    invariants()
    checks, worst = coordinates(Shape(2, 3))
    other, error = coordinates(Shape(8, 2))
    worst = max(worst, error, family_checks(Shape(8, 6, width=2, query=1)))
    centroid = centroid_check()
    print(f"coordinate_checks={checks + other} central_difference_steps=3")
    print("production_radix=8 production_depth=6 reduced_width=2 reduced_query=1")
    print(f"max_derivative_error={worst:.3e} off_path_leaf_derivative={centroid:.9f}")
    print("scope=smooth-readout-only gpu_parity=false full_model_parity=false")


if __name__ == "__main__":
    run()

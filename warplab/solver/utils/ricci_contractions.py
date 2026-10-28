"""Reusable inverse-metric contractions for the Ricci quadratic terms."""


def raised_derivatives(inverse, derivative):
    """Contract the d index once, outside the output-component loops.

    Each cache holds 64 grid arrays. Keeping these intermediates avoids
    repeatedly dispatching the same small operations to a device.
    """
    raised = [[[None] * 4 for _ in range(4)] for _ in range(4)]
    skew = [[[None] * 4 for _ in range(4)] for _ in range(4)]
    connection = [[[None] * 4 for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for r in range(4):
                for d in range(4):
                    first = derivative[b][d][a] * inverse[r][d]
                    second = (derivative[a][d][b] - derivative[a][b][d]) * inverse[r][d]
                    third = (2 * derivative[b][d][a] - derivative[a][b][d]) * inverse[
                        r
                    ][d]
                    if d == 0:
                        raised[a][b][r] = first
                        skew[a][b][r] = second
                        connection[a][b][r] = third
                    else:
                        raised[a][b][r] = raised[a][b][r] + first
                        skew[a][b][r] = skew[a][b][r] + second
                        connection[a][b][r] = connection[a][b][r] + third
    return raised, skew, connection

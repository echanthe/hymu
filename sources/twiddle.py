"""Implementation of the twiddle algorithm
"""

def twiddle(func, x, dx, maxIteration, derror):
    """ Twiddle is search-based method
    to find best parameters
    """
    i = 0
    be = abs(func(x))
    while i < maxIteration and sum(dx) > derror:
        i += 1
        print('{}/{} and {}/{}'.format(i, maxIteration, sum(dx), derror))

        for j in range(len(x)):
            x[j] += dx[j]
            e = func(x)

            if abs(e) < be:
                be = abs(e)
                dx[j] *= 1.1
                print('  ', x, e)
            else:
                x[j] -= 2*dx[j]
                e = func(x)

                if abs(e) < be:
                    be = abs(e)
                    dx[j] *= 1.1
                    print('  ', x, e)
                else:
                    x[j] += dx[j]
                    dx[j] *= .9

    print('best x: {}'.format(x))
    print('error: {}'.format(be))
    print('iteration number: {}'.format(i))
    print('derror: {}'.format(sum(dx)))
    return x, be, i, derror


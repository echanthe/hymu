""" This module provide an optimization decorator for function that compute independant calculations.
Distributes the independant calculations on the machine processors if the number of calculations exceed a given value.

FIXME: optimization is not that term here. Can be confuse with numerical optimization. Should be just distribution or something.
"""

import multiprocessing
from functools import wraps

PROCESSOR_NUMBER = multiprocessing.cpu_count() # set to 1 to disable
#PROCESSOR_NUMBER = 1

def optimization(minimum, attr):

    """
    >>> class Foo:
    ...     def __init__(self, t):
    ...         self.t = t
    >>> lt = [Foo(i) for i in range(40)]
    >>> 
    >>> @optimization(50, 't')
    ... def test(l):
    ...     for ll in l:
    ...         ll.t *= 1000
    ...     return [ll.t for t in l]
    >>> test(lt)
    >>> [l.t for l in lt] == [i*1000 for i in range(40)]
    True
    >>>
    >>> lt = [Foo(i) for i in range(100)]
    >>> 
    >>> @optimization(50, 't')
    ... def test(l):
    ...     for ll in l:
    ...         ll.t *= 1000
    ...     return [ll.t for ll in l]
    >>> test(lt)
    >>> [l.t for l in lt] == [i*1000 for i in range(100)]
    True
    >>>
    >>> lt = [Foo(i) for i in range(100)]
    >>> 
    >>> @optimization(50, 't')
    ... def test(l, plus):
    ...     for ll in l:
    ...         ll.t += plus
    ...     return [ll.t for ll in l]
    >>> test(lt, 50)
    >>> [l.t for l in lt] == [i + 50 for i in range(100)]
    True
    """
    def opti_deco(function):

        @wraps(function)
        def opti_func(*args, **kwargs):
            elmts = args[0]

            if len(elmts) > minimum and PROCESSOR_NUMBER > 1:
                # optimization

                jobs = []
                manager = multiprocessing.Manager()
                values = manager.dict()

                for i in range(PROCESSOR_NUMBER):
                    sub_elmts = elmts[i::PROCESSOR_NUMBER]
                    def job_func(elmts, val, arguments):
                        a = list(arguments)
                        a[0] = elmts
                        a = tuple(a)
                        val[i] = function(*a)
                    p = multiprocessing.Process(target=job_func, args=(sub_elmts, values, args))
                    jobs.append(p)
                    p.start()

                for p in jobs:
                    p.join()
            
                #results = []
                #while len(results) < PROCESSOR_NUMBER:
                #    for i in range(PROCESSOR_NUMBER):
                #        if values.get(i) and i not in results:
                #            sub_elmts = elmts[i::PROCESSOR_NUMBER]
                #            for e, v in zip(sub_elmts, values[i]):
                #                setattr(e, attr, v)
                #            results.append(i)

                for i in range(PROCESSOR_NUMBER):
                    sub_elmts = elmts[i::PROCESSOR_NUMBER]
                    for e, v in zip(sub_elmts, values[i]):
                        setattr(e, attr, v)
            else:
                function(*args)

        return opti_func
    return opti_deco

if __name__ == "__main__":
    import doctest
    doctest.testmod()

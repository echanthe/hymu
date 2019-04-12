from hppn import *

""" This class represent HPPN Multimode
    The trick is to reprensent the multimode using HPPN symbolic places
"""
class Multimode(HPPN):

    def __init__(self, model):
        """ Retrieve multi-mode system of the net

        >>> hppn = test.simple_model()
        >>> multi = Multimode(hppn)
        >>> len(list(multi.place()))
        8
        """
        super().__init__(model.name + ' multimode')
        modeMap = {}
        for mode in model.modes():
            name = mode.name + '\\n({}, {}, {})'.format(mode.ps, mode.pn, mode.ph)
            m = SymbolicPlace(name)
            self.add_place(m)
            modeMap[mode] = m
            if hasattr(mode.ps, 'color'):
                # patch to draw correct color
                m.color = mode.ps.color

        for t in model.transition():
            mi = model.get_transition_input_mode(t)
            mo = model.get_transition_output_mode(t)
            pi = modeMap[mi]
            po = modeMap[mo]
            tCopy = t.copy() 
            self.add_transition([pi], tCopy, [po])

if __name__ == "__main__":
    import doctest
    import test
    import logging
    logging.disable(logging.INFO)
    doctest.testmod()

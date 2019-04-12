
""" This module contains a library to manipulate advanced Petri Nets
"""


import snakes.plugins
import snakes.pnml
snakes.plugins.load("gv", "snakes.nets", "nets")
from nets import *

'''import snakes
from snakes import *'''
from itertools import chain

class SpecialToken(Token):
    """ This class represents a special kind of tokens 
    These tokens are linked with the place they currently belong
    SpecialToken are not real token in the net, ther are objects
    They are carried by GroupToken
    """

    def __init__(self, value = 0, place = None):
        """ Create a SpecialToken

        >>> s = SpecialToken(5)
        >>> s.value
        5
        >>> s.place
        >>> s = SpecialToken([])
        >>> s.value
        []
        >>> s.place
        >>> import numpy
        >>> s = SpecialToken(numpy.array([5, 5]))
        >>> s.value
        array([5, 5])
        >>> s.place
        """
        self.place = place
        super().__init__(value)

    def __str__(self):
        """ Return a string representation of a SpecialToken
        The object ID of the token is given followed by all its attributes value
        """
        return "({}, {}, {})".format(id(self), self.value, self.place.name)

    def __repr__(self):
        """ Return the object ID string of a SpecialToken
        """
        return "({})".format(id(self))

    def __eq__(self, other):
        """ Evaluate if thwo SpecialToken are equal
        Two SpecialToken are equal if they have the same object ID
        
        >>> a = SpecialToken(5); b=SpecialToken(5)
        >>> a == a
        True
        >>> b == b
        True
        >>> b == a
        False
        """
        return id(self) == id(other)
    
    def __hash__(self):
        """ Return the hash value of a SpecialToken
        The hash value of a SpecialToken is its object ID
        
        >>> a = SpecialToken(5)
        >>> hash(a) == id(a)
        True
        """
        return id(self) 

    def copy(self):
        """ Should be implemented depending on the SpecialToken value
        
        >>> a = SpecialToken(5)
        >>> a.copy()
        Traceback (most recent call last):
        ...
        NotImplementedError
        """
        raise NotImplementedError

class GroupToken(Token):
    """ This class represents a special kind of tokens used to hack petri net 
    It is use to move an undetermined number of tokens when firing a transition
    """

    accepted_tag = 'accepted'
    refused_tag = 'refused'

    def __init__(self, tag, tokens = []):
        """ Create a GroupToken

        >>> g = GroupToken(GroupToken.accepted_tag, [SpecialToken()])
        >>> len(g.tokens) == 1
        True
        >>> g.tag == 'accepted'
        True
        """

        self.tag = tag
        self.tokens = tokens
        Token.__init__(self, self.tag)

    def __str__(self):
        """ Return the string representation of the GroupToken

        """
        return "({}:({}, {})".format(id(self),self.tag, ", ".join(str(x) for x in self.tokens))

    def __eq__(self, other):
        """ Return True if both GroupToken have the same object ID

        >>> g1 = GroupToken(GroupToken.accepted_tag)
        >>> g2 = GroupToken(GroupToken.accepted_tag)
        >>> g1 == g2
        False
        >>> g1 == g1
        True
        """
        return id(self) == id(other)

    def __hash__(self):
        """ Return the hash value of a GroupToken
        The hash value of a GroupToken is its object ID
        
        >>> g1 = GroupToken(GroupToken.accepted_tag)
        >>> g2 = GroupToken(GroupToken.accepted_tag)
        >>> hash(g1) == hash(g2)
        False
        >>> hash(g1) == hash(g1)
        True
        """
        return id(self) 

class SpecialPlace(Place):
    """ This class represents a special kind of places 
    These places are linked with their tokens
    """

    def __init__(self, name):
        """ Create a SpecialPlace

        >>> s = SpecialPlace('foo de la-vega()')
        >>> s.name
        'foo de la-vega()'
        >>> s.varname
        'foo_de_la_vega__'
        >>> s.tokens
        MultiSet([])
        """
        Place.__init__(self, name)

        n = name
        badChar = " +-()'\/[]{},;:!?."
        for i in badChar: n = n.replace(i, '_')
        # is used for Arc Variable
        self.varname = n

    def add_tokens(self, tokens):
        """ Add a SpecialToken or a group of SpecialToken to the SpecialPlace
        Added SpecialToken are then linked to the SpecialPlace
        
        >>> p = SpecialPlace('foo')
        >>> p.add_tokens([SpecialToken(0) for i in range(4)])
        >>> all((t.place == p for t in p.tokens))
        True
        """
        for t in tokens: t.place = self
        super().add(tokens)

    def remove_tokens(self, tokens):
        """ Remove the given SpecialToken or group of SpecialToken from the SpecialPlace
        Removed SpecialToken are unlinked from the SpecialPlace
        
        >>> p = SpecialPlace('foo')
        >>> to = SpecialToken(5)
        >>> to2 = SpecialToken(5)
        >>> p.add_tokens([to,to2])
        >>> len(p.tokens) == 2
        True
        >>> p.remove_tokens([to, to2])
        >>> len(p.tokens) == 0
        True
        """
        for t in tokens: 
            t.place = None
            if t in list(self.tokens):
                super().remove([t])
        #super().remove(tokens)

    def empty(self):
        """ Empty the SpecialPlace
        Remove all tokens from the SpecialPlace
        Removed SpecialToken are unlinked from the SpecialPlace

        >>> p = SpecialPlace('foo')
        >>> t1, t2 = SpecialToken(5), SpecialToken(6)
        >>> p.add_tokens([t1, t2])
        >>> p.empty()
        >>> p.tokens == []
        True
        >>> all((t.place is None for t in [t1, t2]))
        True
        """
        for t in self.tokens: 
            if isinstance(t, SpecialToken): t.place = None
        super().empty()

    def reset_tokens(self, tokens):
        """ Reset the SpecialPlace with the given SpecialToken or group of SpecialToken
        Removed SpecialToken are unlinked from the SpecialPlace
        Added SpecialToken are linked to the SpecialPlace

        >>> p = SpecialPlace('foo')
        >>> t1, t2, t3 = SpecialToken(5), SpecialToken(6), SpecialToken(6)
        >>> p.reset_tokens([t2, t3])
        >>> t1.place is None
        True
        >>> all((t.place == p for t in [t2, t3]))
        True
        >>> len(p.tokens) == 2
        True
        >>> groups = []
        >>> for i in range(5) : groups.append(GroupToken("fooToken", [t1,t2,t3]))
        >>> p.reset(groups)
        >>> len(p.tokens) == len(set(p.tokens))
        True
        >>> len(set(groups)) == len(set(p.tokens))
        True
        """
        # first remove all tokens from place to force to unlink
        self.empty()
        self.add_tokens(tokens)

    def group(self, groupsList):
        """ Group accepted tokens by creating a GroupToken for each given list of token
        If the given tokens are not in the SpecialPlace, they are still added in a group

        >>> tokens = [SpecialToken() for i in range(5)]
        >>> p = SpecialPlace('foo')
        >>> p.add_tokens(tokens)
        >>> p.group({'foo1': tokens[0:2], 'foo': tokens[2:3]})
        >>> len(p.tokens)
        4
        >>> len(p.tokens) == len(set(p.tokens))
        True
        >>> sum(1 for g in p.tokens if isinstance(g, GroupToken) and g.tag.endswith(GroupToken.accepted_tag)) == 2
        True
        >>> sum(len(g.tokens) for g in p.tokens if isinstance(g, GroupToken) and g.tag.endswith(GroupToken.accepted_tag)) == 3
        True
        """
        groups = []
        for groupName, tokens in groupsList.items():
            self.remove_tokens(tokens)
            g = GroupToken(groupName + " " + GroupToken.accepted_tag, tokens)
            groups.append(g)

        groups.extend(self.tokens)
        self.reset(groups)

    def flat(self):
        """ Ungroup tokens by flatting all GroupToken in the SpecialPlace
        SpecialToken contained in GroupToken are moved in the SpecialToken and GroupToken are deleted

        >>> tokens = [SpecialToken() for i in range(5)]
        >>> p = SpecialPlace('foo')
        >>> p.add_tokens(tokens)
        >>> p.group({'foo1': tokens[0:2], 'foo2': tokens[2:3]})
        >>> p.flat()
        >>> len(p.tokens) == 5
        True
        >>> len(p.tokens) == len(set(p.tokens))
        True
        >>> all((isinstance(t, SpecialToken) for t in p.tokens))
        True
        """
        tokens = []
        for tok in self:
            if isinstance(tok, GroupToken):
                tokens += tok.tokens
            else:
                tokens.append(tok)
        self.reset_tokens(tokens)

    def get_group_tokens(self, variable):
        tag = '{} {}'.format(str(variable), GroupToken.accepted_tag)
        for g in self.tokens:
            if isinstance(g, GroupToken) and g.tag == tag:
                return g
        return None

class SpecialTransition (Transition):
    """ This class represents a special kind of transition
    """

    def __init__(self, name):
        """ Create a SpecialTransition with the given name

        >>> SpecialTransition('foo')
        SpecialTransition('foo', Expression('True'))
        """
        Transition.__init__(self, name)

    def fire(self):
        """
        >>> p = PetriNet('foo')
        >>> tok1 = [SpecialToken(i) for i in range(5)]
        >>> tok2 = [SpecialToken(i) for i in range(3)]
        >>> p1 = SpecialPlace('p1')
        >>> p1.add_tokens(tok1)
        >>> p.add_place(p1)
        >>> p2 = SpecialPlace('p2')
        >>> p2.add_tokens(tok2)
        >>> p.add_place(p2)
        >>> tr = SpecialTransition('tr')
        >>> p.add_transition(tr)
        >>> p.add_input('p1', 'tr', Variable('x'))
        >>> p.add_output('p2', 'tr', Expression('x'))
        >>> tr._createConditionExpression()
        >>> p1.group({'x': tok1[0:3]})
        >>> tr.accept()
        >>> tr.fire()
        >>> tr.flat()
        >>> len(p1.tokens) == 2
        True
        >>> len(p2.tokens) == 6
        True
        """
        fmode = []
        for (p,a) in self.input():
            for v in a.vars():
                fmode.append((v, p.get_group_tokens(v)))
        fmode = Substitution(fmode)
        super().fire(fmode)
    
    def accept(self):
        """ This method has to be implement when implementing a SpecialTransition
        It should use the SpecialPlace.group() function to select which SpecialToken have to move during the firing of the SpecialTransition
        """
        if __name__ != "__main__":
            raise NotImplementedError

    def flat(self):
        """ Flat all input and output SpecialPlace of the SpecialTransition
        
        >>> p = PetriNet('foo')
        >>> tok1 = [SpecialToken(i) for i in range(5)]
        >>> tok2 = [SpecialToken(i) for i in range(3)]
        >>> p1 = SpecialPlace('p1')
        >>> p1.add_tokens(tok1)
        >>> p.add_place(p1)
        >>> p2 = SpecialPlace('p2')
        >>> p2.add_tokens(tok2)
        >>> p.add_place(p2)
        >>> tr = SpecialTransition('tr')
        >>> p.add_transition(tr)
        >>> p.add_input('p1', 'tr', Variable('x'))
        >>> p.add_output('p2', 'tr', Expression('x'))
        >>> p1.group({'p1_p2' : tok1[0:3]})
        >>> p2.group({'p2' : tok2[0:1]})
        >>> len(p1.tokens)
        3
        >>> len(p2.tokens)
        3
        >>> tr.flat()
        >>> len(p1.tokens) == 5
        True
        >>> len(p2.tokens) == 3
        True
        """
        #s=""
        for (p, a) in self.output():
            p.flat()
        #    s+= "{} ".format(p)
        for (p, a) in self.input():
            p.flat()
        #    s+= "{} ".format(p)

    def _createConditionExpression(self):
        """ Create the SpecialTransition expression condition for the next firing

        >>> p = PetriNet('foo')
        >>> p1 = SpecialPlace('p1')
        >>> p.add_place(p1)
        >>> p2 = SpecialPlace('p2')
        >>> p.add_place(p2)
        >>> tr = SpecialTransition('tr')
        >>> p.add_transition(tr)
        >>> p.add_input('p1', 'tr', Variable('x'))
        >>> p.add_output('p2', 'tr', Expression('x'))
        >>> tr._createConditionExpression()
        >>> tr.guard == Expression("x.tag=='x accepted'")
        True
        >>> p.remove_transition('tr')
        >>> tr2 = SpecialTransition('tr2')
        >>> p.add_transition(tr2)
        >>> p.add_input('p1', 'tr2', MultiArc([Variable("x"), Variable("y")]))
        >>> p.add_output('p2', 'tr2', Expression('x'))
        >>> tr2._createConditionExpression()
        >>> v1 = tr2.guard == Expression("x.tag=='x accepted' and y.tag=='y accepted'")
        >>> v2 = tr2.guard == Expression("y.tag=='y accepted' and x.tag=='x accepted'")
        >>> v1 or v2
        True
        """

        variables = set()
        for (p,a) in self.input():
            for v in a.vars():
                variables.add(v)
        textual_expression = " and ".join(v + ".tag=='" + v + " " + GroupToken.accepted_tag + "'" for v in variables)
        #print(textual_expression.format(GroupToken.accepted_tag))
        try : 
            expression = Expression(textual_expression)
            self.guard = expression
        except: pass 

if __name__ == "__main__":
    import doctest
    doctest.testmod()

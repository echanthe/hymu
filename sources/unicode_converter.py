subs = {u'0': u'\u2080',
        u'1': u'\u2081',
        u'2': u'\u2082',
        u'3': u'\u2083',
        u'4': u'\u2084',
        u'5': u'\u2085',
        u'6': u'\u2086',
        u'7': u'\u2087',
        u'8': u'\u2088',
        u'9': u'\u2089'}

def to_sub(s):
    return ''.join(subs.get(char, char) for char in s)

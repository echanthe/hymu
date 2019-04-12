""" Sandbox with a lot a display fonctions (raw data, diagnosis and prognosis results).
These fonctions generate text, images or videos
"""

from hppn import *  #SymbolicPlace, NumericalPlace
import os, os.path, subprocess, collections
from unicode_converter import to_sub
from multimode import * #Multimode
from diagnoser import Diagnoser

green = ('#00ff00', '#b3ffb3')
orange = ('#ff8000', '#ff9933')
red = ('#ff0000', '#ff3333')
blue = ('#0000ff', '#e6e6ff')
grey = ('#000000', '#e6e6e6')
white = '#ffffff'

def draw_place (hppn) :
    if isinstance(hppn, Multimode):
        return _draw_place('ellipse')
    else:
        return _draw_place('circle')

def _draw_place (shape) :
    def f (place, attr) :
        attr['label'] = to_sub(place.name)
        attr['shape'] = shape
        if shape == 'circle':
            attr['fixedsize'] = 'shape'
        if len(place.tokens) > 0:
            attr['label'] += '\n' + str(len(place.tokens))
        if isinstance(place, SymbolicPlace):
            if hasattr(place, 'color'):
                color = place.color
            else:
                color = orange
            attr['color'] = color[0]
            attr['fillcolor'] = color[1]
            if color == red:
                attr['fontcolor'] = white
                attr['style'] = "filled, bold"
        elif isinstance(place, NumericalPlace):
            attr['color'] = blue[0] 
            attr['fillcolor'] = blue[1]
            #attr['penwidth'] = 2
            attr['style'] = "dashed, filled"
        else:
            attr['color'] = grey[0] 
            attr['fillcolor'] = grey[1]
            attr['penwidth'] = 1.5
            attr['style'] = "dotted,filled"
    return f

def draw_transition (hppn) :
    if isinstance(hppn, Multimode):
        return _draw_transition('and')
    elif isinstance(hppn, Diagnoser):
        return _draw_transition('⟨ {}, {} ⟩')
    else:
        return _draw_transition('⟨ {}, {}, {} ⟩')

def _draw_transition (format_str) :
    def f (trans, attr) :
        diagnoser = False
        if len(list(trans.input())) == 1:
            diagnoser = isinstance(next(p for p,_ in trans.input()), HybridPlace)
            
        attr['label'] = to_sub(trans.name) + '\n'
        if not diagnoser:
            cs = trans.cs.__name__ 
            cn = trans.cn.__name__
            ch = trans.ch.__name__
            cs = [cs,cn,ch]
            for i,c in enumerate(cs):
                if c == 'false' : cs[i] = '⊥'
                if c == 'true' : cs[i] = '⊤'
            cs, cn, ch = cs
                
            if format_str != 'and' :
                attr['label'] += format_str.format(cs, cn, ch)
            else :
                if cs != '⊤' and cn != '⊤':
                    attr['label'] += '{} and {}'.format(cs, cn)
                elif cs != '⊤': 
                    attr['label'] += '{}'.format(cs)
                elif cn != '⊤': 
                    attr['label'] += '{}'.format(cn)

    return f

def draw_arc (hppn) :
    def f (arc, attr) :
        if isinstance(arc.place, NumericalPlace):
            attr['color'] = blue[0] 
            attr['style'] = "dashed"
        elif isinstance(arc.place, HybridPlace):
            attr['color'] = grey[0] 
            attr['style'] = "dotted"
            #attr['penwidth'] = 1.5
        attr['label'] = ''
    return f
    #if hasattr(arc, 'events'):
    #    attr['label'] = ', '.join(e for e in arc.events)
    #else:
    #    attr['label'] = ''

def draw (hppn, graph, filename) :
    # format filename
    if not filename:
        filename = hppn.name
    badChar = " ()'\/[]{},;:!?."
    for i in badChar: filename = filename.replace(i, '_')
    pngFilename = filename + '.png'
    svgFilename = filename + '.svg'
    dotFilename = filename + '.dot'

    engine = 'dot' # 'neato' (double arrow and a bit messy) , 'dot', 'circo' (no suitable), 'twopi' (messy), 'fdp' (straight arrows and messy)

    # write dot 
    with open(dotFilename, 'w') as f:
        txt = graph.dot()
        txt =txt.replace('digraph {', 'digraph {\n  ratio=fill;\n  size="5.83,8.27!";\n node [fontsize=16];') # size = A5 landscape in inchs
        f.write(txt)
        name = f.name
        f.close()

    dots = []
    for filename in [pngFilename, svgFilename] :
        dots.append(subprocess.Popen([engine, "-T" + filename.rsplit(".", 1)[-1],
                                "-o" + filename, name],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE))

    for dot in dots:
        dot.communicate()

    #os.unlink(dotFilename)
    for dot in dots:
        if dot.returncode != 0 :
            raise IOError("%s exited with status %s" % (engine, dot.returncode))

import sys

import numpy as N
import veusz.embed as veusz

def main(outfile):
    # note - avoid putting text in here to avoid font issues

    embed = veusz.Embedded(hidden=True)
    x = N.arange(5)
    y = x**2
    embed.SetData('a', x)
    embed.SetData('b', y)

    page = embed.Root.Add('page')
    graph = page.Add('graph')
    xy = graph.Add('xy', xData='a', yData='b', marker='square')
    graph.x.TickLabels.hide.val = True
    graph.y.TickLabels.hide.val = True
    xy.MarkerFill.color.val = 'blue'

    embed.Export(outfile)

if __name__ == '__main__':
    main(sys.argv[1])

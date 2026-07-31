===========
Core Module
===========

.. currentmodule:: fucrimodo.core

The core module contains the basic classes and functions that are used to
implement the multi-stage search algorithm. 
The :class:`MultiStageSearch` class is the main class that is used to run the
multi-stage search algorithm. 
It can perform a :class:`modules.Stage` with the :meth:`MultiStageSearch.run` method.


.. tikz:: [node distance=2cm, auto] 

    \tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,text centered, draw=black, fill=red!30]
    \tikzstyle{io} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]
    \tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=orange!30]
    \tikzstyle{decision} = [diamond, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=green!30]
    \tikzstyle{arrow} = [thick,->,>=stealth]
    \tikzset{
        myclass/.style={
            draw, rectangle split, rectangle split parts=3, align=left,
            text centered, rounded corners, minimum width=3cm,
            rectangle split part align={center,left,left}
        }
    }

    \node (start) [startstop] {Start};
    \node (pop) [io, below of=start] {Population};

    \begin{scope}[local bounding box=stage]
    \node (selStartPop) [process, below of=pop, yshift=-1cm] {Select Population};
    \node (ga) [process, right of=selStartPop, xshift=2.5cm] {GA workflow};
    \end{scope}
    \node[draw, fit=(stage), inner sep=0.2cm, label=above:Stage, thick] {};

    \node (break) [decision, right of=pop, xshift=2.5cm] {Last Stage?};
    \node (end) [startstop, right of=break, xshift=2cm] {End};

    % Connect nodes
    \draw [arrow] (start) -- (pop);
    \draw [arrow] (pop) -- (selStartPop);
    \draw [arrow] (selStartPop) -- (ga);
    \draw [arrow] (ga) -- (break);
    \draw [arrow] (break) -- node[anchor=north] {yes} (end);
    \draw [arrow] (break) -- node[anchor=north] {no} (pop);

    :libs: arrows, shapes, positioning, fit

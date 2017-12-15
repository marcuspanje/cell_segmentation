from utilString import *
import json

#generates latex document for displaying pictures

f = open('results.tex', 'w') 

#preamble = \
preamble = r"""
\documentclass[11pt, oneside]{article}
\usepackage[headheight=13.6pt, textheight=620pt]{geometry}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{verbatim}
\usepackage{multicol}
\usepackage{fancyhdr}
\usepackage{color, colortbl}
\usepackage{subcaption}
\usepackage{caption}
\usepackage{float}

\begin{document}
\subsection*{Cell semantic segmentation results}
Order: Raw image, ground truth labels, FCN, FCN\\"""

f.write(preamble)

with open('fileNames.json') as fjs:
  allNames = json.load(fjs)

testNames = allNames['test']

prefix1 = 'output_files'
prefix2 = 'output_files'
nDisplay = 5
for i in range(min(nDisplay, len(testNames))):
  fName = 'cropped_' + testNames[i]
  lName = getLabeledName(fName)
  slash = lName.rfind('/')
  
  out1 = prefix1 + lName[slash:]
  out2 = prefix2 + lName[slash:]

  
  imgString = r"""  
\begin{figure}[H]
  \centering
  \begin{subfigure}[b]{0.23\textwidth}
  \includegraphics[width=\textwidth]{%s}
  \end{subfigure}
  \begin{subfigure}[b]{0.23\textwidth}
  \includegraphics[width=\textwidth]{%s}
  \end{subfigure}
  \begin{subfigure}[b]{0.23\textwidth}
  \includegraphics[width=\textwidth]{%s}
  \end{subfigure}
  \begin{subfigure}[b]{0.23\textwidth}
  \includegraphics[width=\textwidth]{%s}
  \end{subfigure}
\end{figure}
""" % (latexFileName(fName), latexFileName(lName), latexFileName(out1), latexFileName(out2))
  f.write(imgString)
  
f.write('\end{document}')
f.close()

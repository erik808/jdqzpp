# JDQZ++
Templated C++ implementation of the JDQZ generalized eigenvalue problem solver.

## Introduction
 This is an almost 1-1 port of the original FORTRAN code by Fokkema and van Gijzen, see
 
 - http://www.win.tue.nl/casa/research/scientificcomputing/topics/jd/software.html
 - http://www.staff.science.uu.nl/~sleij101/JD_software/jd.html

A few of the old routines survive and are compiled into a separate library.

The main difference with the original is that vectors are now templated. We require vectors to have a few standard members (e.g. `dot()`, `axpy()`, `scale()`, etc.) with complex arithmetic. For manipulations of the projected problem we use containers provided by the STL.

## Installation
This project depends on
- cmake
- lapack
- gtest


# IOmetry
A Multi-purpose, Application-Centric, Scalable I/O Proxy Application

IOmetry is being developed to fill a long-existing void in co-design
proxy applications that allow for I/O performance testing and evaluation
of tradeoffs in data models, I/O library interfaces, and parallel I/O
paradigms for multi-physics, HPC applications.
 
Two key design features of IOmetery set it apart from existing I/O proxy
applications and benchmarking tools. The first is the level of abstraction
(LOA) at which IOmetry is being designed to operate. The second is the
degree of flexibility, IOmetry is being designed to provide in driving an
HPC I/O workload through parameterized, user-defined data objects, parallel I/O paradigms, and I/O interfaces.
 
Combined, these features allow IOmetry to closely mimic I/O workloads for
a wide variety of real applications and, in particular, multi-physics
applications where data object distribution and composition vary
dramatically both within and across parallel tasks. These data objects
can then be marshaled using one or more I/O interfaces and parallel I/O
paradigms, allowing for direct comparisons of software interfaces,
parallel I/O paradigms, and file system technologies with the same set
of customizable data objects.


## Installing


See the [INSTALLING](/INSTALLING.md) file for manual installation instructions.


## Release

IOmetry is released under the terms of the GPL license. For full details
see the [LICENSE](/LICENSE) file.


# Zappy

Zappy is a simple electrical load flow modeling library for AC, DC and hybrid electrical systems based on the equations originally published in [this paper](https://ieeexplore.ieee.org/document/7961244).
The repository also includes a simple set of electical components for completing node-voltage circuit analyses.

Zappy is built on top of the OpenMDAO framework, with the code relying on OpenMDAO for data passing, solvers and optimizers among other things.  
The Zappy implementation includes analaytic derivatives for all components to enable efficient gradient-based optimization when the analysis is included in larger MDAO problems.

Disclosure: There are no docs for the software at this point. We're hoping to improve this, but for the moment this is what you get. We suggest you look in the examples folder for some indications of how to run this code.

## Install Zappy

git clone http://github.com/OpenMDAO/zappy

pip install zappy

## Zappy Applications
Zappy has been used as part of MDAO studies examining serveral electric aircraft concepts.  Papers describing these studies are listed below:

- [Load Flow Analysis with Analytic Derivatives for Electric Aircraft Design Optimization](https://arc.aiaa.org/doi/10.2514/6.2019-1220) by Hendricks, Chapman and Aretskin-Hariton
- [Multidisciplinary Optimization of a Turboelectric Tiltwing Urban Air Mobility Aircraft](https://arc.aiaa.org/doi/10.2514/6.2019-3551) by Hendricks, Falck, Gray, Aretskin-Hariton, Ingraham, Chapman, Schnulo, Chin, Jasa and Bergeson
- [Multidisciplinary Optimization of an Electric Quadrotor Urban Air Mobility Aircraft](https://arc.aiaa.org/doi/10.2514/6.2020-3176) by Hendricks, Aretskin-Hariton, Ingraham, Gray, Schnulo, Chin, Falck and Hall

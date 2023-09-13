***********
Altimetry
***********

The **Altimetry** module contains the objects and functions needed for plotting **altimetric profiles**.
These profiles are delivered as an artifact to consulting clients.
They are used for a quick scout of the terrain profile which includes the buildings that are going to be built.

There are some objects to abstract business logic such as:


* AltimerySection:
  
  * Holds logic for creating a section
    
* AltimeryProbe:
  
  * Parses a csv table and generates a list of probes for defining building position and section plane
  
* AltimeryShed

  * Abstracts the slice of a building cut by the section
  
* SectionVertices

  * Object to handle the vertices generated from slicing a surface
***********
Models
***********

This module abstract all of the common interface between **use cases** modules. It serves as an API
for developing other modules. It should contain only the objects that are known for sure as stable.
**Modifications on these objects can break something**, so the best practice is to avoid changing them, only if necessary.
And when changed, tests should be ran to ensure that the changes still work properly

.. important:: Modifications on these objects can break something. Avoid changing them!

The objects that abstract these common interfaces are:


* Point:
  
  * 3D point with its coordinates
    
* Line:
  
  * 3D line defined by two points and a discrete resolution
  
* Plane

  * 3D plane defined by a normal and a origin

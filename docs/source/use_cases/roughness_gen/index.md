# Roughness Elements

The **Roughness Elements** module is used to generate the geometry of the roughness elements, used in CFD Simulations.
These elements are used to represent the roughness of the terrain, just like how it is done in physical wind tunnels.

A standard configuration of the objects used for representing atmospheric flow can be seen in the following image:

```{figure} /_static/roughness_gen/wind_tunnel.png
```

The roughness of the terrain affects the mean velocity profile in the **Atmospheric Boundary Layer** (ABL).
The following image shows this effect:

```{figure} /_static/roughness_gen/ABL.png
```

According to Brazilian and European wind standards, the ABL profile can me represented by a roughness factor ($z_0$).
The objective of this module is to serve as a tool for generating the geometry of the roughness elements, in order to achieve a corresponding ABL profile.
The validation of the ABL profile is based on the mean velocity profile and turbulence intensity.
This profile is then used to obtain a corresponding roughness factor and compared to the ones presented by the standards.

## Usage

There are several ways to use **Roughness Elements generation** module. The main one is to run as a module, using:

```Bash
uv run python -m cfdmod.roughness --config {CONFIG_PATH} --output {OUTPUT_PATH}
```

It takes two arguments: the path for the **.yaml configuration file** with the generation parameters and the **output path** for saving the .STL file.
A third, optional `--mode` argument selects the generation mode: `linear` (the default, described here), `position` (elements draped onto a terrain, below) or `radial` (fins arranged in rings around a centre).
For standard use, the user must fullfill a configuration file with the parameters.
One example of the configuration is as it follows:

```{literalinclude} /_static/roughness_gen/roughness_params.yaml
:language: yaml
```

The parameters consist in defining the number of replication in each axis.
The figure below shows how the blocks are generated through configuration

```{figure} /_static/roughness_gen/roughness_config.png
:width: 100%
```

Another way is running via [notebook](./gen_roughness_elements.ipynb).

## Positioned roughness elements

One typical application of the roughness elements is to use them to mantain the turbulence near the ground, in the **upwind direction**.
Because of that, their position should follow the terrain profile.

To generate and position the roughness elements following a terrain profile, the **user must specify a bounding box** to where to spawn roughness elements.
The image below illustrates a bounding box that is going to delimit the roughness elements spawn location:

```{figure} /_static/roughness_gen/bounding_box.png
:width: 80%
```

The user can also specify multiple surfaces if the bounding box crosses them.
Parameters file must also be an input, such as the following example:

```{literalinclude} /_static/roughness_gen/position_params.yaml
:language: yaml
```

Run it with the `position` mode:

```Bash
uv run python -m cfdmod.roughness --config {CONFIG_PATH} --output {OUTPUT_PATH} --mode position
```

It writes `positioned_elements.stl` in the output path.

The array fills the intersection of the bounding box and the surfaces' own extent.
Only the X and Y of the bounding box constrain the placement: the Z of each element comes from the surface it is draped onto.
Each element is seated on the lowest surface height across its base span, so an element on a slope rests on the terrain instead of floating over its downhill half.

An element whose base lands over no surface has no height to be seated on.
`on_missing_surface` decides what happens to it: `drop` (the default) removes it, and `keep` leaves it unlifted with its base at z = 0.
The same option exists on the radial mode, and both read it from the configuration file (see the example above).

The same routine is available from Python, and there it also accepts surfaces already in memory:

```python
from cfdmod.roughness import PositionParams, position_pattern

cfg = PositionParams.from_file(config_path)
triangles, normals = position_pattern(
    element_params=cfg.element_params,
    spacing_params=cfg.spacing_params,
    bounding_box=cfg.bounding_box,
    surfaces=[terrain_lnas],  # a path, an LnasFormat, an LnasGeometry or an (N, 3) vertex array
)
```

Surface heights are sampled from a triangulation of the surfaces' vertices.
Terrain surfaces routinely carry hundreds of thousands of vertices, and the triangulation grows faster than linearly with that count, so the sample is capped at `max_points` (default 100000).
The boundary of the surface is kept at full density when the cap bites, so neither the extent covered nor the accuracy of the drape near the edge changes; pass `max_points=None` to use every vertex.

A worked example is in the [notebook for positioning elements](./position_roughness_elements.ipynb).

## Output

The expected output is a STL file containing the information for creating the geometry.
This file can be inspected in CAD softwares, such as mesh lab.
An example of the output can be seen below:

```{figure} /_static/roughness_gen/elements.png
```

```{toctree}
:maxdepth: -1
:hidden:

Generating roughness elements <gen_roughness_elements.ipynb>
Positioning elements in terrain <position_roughness_elements.ipynb>
```

For the second use case, generating elements that conform to the terrain surface, the output geometry will be:

```{figure} /_static/roughness_gen/elements_positioned.png
```

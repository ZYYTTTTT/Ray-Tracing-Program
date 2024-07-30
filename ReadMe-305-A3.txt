README for Ray Tracing Program
==============================

Overview:
---------
This program is a ray tracer that simulates the way light interacts with objects in a 3D environment to generate photorealistic images. It supports basic geometric shapes like spheres and light sources, handling reflections, shading, and shadows.

Usage:
------
To run the program, use the following command in a terminal:
    python ray_tracer.py [scene_file]

Here, `[scene_file]` should be replaced with the path to a text file describing the scene.

Scene File Format:
------------------
The scene file is a plain text file containing information about spheres, lights, camera, and other scene settings. Each line in the file represents a different element or setting:

- RES [width] [height]: Sets the resolution of the output image.
- SPHERE [name] [xPos] [yPos] [zPos] [xScale] [yScale] [zScale] [r] [g] [b] [ka] [kd] [ks] [kr] [n]: Defines a sphere.
- LIGHT [name] [xPos] [yPos] [zPos] [r] [g] [b]: Defines a light source.
- BACK [r] [g] [b]: Sets the background color.
- AMBIENT [r] [g] [b]: Sets the ambient light color.
- OUTPUT [filename]: Specifies the output file for the rendered image.

The parameters for spheres and lights should be self-explanatory. `ka`, `kd`, `ks`, `kr`, and `n` are material properties (ambient, diffuse, specular coefficients, reflectivity, and shininess, respectively).

Modules and Classes:
--------------------
- Sphere: Represents a 3D sphere in the scene.
- Light: Represents a light source.
- Ray: Represents a ray of light with an origin, direction, and depth.
- Additional utility functions handle vector operations, ray-sphere intersections, light calculations, and ray tracing logic.

Output:
-------
The program outputs a PPM file with the rendered scene based on the specified settings in the scene file.

Note:
-----
Ensure you have Python installed on your system along with the NumPy library, as it is required for the program to run successfully.

Author:
-------
Yue Zhang
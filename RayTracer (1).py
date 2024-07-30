import sys
import numpy as np

MAX_DEPTH = 3
CAMERA_POS = [0,0,0]

class Sphere:
    def __init__(self, data):
        self.name   = data[1]
        self.xPos   = float(data[2])
        self.yPos   = float(data[3])
        self.zPos   = float(data[4])
        self.xScale = float(data[5])
        self.yScale = float(data[6])
        self.zScale = float(data[7])
        self.colour = [float(data[8]), float(data[9]), float(data[10])]
        self.ka     = float(data[11])
        self.kd     = float(data[12])
        self.ks     = float(data[13])
        self.kr     = float(data[14])
        self.n      = int(data[15])

    def __str__(self):
        return f'name {self.name}'

class Light:
    def __init__(self, data):
        self.name   = data[1]
        self.pos   = [float(data[2]), float(data[3]), float(data[4])]
        self.colour    = [float(data[5]), float(data[6]), float(data[7])]

    def __str__(self):
        return f'name {self.name} Colour: {self.colour}'

class Ray:
    def __init__(self, origin, direction, depth=1):
        self.origin = origin
        self.direction = direction
        self.depth = depth

    def __str__(self):
        return f'Origin: {self.origin} Direction: {self.direction}'

def normalize(v):
    return v / magnitude(v)

def magnitude(v):
    return (v[0]**2 + v[1]**2 + v[2]**2)**(1/2)

def hit_sphere(ray, sphere, invM, homoOrigin, homoDir):
    """
    Calculate the intersection of a ray with a transformed sphere.

    Parameters:
    ray (Ray): The ray to test for intersection.
    sphere (Sphere): The sphere which the ray may intersect.
    invM (np.array): Inverse of the transformation matrix of the sphere.
    homoOrigin (np.array): Homogeneous coordinates of the ray's origin.
    homoDir (np.array): Homogeneous coordinates of the ray's direction.

    """
    invS = np.matmul(invM, homoOrigin)[:3]  # Transformed ray origin
    invC = np.matmul(invM, homoDir)[:3]     # Transformed ray direction

    # Calculate the coefficients of the quadratic equation for ray-sphere intersection
    a = magnitude(invC)**2
    b = np.dot(invS, invC)
    c = magnitude(invS)**2 - 1

    # Calculate the discriminant to determine the nature of the intersections
    discriminant = b*b - a * c

    # If the discriminant is negative, there are no real intersections (ray misses the sphere)
    if discriminant < 0:
        return []
    else:
        # Two intersection points are calculated (entering and exiting points on the sphere)
        # The formula is derived from solving the quadratic equation for a sphere
        return [-b / a + np.sqrt(discriminant) / a, -b / a - np.sqrt(discriminant) / a]


def get_reflected_ray(incident, P, N):
    # Normalize the surface normal vector.
    # This is necessary to ensure accurate reflection calculations.
    normN = normalize(N)

    # Calculate the reflection direction.
    # The formula is based on the law of reflection: R = D - 2(D · N)N
    # where R is the reflection vector, D is the incident direction, and N is the normalized normal.
    # The dot product (D · N) represents the scalar projection of D onto N.
    # The term 2(D · N)N represents the component of D that is parallel to N, doubled.
    # Subtracting this from D reverses the direction of the parallel component, achieving reflection.
    v = -2 * np.dot(normN, incident.direction) * normN + incident.direction


    return Ray(P, v, incident.depth + 1)


def contributes_light(startSphere, endSphere, side, distToIntersect, dirToLight):
    # Calculate the squared distance from the intersection point to the light source.
    distToLight = np.dot(dirToLight, dirToLight)

    # Determine if the intersection is on the near side of the starting sphere.
    hitNear = side == "near"

    # Check if the ray has intersected any sphere on its path to the light source.
    hitSphere = endSphere is not None

    # Check if the ray has intersected the starting sphere itself (for cases like internal reflections).
    hitSelf = startSphere.name == endSphere.name if hitSphere else False

    # Case 1: The light source is visible from the intersection point.
    # This happens when the ray does not hit any sphere (no shadow) and the intersection is on the near side.
    if not hitSphere and hitNear:
         return True

    # Case 2: The light source is within the starting sphere and the ray has intersected the far side.
    # This can occur with transparent or semi-transparent spheres where the light source is enclosed within the sphere.
    elif hitSelf and not hitNear and distToLight < distToIntersect:
        return True

    # Case 3: The ray hits another sphere before reaching the light source, creating a shadow.
    # Or any other case where the light source is not directly visible.
    else:
        return False


# Returns the combined diffuse and specular value that a light contributes to a pixel
def get_light_value(light, spheres, P, hitSphere, N, near, side):
    """
    Calculate the combined diffuse and specular value that a light contributes to a pixel.

    Parameters:
    light (Light): The light source.
    spheres (list of Sphere): The list of spheres in the scene.
    P (np.array): The point of intersection.
    hitSphere (Sphere): The sphere that was hit by the ray.
    N (np.array): The normal at the point of intersection.
    near (float): Near plane distance for visibility checks.
    side (str): The side ('near' or 'far') of the sphere that was hit.

    Returns:
    np.array: The combined color contribution from the light source.
    """
    L = light.pos - P
    rayToLight = Ray(P, L)
    t, nearestSphere, _, _ = get_nearest_intersect(spheres, rayToLight)
    if(not contributes_light(hitSphere, nearestSphere, side, t, L)):
        return [0,0,0] # Shadow
    normN = normalize(N)
    normL = normalize(L)
    if side == "far":
        normN = -normN
    diffuse = hitSphere.kd * np.multiply(light.colour, np.dot(normN, normL)) * hitSphere.colour
    specular = [0,0,0]
    V = np.array(P) * -1
    normV = normalize(V)
    R = 2*np.multiply(np.dot(normN, L), normN) - L
    normR = normalize(R)
    RdotV = np.dot(normR, normV)**hitSphere.n
    specular = hitSphere.ks * np.multiply(light.colour, RdotV)
    return np.add(diffuse, specular)


# Returns the nearest intersection between the spheres and a ray, the sphere that was hit, and the side (near or far) that the ray hit
def get_nearest_intersect(spheres, ray, near=-1):
    """
    Find the nearest intersection between a ray and a set of spheres.

    Parameters:
    spheres (list of Sphere): The list of spheres in the scene.
    ray (Ray): The ray to test for intersections.
    near (float): The distance to the near plane for intersection checks.

    Returns:
    tuple: The nearest intersection distance, the sphere hit, the normal at the intersection, and the intersection side ('near' or 'far').
    """
    closestCircle = None
    t = 100000
    for sphere in spheres:
        invM = [[1/sphere.xScale, 0, 0, -sphere.xPos/sphere.xScale],[0, 1/sphere.yScale, 0, -sphere.yPos/sphere.yScale], [0, 0, 1/sphere.zScale, -sphere.zPos/sphere.zScale], [0, 0, 0, 1]]
        homoOrigin = np.append(ray.origin, 1)
        homoDir = np.append(ray.direction, 0)
        nextHits = hit_sphere(ray, sphere, invM, homoOrigin, homoDir)
        for hit in nextHits:
            zDist = 0
            if near != -1:
                distAlongLine = np.array(ray.direction) * hit
                zDist = np.dot(np.array([0,0,-1]), distAlongLine)
            if hit > 0.000001 and hit < t and (zDist > near or ray.depth != 1):
                t = hit
                closestCircle = sphere
    invN = None
    side = None
    if closestCircle is not None:
        M = [[closestCircle.xScale, 0, 0, closestCircle.xPos],[0, closestCircle.yScale, 0, closestCircle.yPos], [0, 0, closestCircle.zScale, closestCircle.zPos], [0, 0, 0, 1]]
        center = np.array([closestCircle.xPos, closestCircle.yPos, closestCircle.zPos])
        P = ray.origin + ray.direction * t
        N = np.subtract(P, center)
        homoN = np.append(N, 1)
        inversed = np.matmul(homoN, np.linalg.inv(M))
        invN = np.matmul(np.linalg.inv(np.transpose(M)), inversed)[:3]
        side = "far" if np.dot(ray.direction, invN) > 0 else "near"
    return (t, closestCircle, invN, side)


def rayT(ray, spheres, lights, sceneInfo):
    # Check if the ray has exceeded the maximum depth of recursion
    # This prevents infinite recursion and controls the complexity of the rendering
    if ray.depth > MAX_DEPTH:
        return [0, 0, 0]

    # Find the nearest intersection of the ray with any sphere in the scene
    nearestHit, closestCircle, N, side = get_nearest_intersect(spheres, ray, sceneInfo["NEAR"])

    # If no sphere is intersected by the ray
    if not closestCircle:
        # If this is the primary ray (first recursion level), return the background color
        if ray.depth == 1:
            return sceneInfo["BACK"]
        else:
            # For rays beyond the first level, return black (indicating no contribution to the color)
            return [0, 0, 0]

    # Calculate the point of intersection on the sphere
    P = ray.origin + ray.direction * nearestHit

    # Initialize the variable to accumulate the diffuse light contribution from all light sources
    diffuseLight = np.array([0, 0, 0])
    for light in lights:
        # Calculate and add the light contribution from each light source
        diffuseLight = np.add(diffuseLight, get_light_value(light, spheres, P, closestCircle, N, sceneInfo["NEAR"], side))

    # Calculate the ambient light contribution based on the sphere's material and the scene's ambient light setting
    ambient = closestCircle.ka * np.multiply(sceneInfo["AMBIENT"], closestCircle.colour)

    # Calculate the reflected ray if the sphere is reflective
    refRay = get_reflected_ray(ray, P, N)

    # Combine the ambient, diffuse, and reflective light contributions to get the final color
    return ambient + diffuseLight + closestCircle.kr * np.array(rayT(refRay, spheres, lights, sceneInfo))


def get_data(sceneInfo, spheres, lights, outputFile):
    # Print scene configuration, including general settings and object details.
    # This includes all parsed scene information such as resolution, background color, ambient light settings,
    # along with the details of each sphere and light source in the scene, and the output file name.
    # Useful for debugging to ensure that scene data is correctly parsed and stored.

    print(sceneInfo)

    for sphere in spheres:
        print(sphere)

    for light in lights:
        print(light)

    print(outputFile)


def get_ppm(info, spheres, lights, outputFile):
    # Set up image dimensions and PPM header
    width = info["RES"]["x"]
    height = info["RES"]["y"]
    ppm_header = f'P6 {width} {height} {255}\n'
    image = np.zeros([width * height * 3])

    # Define base vectors for ray direction calculation
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    n = np.array([0, 0, -1])
    percentInc = int(height / 10)

    # Iterate over each pixel and compute its color using ray tracing
    for r in range(height):
        # Print progress at intervals
        if (r % percentInc == 0):
            print(f'{r / percentInc * 10}% Complete')

        for c in range(width):
            # Calculate the ray's direction for the current pixel
            origin = CAMERA_POS
            xComp = info["RIGHT"] * (2.0 * float(c) / float(width) - 1)
            yComp = info["TOP"] * (2.0 * (height - r) / height - 1)
            zComp = info["NEAR"]
            direction = np.add(xComp * u, yComp * v)
            direction = np.add(direction, zComp * n)
            ray = Ray(origin, direction)

            # Determine the pixel color using ray tracing
            pixelColour = rayT(ray, spheres, lights, info)
            startIndex = 3 * (r * width + c)
            clippedPix = np.clip(pixelColour, 0, 1) * 255
            image[startIndex] = int(clippedPix[0])
            image[startIndex + 1] = int(clippedPix[1])
            image[startIndex + 2] = int(clippedPix[2])

    # Write the computed image data to a PPM file
    with open(outputFile, 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        image.astype('int8').tofile(f)

    # Indicate rendering completion
    print("Render Complete. Output to", outputFile)


def main():
    fileName = sys.argv[1]
    # Retrieve the scene file name from command line arguments
    sceneInfo = {}
    spheres = []
    lights = []
    outputFile = None
    with open(fileName) as fp:
        for i, line in enumerate(fp):
            sl = line.split()
            if len(sl) == 0:
                continue
            if(sl[0] == "RES"):
                sceneInfo["RES"] = {}
                sceneInfo["RES"]["x"] = int(sl[1])
                sceneInfo["RES"]["y"] = int(sl[2])
            elif(sl[0] == "SPHERE"):
                spheres.append(Sphere(sl))
            elif(sl[0] == "LIGHT"):
                lights.append(Light(sl))
            elif(sl[0] == "BACK"):
                sceneInfo["BACK"] = [float(sl[1]), float(sl[2]), float(sl[3])]
            elif(sl[0] == "AMBIENT"):
                sceneInfo["AMBIENT"] = [float(sl[1]), float(sl[2]), float(sl[3])]
            elif(sl[0] == "OUTPUT"):
                outputFile = sl[1]
            else:
               sceneInfo[sl[0]] = float(sl[1])
    # Perform ray tracing and write the output image
    get_ppm(sceneInfo, spheres, lights, outputFile)


if __name__ == '__main__':
    main()
Mathematical Formulation of NDC
1. Ray Representation
A ray in 3D space is defined by:
Origin: $\mathbf{o} = (o_x, o_y, o_z)$
Direction: $\mathbf{d} = (d_x, d_y, d_z)$ (unit vector)
Ray equation: $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$
2. The Parameter $t$ Calculation
The parameter $t$ represents the distance along the ray to reach the near plane:
$$t = -\frac{\text{near} + o_z}{d_z}$$

Derivation:
The near plane is at $z = -\text{near}$ (assuming camera looks along negative z-axis)
We want to find where the ray intersects this plane
At intersection: $o_z + t \cdot d_z = -\text{near}$
Solving for $t$: $t = -\frac{\text{near} + o_z}{d_z}$

3. Shifting Ray Origins
Shift all ray origins to the near plane:
$$\mathbf{o}' = \mathbf{o} + t\mathbf{d}$$
This ensures all rays start from the same normalized depth plane.

4. Perspective Projection
The projection transforms 3D points to 2D normalized coordinates:
For Ray Origins:
$$o'0 = -\frac{1}{W/(2f)} \cdot \frac{o'_x}{o'_z}$$
$$o'1 = -\frac{1}{H/(2f)} \cdot \frac{o'_y}{o'_z}$$
$$o'2 = 1 + \frac{2 \cdot \text{near}}{o'_z}$$
Where:
$W, H$ = image width and height in pixels
$f$ = focal length
$o'x, o'_y, o'_z$ = shifted ray origin coordinates
For Ray Directions:
$$d'0 = -\frac{1}{W/(2f)} \cdot \left(\frac{d_x}{d_z} - \frac{o'_x}{o'_z}\right)$$
$$d'1 = -\frac{1}{H/(2f)} \cdot \left(\frac{d_y}{d_z} - \frac{o'_y}{o'_z}\right)$$
$$d'2 = -\frac{2 \cdot \text{near}}{o'_z}$$
Mathematical Intuition
1. Perspective Division
The key insight is the perspective division $\frac{x}{z}, \frac{y}{z}$:
This comes from the pinhole camera model:
$$\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \frac{1}{Z} \begin{pmatrix} f & 0 & 0 \\ 0 & f & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}$$
Where $(u,v)$ are the image coordinates and $(X,Y,Z)$ are 3D world coordinates.
2. Normalization Factors
The factors $\frac{W}{2f}$ and $\frac{H}{2f}$ convert from:
Image coordinates (pixels) to normalized device coordinates (range [-1, 1])
This is because:
Image center is at $(W/2, H/2)$
Focal length $f$ determines the field of view
The ratio gives us the scale factor for normalization
3. Depth Normalization
The $o'2$ calculation:
$$o'2 = 1 + \frac{2 \cdot \text{near}}{o'_z}$$
This maps depth from:
World coordinates: $[-\text{near}, \infty)$
Normalized coordinates: $[0, 1]$
The transformation is:
$$z_{\text{norm}} = \frac{z_{\text{world}} + \text{near}}{2 \cdot \text{near}}$$
Complete Mathematical Flow
Input:
Ray origin: $\mathbf{o} = (o_x, o_y, o_z)$
Ray direction: $\mathbf{d} = (d_x, d_y, d_z)$
Camera parameters: $W, H, f, \text{near}$
Step 1: Calculate shift parameter
$$t = -\frac{\text{near} + o_z}{d_z}$$
Step 2: Shift ray origin
$$\mathbf{o}' = \mathbf{o} + t\mathbf{d}$$
Step 3: Apply perspective projection
$$\mathbf{o}{\text{ndc}} = \begin{pmatrix} -\frac{1}{W/(2f)} \cdot \frac{o'_x}{o'_z} \\ -\frac{1}{H/(2f)} \cdot \frac{o'_y}{o'_z} \\ 1 + \frac{2 \cdot \text{near}}{o'_z} \end{pmatrix}$$
$$\mathbf{d}{\text{ndc}} = \begin{pmatrix} -\frac{1}{W/(2f)} \cdot \left(\frac{d_x}{d_z} - \frac{o'_x}{o'_z}\right) \\ -\frac{1}{H/(2f)} \cdot \left(\frac{d_y}{d_z} - \frac{o'_y}{o'_z}\right) \\ -\frac{2 \cdot \text{near}}{o'_z} \end{pmatrix}$$
Why This Works
1. Coordinate System Normalization
All scenes are mapped to a consistent coordinate system regardless of original camera distances.
2. Perspective Consistency
The perspective division $\frac{x}{z}, \frac{y}{z}$ ensures that:
Objects further away appear smaller (perspective effect)
All cameras have consistent perspective projection
3. Depth Range Standardization
The depth mapping ensures that:
Near objects are at $z_{\text{ndc}} \approx 0$
Far objects are at $z_{\text{ndc}} \approx 1$
This prevents numerical issues during training
4. Ray Direction Transformation
The direction transformation accounts for the fact that:
In NDC space, ray directions change as we move through the normalized coordinate system
The subtraction term $\frac{o'x}{o'_z}$ ensures rays point in the correct direction in NDC space
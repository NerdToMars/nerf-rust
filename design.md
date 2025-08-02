# NeRF: Mathematical Design Document

## 1. High-Level Overview

### 1.1 Problem Statement
- **Goal**: Generate novel views of a 3D scene from a set of 2D images with known camera poses
- **Input**: Multiple images + camera poses (intrinsics & extrinsics)
- **Output**: Novel view synthesis from arbitrary camera positions

### 1.2 Core Idea
- Represent 3D scene as a **Neural Radiance Field**: $F(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$
- **Volume Rendering** to synthesize novel views
- **Differentiable rendering** for end-to-end training

## 2. Mathematical Foundations

### 2.1 Neural Radiance Field Representation
- **Function**: $F: \mathbb{R}^3 \times \mathbb{S}^2 \rightarrow \mathbb{R}^3 \times \mathbb{R}^+$
- **Input**: 3D position $\mathbf{x} = (x, y, z)$ + view direction $\mathbf{d} = (d_x, d_y, d_z)$
- **Output**: RGB color $\mathbf{c} = (r, g, b)$ + volume density $\sigma$

### 2.2 Volume Rendering Equation
- **Classical Volume Rendering**: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$
- **Transmittance**: $T(t) = \exp(-\int_{t_n}^t \sigma(\mathbf{r}(s)) ds)$
- **Discrete Approximation**: $C(\mathbf{r}) = \sum_{i=1}^N T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$

## 3. Ray Generation and Camera Model

### 3.1 Camera Intrinsics and Extrinsics
- **Intrinsics Matrix**: $K = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}$
- **Camera-to-World**: $c2w = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$

### 3.2 Ray Generation
- **Pixel Coordinates**: $(i, j)$ where $i \in [0, W-1], j \in [0, H-1]$
- **Camera Space Directions**: $\mathbf{d}_{cam} = \begin{bmatrix} \frac{i - c_x}{f} \\ \frac{-(j - c_y)}{f} \\ -1 \end{bmatrix}$
- **World Space Rays**: $\mathbf{o} = c2w[:3, -1]$, $\mathbf{d} = R \cdot \mathbf{d}_{cam}$

### 3.3 Normalized Device Coordinates (NDC)
- **Purpose**: Normalize forward-facing scenes for stable training
- **Transformation**: $\mathbf{o}_{ndc}, \mathbf{d}_{ndc} = \text{NDC}(\mathbf{o}, \mathbf{d}, H, W, f, \text{near})$

## 4. Positional Encoding

### 4.1 High-Frequency Function Encoding
- **Encoding Function**: $\gamma(\mathbf{x}) = [\mathbf{x}, \sin(2^0\pi\mathbf{x}), \cos(2^0\pi\mathbf{x}), ..., \sin(2^{L-1}\pi\mathbf{x}), \cos(2^{L-1}\pi\mathbf{x})]$
- **3D Positions**: $L = 10$ (multires)
- **View Directions**: $L = 4$ (multires_views)

### 4.2 Mathematical Justification
- **Fourier Features**: Enable learning of high-frequency functions
- **Input Dimensions**: $3 + 2 \cdot L \cdot 3$ for positions, $3 + 2 \cdot L \cdot 3$ for directions

## 5. Neural Network Architecture

### 5.1 Network Structure
- **Coarse Network**: 8 layers, 256 hidden units
- **Fine Network**: 8 layers, 256 hidden units (hierarchical sampling)
- **Skip Connections**: At layer 4

### 5.2 Forward Pass
- **Position Branch**: $h_i = \text{ReLU}(W_i h_{i-1} + b_i)$
- **Skip Connection**: $h_4 = \text{ReLU}(W_4 [h_3, \gamma(\mathbf{x})] + b_4)$
- **View-Dependent Branch**: $h_{view} = \text{ReLU}(W_{view} [h_{pos}, \gamma(\mathbf{d})] + b_{view})$

### 5.3 Output Heads
- **Density**: $\sigma = \text{ReLU}(W_\sigma h_{pos} + b_\sigma)$
- **Color**: $\mathbf{c} = \text{Sigmoid}(W_c h_{view} + b_c)$

## 6. Volume Rendering Implementation

### 6.1 Mathematical Foundation

#### **6.1.1 Continuous Volume Rendering Equation**

Volume rendering models how light interacts with a participating medium (like fog, smoke, or a 3D scene). The fundamental equation describes the color observed along a ray:

**Ray Definition:**
$
\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}, \quad t \in [t_n, t_f]
$

**Volume Rendering Equation:**
$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt
$

Where:
- \(T(t)\) = **Transmittance** (probability light travels from \(t_n\) to \(t\) without being absorbed)
- $\sigma(\mathbf{r}(t))$ = **Volume density** at point $\mathbf{r}(t)$
- $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ = **Color** at point $\mathbf{r}(t)$ viewed from direction $\mathbf{d}$

**Transmittance Derivation:**
$
T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) ds\right)
$

This follows from the differential equation: $\frac{dT}{dt} = -\sigma(t) T(t)$, which models light attenuation.

#### **6.1.2 Discrete Approximation**

Since we can't compute the integral analytically, we discretize it by sampling \(N\) points along the ray:

**Sampling Points:**
$
t_i = t_n + \frac{i}{N-1}(t_f - t_n), \quad i = 0, 1, ..., N-1
$

**Discrete Approximation:**
$
C(\mathbf{r}) \approx \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i
$

Where:
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ = **Alpha value** (opacity of sample \(i\))
- $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$ = **Accumulated transmittance**
- $\delta_i = t_{i+1} - t_i$ = **Distance between samples**

**Mathematical Derivation of Alpha:**
The probability of a ray terminating at sample \(i\) is:
$
\alpha_i = 1 - \exp(-\sigma_i \delta_i)
$

This comes from the Beer-Lambert law: \(I = I_0 \exp(-\sigma d)\), where the absorbed fraction is \(1 - \exp(-\sigma d)\).

### 6.2 Intuitive Examples

#### **6.2.1 Foggy Room Example**

Imagine looking through a foggy room with colored smoke at different depths:

- **Clear air** ($\sigma = 0$): You see through to the background
- **Dense smoke** ($\sigma = \infty$): You see only the smoke color, nothing behind
- **Moderate fog** ($\sigma = 0.1$): You see a blend of smoke colors and background

**Visual Example:**
```
Ray: [Camera] -----> [Smoke Layer 1] -----> [Smoke Layer 2] -----> [Background]
      t=0            t=3, σ₁=0.1, c₁=red    t=5, σ₂=0.2, c₂=blue   t=10
```

**Calculation:**
- $\alpha_1 = 1 - \exp(-0.1 \cdot 3) = 0.259$
- $\alpha_2 = 1 - \exp(-0.2 \cdot 5) = 0.632$
- $T_1 = 1$ (first sample)
- $T_2 = 1 - \alpha_1 = 0.741$
- Final color = $0.259 \cdot \text{red} + 0.741 \cdot 0.632 \cdot \text{blue}$

#### **6.2.2 NeRF Scene Example**

In NeRF, think of the 3D scene as a "cloud" of colors and densities:

- **Empty space**: $\sigma \approx 0$, $\alpha \approx 0$ (transparent)
- **Object surface**: $\sigma$ high, $\alpha$ high (opaque)
- **Object interior**: $\sigma$ moderate, gradual accumulation

### 6.3 Code Implementation Walkthrough

#### **6.3.1 Ray Sampling (from render_rays function)**

```python
# Line 375-376 in run_nerf.py
rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]  # [N_rays, 3] each
viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
```

**Mathematical Translation:**
- `rays_o` = \(\mathbf{o}\) (ray origins)
- `rays_d` = \(\mathbf{d}\) (ray directions)
- `viewdirs` = \(\mathbf{d}\) (view directions, same as ray directions)

```python
# Line 378-385: Depth sampling
t_vals = torch.linspace(0., 1., steps=N_samples)
if not lindisp:
    z_vals = near * (1.-t_vals) + far * (t_vals)  # Linear sampling
else:
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))  # Disparity sampling
```

**Mathematical Translation:**
- Linear sampling: $t_i = t_n + \frac{i}{N-1}(t_f - t_n)$
- Disparity sampling: $t_i = \frac{1}{\frac{1}{t_n}(1-\frac{i}{N-1}) + \frac{1}{t_f}\frac{i}{N-1}}$

```python
# Line 387-395: Stratified sampling (perturbation)
if perturb > 0.:
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand
```

**Mathematical Translation:**


Adds random jitter within each sampling interval: $t_i' = t_i + \text{rand}(0, \delta_i)$

```python
# Line 397: 3D point sampling
pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
```

**Mathematical Translation:**

$\mathbf{x}_i = \mathbf{o} + t_i \mathbf{d}$ for each sample point

#### **6.3.2 Network Inference**

```python
# Line 400: Network query
raw = network_query_fn(pts, viewdirs, network_fn)
```

**Mathematical Translation:**
$\mathbf{c}_i, \sigma_i = F(\mathbf{x}_i, \mathbf{d})$ for each sample point

#### **6.3.3 Volume Rendering (raw2outputs function)**

```python
# Line 284-330 in run_nerf.py: raw2outputs function
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
    # Calculate distances between samples
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
```

**Mathematical Translation:**
- \(\delta_i = t_{i+1} - t_i\) (distances between samples)
- Last distance set to \(\infty\) for proper handling

```python
    # Extract RGB and apply sigmoid
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    # Add noise to density for regularization
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
    
    # Calculate alpha values
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
```

**Mathematical Translation:**
- $\mathbf{c}_i = \text{sigmoid}(\text{raw}_{i,1:3})$ (RGB colors)
- $\sigma_i = \text{ReLU}(\text{raw}_{i,4})$ (density)
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ (alpha values)

```python
    # Calculate weights (alpha compositing)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
```

**Mathematical Translation:**
\(w_i = \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)\) (weights for each sample)

```python
    # Final color computation
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
    # Depth and disparity
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
```

**Mathematical Translation:**
- $C(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}_i$ (final color)
- $d(\mathbf{r}) = \sum_{i=1}^N w_i t_i$ (depth)
- $\text{disp}(\mathbf{r}) = \frac{1}{d(\mathbf{r})}$ (disparity)
- $\text{acc}(\mathbf{r}) = \sum_{i=1}^N w_i$ (accumulated opacity)

### 6.4 Hierarchical Sampling

#### **6.4.1 Two-Stage Process**

```python
# Line 402-420: Hierarchical sampling
if N_importance > 0:
    rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
    # Sample additional points based on coarse network weights
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))
    
    # Combine coarse and fine samples
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    # Query fine network
    raw = network_query_fn(pts, viewdirs, run_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
```

**Mathematical Translation:**
1. **Coarse Stage**: Uniform sampling with \(N_c\) points
2. **Fine Stage**: Importance sampling with \(N_f\) additional points based on coarse weights
3. **PDF**: \(p(t) \propto w_i\) from coarse network

#### **6.4.2 Importance Sampling (sample_pdf function)**

```python
# Line 176-242 in run_nerf_helpers.py
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Convert weights to PDF
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    
    # Sample from CDF
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    
    # Invert CDF to get samples
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    # Linear interpolation
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    
    return samples
```

**Mathematical Translation:**
1. **PDF Construction**: \(p_i = \frac{w_i}{\sum_j w_j}\)
2. **CDF**: \(F_i = \sum_{j=1}^i p_j\)
3. **Inverse CDF Sampling**: \(t = F^{-1}(u)\) where \(u \sim \text{Uniform}(0,1)\)

### 6.5 Background Handling

#### **6.5.1 White Background**

```python
# Line 328-329 in raw2outputs
if white_bkgd:
    rgb_map = rgb_map + (1.-acc_map[...,None])
```

**Mathematical Translation:**
\(C_{\text{final}}(\mathbf{r}) = C(\mathbf{r}) + (1 - \sum w_i) \cdot \text{white}\)

This adds white background color to transparent regions.

### 6.6 References and Cross-Checking

#### **6.6.1 Primary References**

1. **NeRF Paper**: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
   - Section 4: Volume Rendering with Radiance Fields
   - Section 5.2: Hierarchical Volume Sampling

2. **Volume Rendering Fundamentals**: 
   - [Real-Time Volume Graphics](https://www.amazon.com/Real-Time-Volume-Graphics-Klaus-Engel/dp/1568812663)
   - [Volume Rendering Techniques](https://www.researchgate.net/publication/220184455_Volume_Rendering_Techniques)

3. **Computer Graphics References**:
   - [Physically Based Rendering](https://www.pbr-book.org/) - Chapter 11: Volume Scattering
   - [Real-Time Rendering](https://www.realtimerendering.com/) - Chapter 14: Volume Rendering

#### **6.6.2 Code References**

1. **NeRF Implementation**: 
   - `run_nerf.py`: Lines 284-330 (`raw2outputs` function)
   - `run_nerf.py`: Lines 375-420 (`render_rays` function)
   - `run_nerf_helpers.py`: Lines 176-242 (`sample_pdf` function)

2. **Alternative Implementations**:
   - [Official NeRF TensorFlow](https://github.com/bmild/nerf)
   - [PyTorch Lightning NeRF](https://github.com/ashawkey/torch-ngp)

#### **6.6.3 Mathematical Verification**

**Key Equations to Verify:**

1. **Alpha Calculation**: \(\alpha = 1 - \exp(-\sigma \delta)\)
   - Verify: When \(\sigma = 0\), \(\alpha = 0\) (transparent)
   - Verify: When \(\sigma = \infty\), \(\alpha = 1\) (opaque)

2. **Weight Calculation**: \(w_i = \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)\)
   - Verify: \(\sum w_i \leq 1\) (conservation of energy)
   - Verify: \(w_i \geq 0\) (non-negative weights)

3. **Final Color**: \(C(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}_i\)
   - Verify: Each component \(C_r, C_g, C_b \in [0, 1]\) (valid color range)

### 6.7 Common Pitfalls and Debugging

#### **6.7.1 Numerical Issues**

1. **NaN/Inf Values**: Often caused by division by zero in disparity calculation
   ```python
   disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
   ```

2. **Gradient Explosion**: Can occur with high density values
   ```python
   alpha = 1.-torch.exp(-torch.relu(raw[...,3])*dists)  # ReLU prevents negative density
   ```

#### **6.7.2 Performance Issues**

1. **Memory Usage**: Process rays in chunks
   ```python
   chunk = 1024*32  # Adjust based on GPU memory
   ```

2. **Computation Time**: Use hierarchical sampling to focus computation on important regions

### 6.8 Summary

Volume rendering in NeRF combines:
- **Mathematical Foundation**: Classical volume rendering equations
- **Neural Representation**: MLP predicts color and density at each 3D point
- **Efficient Sampling**: Hierarchical sampling reduces computational cost
- **Differentiable Implementation**: Enables end-to-end training

The key insight is that we can represent any 3D scene as a continuous field of colors and densities, then use volume rendering to synthesize novel views by integrating along rays.


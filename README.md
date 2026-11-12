# splatity

A tiny gaussian splatting renderer, written by hand on the CPU to see how the
data actually becomes a picture. In the spirit of
[minity](https://github.com/quosa/minity): no GPU, no libraries, every pixel is
computed in plain TypeScript and put on a canvas.

![The scene rendered with gaussian splats](./doc/img/splatity.png)

The same scene with every splat shrunk to a fifth of its size, which shows what
the data really is: a cloud of small, flat, coloured blobs.

![The same scene with shrunken splats](./doc/img/splatity-points.png)

## Run it

```bash
npm install
```

```bash
npm run dev
```

Drag to orbit, scroll or pinch to zoom, and use the *Splat size* slider to
shrink the splats. `npm test` runs the unit tests and `npm run build` makes a
static site in `dist/`.

## What a splat scene is

There are no triangles, textures or lights. A scene is just a list of
gaussians, each with five properties (`src/scene.ts`):

| Property   | Meaning                                              |
| ---------- | ---------------------------------------------------- |
| `position` | centre of the blob in 3D                             |
| `scale`    | its size (standard deviation) along three local axes |
| `rotation` | a quaternion orienting those axes                    |
| `color`    | RGB                                                  |
| `opacity`  | how solid it is at its centre                        |

Real scenes are *trained*: an optimiser adjusts millions of these until
renders match a set of photographs. Here the roughly 5000 splats are generated
in code instead. Surfaces are made of thin pancakes lying flat on the surface;
the fog is made of big, round, nearly transparent blobs.

## How it is rendered

Everything is in `src/renderer.ts`, and follows the reference rasterizer from
[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
(Kerbl et al. 2023).

**0. Covariance (once).** Scale `S` and rotation `R` are combined into a 3x3
covariance matrix `Σ = R·S·Sᵀ·Rᵀ`, which describes the blob as an ellipsoid.

**1. Project.** Each centre goes through the usual perspective projection. The
ellipsoid is flattened to a 2D ellipse with `Σ' = J·W·Σ·Wᵀ·Jᵀ`, where `W` is
the camera rotation and `J` is the Jacobian of the perspective projection at
the splat centre (a local linear approximation, because a gaussian seen
through a true perspective projection is not a gaussian any more).

**2. Sort.** Splats are sorted by depth, nearest first. A z-buffer cannot do
this job, because blending transparent things depends on their order.

**3. Blend.** Each splat is drawn into the pixels it covers. For a pixel at
offset `d` from the splat centre:

```
alpha  = opacity · exp(-½ · dᵀ · Σ'⁻¹ · d)
colour += splatColour · alpha · T
T      *= (1 - alpha)
```

`T` is the pixel's transmittance: how much light still gets through
everything drawn so far. It starts at 1, and once it is nearly 0 the pixel is
finished and later splats skip it.

## What is left out

- **Spherical harmonics.** Real splats store colour as SH coefficients so it
  can change with the viewing direction (shiny things). Here each splat has
  one fixed colour with the lighting baked in.
- **Tiles and the GPU.** The reference renderer bins splats into 16x16 pixel
  tiles and runs on the GPU to handle millions of them. This one visits splats
  one by one, so it is limited to a few thousand at a small resolution.
- **Training.** Nothing is learned from photographs.
- **Loading files.** There is no `.ply` / `.splat` loader.

## Conventions

Right-handed world with +Y up, as in minity. Camera space is x right, y up and
z the distance in front of the camera. Screen coordinates start top-left.
Matrices are row-major.

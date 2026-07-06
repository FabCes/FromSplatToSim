# splat2sim — from real-world video to simulation-ready Gaussian Splat assets

Turn a smartphone or drone **video** of a real place into a **photorealistic, physics-enabled USD/USDZ asset** you can drop into **NVIDIA Isaac Sim** — and drive robots through it.

```
video (.mp4)                     ┌──────────────────────────────────────────┐
   │                             │  final asset (.usdz)                     │
   ▼                             │                                          │
frames ──► COLMAP poses ──►      │  /World  (Z-up, meters)                  │
3D Gaussian Splat training ──►   │   ├─ Visual/GaussianSplat   (render)     │
orientation + cleanup ──►        │   ├─ Physics/Collider       (invisible)  │
collider meshing + ground ──►    │   └─ Physics/GroundCollider (hole-free)  │
USD assembly ────────────────►   └──────────────────────────────────────────┘
```

---

## Why

**What is a gaussian splat?** 3D Gaussian Splatting (3DGS) represents a scene as millions of small, colored, semi-transparent 3D gaussians optimized so that rendering them reproduces your input photos. The result looks dramatically more real than a textured mesh reconstruction — reflections, vegetation, thin structures and soft materials all come out convincingly — and it renders in real time.

**The problem:** splats are volumetric particles, not surfaces. A physics engine like PhysX **cannot collide with them**. Import a splat into Isaac Sim and your robot drives straight through the world.

**The solution in this repo:** a **two-layer asset**. The gaussian splat is the *visual* layer (what cameras and humans see). Underneath, invisible and perfectly aligned, sits a *physics* layer: a collider mesh derived from the same data (what PhysX and contact sensors see). RTX lidar and cameras render against the visual/proxy geometry; rigid-body physics runs against the colliders. Both layers are Z-up, in meters, with semantic labels for synthetic data generation.

---

## The three tools

| Tool | Input | Output | GPU | When to use it |
|---|---|---|---|---|
| **splat2sim** | video (+ optional lidar) | final two-layer asset | yes (training) | End-to-end guided pipeline: you start from footage. |
| **splatlite** | existing COLMAP folder or trained `.ply` | clean, leveled splat `.usdz` + `.ply` | yes (training only) | You already have poses; you need training, orientation and parametric cleanup with instant 3D preview. |
| **splatmesh** | splat `.usdz` / clean `.ply` | final two-layer asset | **no** | You have a splat; derive colliders + ground and assemble the final asset in seconds. |

They compose: `splatlite` (clean splat) → `splatmesh` (colliders + assembly) is the lightweight path; `splat2sim` does everything in one guided flow and embeds all the features of the two lite tools (orientation, gaussians-based meshing, ground layer, artifact filters).

---

## Installation

Tested on Ubuntu 22.04 and 24.04. Training requires an NVIDIA GPU (≥ 8 GB VRAM recommended) with driver + CUDA toolkit 11.8/12.x; `splatmesh` needs no GPU at all.

### splat2sim (full pipeline)
```bash
tar xzf splat2sim.tar.gz && cd splat2sim
./install.sh                    # venv + clones/sets up 3dgrut and 2DGS backends
source .venv/bin/activate
splat2sim doctor                # checks ffmpeg, colmap, nvcc, GPU, backends
```
`install.sh` is idempotent and prints exactly what to fix (e.g. which CUDA toolkit package to install for your Ubuntu version); re-run it after fixing. System deps: `sudo apt install ffmpeg colmap` and the `uv` launcher for the 3dgrut installer.

### splatlite and splatmesh (standalone)
```bash
tar xzf splatlite.tar.gz && cd splatlite
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
splatlite                       # http://localhost:8081
```
```bash
tar xzf splatmesh.tar.gz && cd splatmesh
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
splatmesh-gui                   # http://localhost:8082   (CLI: splatmesh -h)
# optional smoother meshing engine: pip install -e ".[poisson]"
```
splatlite drives an **existing** 3dgrut checkout: set its path in section 0 of the GUI (persisted in `~/.config/splatlite.yaml`, or env `SPLATLITE_3DGRUT`).

---

## Capturing good footage

Reconstruction quality is decided at capture time. Move in **lateral arcs** around the subject (never rotate in place), keep 60fps or a fast shutter, lock exposure, do multiple passes at different heights, and favor texture-rich lighting. If the ground matters (it usually does), make sure it is actually visible in the footage. Verify early: run `ingest` + `poses` and check how many images COLMAP registered before spending GPU time on training.

---

## splat2sim — usage

```bash
splat2sim init myscene --video capture.mp4 [--lidar scan.ply] [--preset outdoor|object]
splat2sim run  myscene                    # all six steps, resumable
splat2sim run  myscene --step cleanup     # or one step at a time
splat2sim status myscene                  # what is done
splat2sim gui                             # web wizard at :8080 with live logs,
                                          # 3D preview, and every knob below
```

The six steps: **ingest** (frame extraction + blur filtering, lidar normalization) → **poses** (COLMAP SfM, GPU, sequential matching + loop detection, undistortion) → **train_splat** (3dgrut; also exports the native Isaac Sim splat `.usdz`) → **train_geo** (collider geometry) → **cleanup** (all filters below) → **export** (USD assembly + packaging).

### Orientation — level before cropping
COLMAP scenes come out arbitrarily tilted. The crop box is axis-aligned, so level first:
```bash
splat2sim orient myscene --auto            # RANSAC dominant plane → horizontal
splat2sim orient myscene --pca             # minimum-variance axis → +Z
splat2sim orient myscene --rotate 0,0,15   # manual degrees (X,Y,Z)
splat2sim orient myscene --ground-zero     # 5th height percentile → z=0
```
Rotations are applied to gaussian **positions and per-gaussian quaternions** and to the geometric mesh, so the two layers stay aligned. (Spherical harmonics of degree > 0 are not re-oriented — imperceptible for leveling-scale angles.) All of this is also in the GUI with one-click buttons.

### Configuration reference (`project.yaml`)

**geo** — how the collider geometry is produced:
- `engine: 2dgs` — second training pass + TSDF fusion. Best surface quality; costs GPU minutes. `tsdf_voxel` (smaller = finer mesh, more RAM) and `depth_trunc` control fusion.
- `engine: gaussians` — mesh straight from the trained splat: opacity-weighted voxel density + marching cubes. No GPU, seconds. `grid_res` (voxels along the scene diagonal, ≤ 448: higher = finer detail) and `iso` (density threshold 0–1: **lower = thicker, closed surfaces** — robust colliders; **higher = tighter fit** but possible holes).
- `engine: none` — visual-only asset.

**cleanup** — applied to splat and mesh:
- `min_opacity` (default 0.05): removes near-transparent "training dust". Beyond ~0.2 you may thin real semi-transparent surfaces.
- `max_scale`: removes gaussians larger than this (scene units) — giant blobs are typically sky or halos. Start near ¼ of the scene size, lower until they vanish. 0 = off.
- `outlier_k` / `outlier_std`: statistical floater removal — points more isolated than mean + σ·std of their k-neighbor distance go. σ 1–1.5 aggressive, 3+ conservative.
- `crop`: axis-aligned region of interest (`center`, `size`); everything outside is dropped from both layers. Use the GUI "Fit box to mesh" to initialize.
- `target_faces`: collider decimation budget (PhysX is comfortable at 100–300k for static scenes).
- **Component artifact filters** (structure collider): `min_extent` removes disconnected pieces whose bounding-box diagonal is below the threshold — localized blobs die regardless of face count. `max_hover` removes pieces floating more than this above the ground layer; `hover_max_extent` restricts the hover filter to components smaller than the given diagonal, so **extended elevated geometry (canopies, cables, elevated decks) is never removed** even when it doesn't touch the ground — an artifact is suspended *and* localized.

**ground** — optional dedicated terrain collider:
- `enabled: true` builds a regular heightfield over the **entire XY extent** — hole-free by construction — from a low `percentile` of heights per cell (the ground, ignoring anything built or grown above it), interpolating empty cells.
- `robust_band`: artifact filter for **misaligned cloud patches**. A smooth reference terrain is estimated from coherent cells only (masked normalized convolution with region growing); cells deviating beyond the band — pits *and* bumps, even extended ones — are corrected. Set it ≈ to the artifact amplitude you observe; lower = more aggressive. Real terrain, coherent at large scale, is untouched.
- `object_size`: morphological object filter (the standard DSM→DTM technique). Localized bumps with a footprint up to this size are not terrain and get flattened to the surrounding level; extended relief (hills, slopes) survives because it exceeds the window.
- `smooth`: final gaussian smoothing σ in cells. 2–3 keeps local undulation, 4–6 the general profile, 8–15 nearly a tilted plane.
- `grid_res`: heightfield resolution along the long side.

The ground enters the asset as a separate invisible collider `/World/Physics/GroundCollider`, semantically labeled `ground`. Note it can only read terrain where terrain is visible in the capture.

**export**:
- `mode: static` — exact-mesh colliders + a `PhysicsScene` (environments).
- `mode: dynamic` — rigid body with `convexDecomposition` approximation (manipulable objects).
- `scale_to_meters` — COLMAP scenes are scale-arbitrary: measure a known distance in scene units and set `real / measured`.
- `semantic_label` — written in both modern (`UsdSemantics.LabelsAPI`) and classic Isaac formats, for segmentation and synthetic data.

---

## splatlite — usage

Sections in the GUI (`splatlite`, port 8081):
0. **Configuration** — path to your 3dgrut checkout.
1. **Input** — train from a COLMAP folder (`images/` + `sparse/0/`) with live logs and iteration control, or load an existing `.ply`.
2. **Orientation** — RANSAC auto-level (reports inlier %), PCA, manual XYZ degrees, ground-to-z=0.
3. **Cleanup** — min opacity, max scale, k/σ outlier removal, crop box with robust fit-to-bounds; every slider explains its effect and when to move it.
4. **Preview** — kept centers in color vs removed in gray, red crop box.
5. **USDZ export** — writes the clean `.ply` and converts it to the Isaac Sim splat schema (`ParticleField3DGaussianSplat`). The bundled converter drives the 3dgrut API with camera export disabled, so it works from a bare PLY without the original dataset.

Keep the clean `.ply`: it is the best input for splatmesh.

## splatmesh — usage

GUI (`splatmesh-gui`, port 8082) mirrors the CLI: input → mesh parameters → optional ground → structure artifact filters → 3D preview (structure gray, ground green) → assembly. The generate→inspect→retune loop costs seconds.

CLI, full form:
```bash
splatmesh INPUT [-o OUT.usdz] [--visual SPLAT.usdz]
  --engine auto|voxel|poisson      # auto = poisson if open3d present, else voxel
  --grid-res 300 --iso 0.35        # voxel engine (see geo.gaussians above)
  --voxel SIZE                     # explicit voxel size instead of grid-res
  --poisson-depth 10               # poisson engine octree depth
  --target-faces 200000 --min-component-faces 300 --keep-components 0
  --ground --ground-smooth 4 --ground-res 120 --ground-percentile 5
  --ground-object-size 0 --ground-bump-thresh 0 --ground-robust-band 0
  --min-extent 0 --max-hover 0 --hover-max-extent 0
  --mode static|dynamic --scale 1.0 --label NAME
  --save-mesh collider.ply --save-ground ground.ply
```
Every parameter has the same meaning as in the splat2sim configuration reference above. Typical recipe: mesh from the clean `.ply`, visual from the `.usdz`, `--ground --ground-robust-band` ≈ artifact height, `--ground-object-size` ≈ largest clutter footprint, `--min-extent` ≈ largest blob, `--max-hover 0.3 --hover-max-extent` between your largest artifact and your smallest genuine elevated element.

---

## Importing into Isaac Sim

Requires Isaac Sim ≥ 5.0 for splat rendering (enable the *USD RTX NuRec schemas* extension, RTX – Real Time renderer).

1. Create a **new stage**.
2. **Drag the `.usdz` in as a reference** — never open it as the root stage (known limitation: the splat prim appears empty).
3. Verify physics with *Show → Physics → Colliders*: the invisible meshes render as overlay under the splat.
4. RTX lidar and cameras see the visual/proxy geometry; PhysX sees the colliders.
5. Check scale: if you didn't set it at export, measure a known distance and re-export with the right `scale_to_meters`.

---

## Troubleshooting

- **`doctor` says nvcc is too old** — install the official CUDA toolkit for your distro (it coexists with the apt one); `install.sh` prints the exact commands and resumes where it left off.
- **COLMAP registers few images** — footage too fast/blurry, low overlap or low parallax. Re-capture with lateral arcs and more frames (`ingest.target_frames`).
- **Collider has holes / misses thin parts** — lower `iso`, raise `grid_res`, or use the 2DGS/Poisson engines.
- **Ground bulges under structures** — the heightfield can only read terrain visible in the capture; include passes where the ground is seen, or raise `smooth` so coherent areas pull the interpolation.
- **Splat renders black in Isaac** — use the native export produced during training, make sure the NuRec extension is enabled, and reference (don't open) the file.

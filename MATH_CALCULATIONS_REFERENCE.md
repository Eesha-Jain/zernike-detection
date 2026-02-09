# Quick Reference: Zernike Moments Mathematical Calculations

## Location of Key Mathematical Operations

### File: `functions.py`

---

## Function: `ghosal_edge_v2()` (Lines 470-565)

### **STEP 1: Zernike Kernel Construction** (Lines 502-514)

**Purpose**: Create Zernike polynomial kernels for convolution

**Key Calculations**:
```python
# Normalize to unit circle
Kx = 2*j/Ks - offset
Ky = 2*i/Ks - offset

# Z₁₁ polynomial kernel (complex)
Vc11[i,j] = Kx - Ky*1j

# Z₂₀ polynomial kernel (real)
Vc20[i,j] = 2*Kx² + 2*Ky² - 1
```

**Mathematical Formula**:
- Z₁₁: T₁₁(r,θ) = r·exp(jθ) = (x + jy)
- Z₂₀: T₂₀(r,θ) = 2r² - 1

---

### **STEP 2: Zernike Moments Calculation** (Lines 524-528)

**Purpose**: Compute Zernike moments via convolution

**Key Calculations**:
```python
# Normalization factor
Anorm(n) = (n+1)/π

# Compute moments
A11 = (2/π) * convolve(img, Vc11)
A20 = (3/π) * convolve(img, Vc20)
```

**Mathematical Formula**:
```
Z_nm = (n+1)/π * A_nm
A_nm = ∫∫ f(u,v) * T_nm(u,v) du dv
```

---

### **STEP 3: Edge Parameter Extraction** (Lines 530-535)

**Purpose**: Extract edge angle, distance, and strength from moments

**Key Calculations**:

#### 3a. Edge Angle (Line 530)
```python
phi = arctan(Im(A11) / Re(A11))
```
**Formula**: φ = arg(A₁₁) = atan2(Im(A₁₁), Re(A₁₁))

#### 3b. Rotated Moment (Line 531)
```python
Al11 = Re(A11)*cos(phi) + Im(A11)*sin(phi)
```
**Formula**: A'₁₁ = Re(A₁₁)·cos(φ) + Im(A₁₁)·sin(φ)

#### 3c. Edge Distance (Line 532)
```python
l = A20 / Al11
```
**Formula**: l = A₂₀ / A'₁₁

#### 3d. Edge Strength (Line 535)
```python
k = |3*Al11 / (2*(1-l²)^(3/2))|
```
**Formula**: k = 3·A'₁₁ / (2·(1-l²)^(3/2))

---

### **STEP 4: Sub-Pixel Position Conversion** (Lines 555-557)

**Purpose**: Convert from unit circle coordinates to pixel coordinates

**Key Calculations**:
```python
i_s = i + l*Ks/2 * cos(phi)
j_s = j + l*Ks/2 * sin(phi)
```

**Mathematical Formula**:
```
[u_i]   [ũ_i]   N·l   [cos(φ)]
[v_i] = [ṽ_i] + --- * [sin(φ)]
                 2
```

Where:
- `(i, j)` = original pixel center
- `(i_s, j_s)` = sub-pixel edge position
- `l` = distance from center (in unit circle coordinates)
- `φ` = edge angle
- `Ks/2` = scaling factor (N/2 in formula)

---

## Summary of Mathematical Flow

```
Image → [Convolution with Zernike Kernels] → A₁₁, A₂₀
                                              ↓
                                    [Extract Parameters]
                                              ↓
                                    φ, A'₁₁, l, k
                                              ↓
                                    [Convert to Pixels]
                                              ↓
                                    Sub-pixel Edge Positions
```

---

## Coordinate System Notes

- **Image coordinates**: `(i, j)` where `i` = row, `j` = column
- **Output format**: `(j_s, i_s)` = `(x, y)` = `(column, row)`
- **Unit circle**: Normalized coordinates in range [-1, 1]
- **Kernel size**: `Ks` must be odd (typically 5, 7, or 9)

---

## Key Variables Reference

| Variable | Meaning | Range/Type |
|---------|---------|-------------|
| `Vc11` | Z₁₁ Zernike kernel | Complex array |
| `Vc20` | Z₂₀ Zernike kernel | Real array (stored as complex) |
| `A11` | Zernike moment Z₁₁ | Complex array |
| `A20` | Zernike moment Z₂₀ | Real array |
| `phi` (φ) | Edge angle | Radians |
| `Al11` (A'₁₁) | Rotated moment | Real array |
| `l` | Edge distance from center | [-1, 1] |
| `k` | Edge strength parameter | ≥ 0 |
| `Ks` | Kernel size | Odd integer (5, 7, 9, ...) |


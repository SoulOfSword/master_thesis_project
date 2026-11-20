"""Circular velocity in the disc plane of a galaxy.

Two methods:
    vcirc_disc_plane    from the snapshot's stored potential at the particles,
                        averaged in rings inside a thin slab about the plane
                        and differentiated numerically;
    vcirc2_components   from the softened gravity of the particles, summed
                        directly per mass component, averaged exactly over
                        circles in the plane.
"""

import math

import numpy as np
from numba import njit

# AREPO's gravitational constant, GRAVITY = 6.6738e-8 cgs with SOLAR_MASS =
# 1.989e33 g (allvars.h), in kpc (km/s)^2 / Msun
G_AREPO = 6.6738e-8 * 1.989e33 / (3.085678e21 * 1e10)

# Gauss-Legendre nodes per piece of the azimuthal integral for particles
# closer to the circle than their softening
N_GL = 48


def vcirc_disc_plane(xyz, phi, zhat, r_edges, slab_half=0.3, n_azimuth=12,
                     min_count=10):
    """Circular velocity in the disc plane from the gravitational potential.

    v_circ(R) = sqrt(R dPhi/dR), with Phi averaged azimuthally in cylindrical
    rings over the particles in the slab |z| < slab_half about the plane
    normal to `zhat`. Phi is averaged within `n_azimuth` sectors of each ring
    first and then over the sectors that hold particles, so the densest parts
    of a ring (arms, bar, clumps) do not dominate the mean.

    Parameters
    ----------
    xyz : array, shape (N, 3)
        Particle positions in kpc, centred on the galaxy.
    phi : array, shape (N,)
        Gravitational potential at the particles, physical (km/s)^2.
    zhat : array, shape (3,)
        Normal of the disc plane.
    r_edges : array
        Cylindrical radial ring edges in kpc.
    slab_half : float
        Slab half-thickness in kpc.
    n_azimuth : int
        Azimuthal sectors per ring.
    min_count : int
        Rings with fewer particles in the slab get nan.

    Returns
    -------
    R : array
        Mean cylindrical radius of the particles in each ring, kpc (nan for
        rings with fewer than `min_count` particles).
    v_circ : array
        Circular velocity in km/s; nan where R is nan or dPhi/dR <= 0.
    count : array
        Particles of each ring inside the slab.
    """
    xyz = np.asarray(xyz, dtype=float)
    phi = np.asarray(phi, dtype=float)
    zhat = np.asarray(zhat, dtype=float) / np.linalg.norm(zhat)

    z = xyz @ zhat # projection onto the normal
    slab = np.abs(z) < slab_half # masks particles in the slab
    xy = xyz[slab] - np.outer(z[slab], zhat) # projection onto the plane
    R = np.linalg.norm(xy, axis=1) # cylindrical radius of the particles in the slab
    e1 = np.cross(zhat, [1.0, 0.0, 0.0]) # in-plane axes for the azimuth
    if np.linalg.norm(e1) < 1e-6: # zhat is along x, use y instead
        e1 = np.cross(zhat, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1) # normalize
    e2 = np.cross(zhat, e1) # second in-plane axis
    theta = np.arctan2(xy @ e2, xy @ e1) # azimuth of the particles in the slab

    n_r = len(r_edges) - 1
    ir = np.digitize(R, r_edges) - 1
    ia = np.minimum(((theta + np.pi) / (2 * np.pi) * n_azimuth).astype(int), n_azimuth - 1) # azimuthal sector of each particle
    ok = (ir >= 0) & (ir < n_r) # masks particles in the correct rings and azimuthal sectors
    cell = ir[ok] * n_azimuth + ia[ok] # unique cell index for each particle in the slab, used for binning
    n_cell = np.bincount(cell, minlength=n_r * n_azimuth).reshape(n_r, n_azimuth) # number of particles in each ring and azimuthal sector
    phi_sum = np.bincount(cell, weights=phi[slab][ok],
                          minlength=n_r * n_azimuth).reshape(n_r, n_azimuth) # sum of potential in each ring and azimuthal sector
    filled = n_cell > 0
    sector_mean = np.divide(phi_sum, n_cell, out=np.zeros_like(phi_sum), where=filled)
    n_sectors = filled.sum(axis=1)
    phi_ring = np.divide(sector_mean.sum(axis=1), n_sectors,
                         out=np.full(n_r, np.nan), where=n_sectors > 0)
    count = n_cell.sum(axis=1) # number of particles in each ring inside the slab
    R_ring = np.bincount(ir[ok], weights=R[ok], minlength=n_r) / np.maximum(count, 1) # mean cylindrical radius of the particles in each ring

    good = np.flatnonzero(count >= min_count)
    R_out = np.full(n_r, np.nan) # mean cylindrical radius of the particles in each ring, nan for rings with fewer than `min_count` particles
    v_circ = np.full(n_r, np.nan) # circular velocity in km/s; nan where R is nan or dPhi/dR <= 0
    R_out[good] = R_ring[good]
    if len(good) >= 2:
        v2 = R_ring[good] * np.gradient(phi_ring[good], R_ring[good])
        v_circ[good] = np.sqrt(np.where(v2 > 0, v2, np.nan))
    return R_out, v_circ, count


@njit(cache=True)
def _spline_pull(d, h):
    """M(<d) / (m d^3) of AREPO's cubic-spline kernel with support h.

    The pull toward a particle of mass m at distance d is
    G m _spline_pull(d, h) d; Newtonian (1 / d^3) for d >= h. Coefficients as
    AREPO's SOFTFAC1-3 and SOFTFAC8-12 (allvars.h).
    """
    if d >= h:
        return 1.0 / (d * d * d)
    q = d / h
    if q < 0.5:
        return (10.666666666667 + q * q * (32.0 * q - 38.4)) / (h * h * h)
    return (21.333333333333 - 48.0 * q + 38.4 * q * q - 10.666666666667 * q * q * q
            - 0.066666666667 / (q * q * q)) / (h * h * h)


@njit(cache=True)
def _ellipke(m):
    """Complete elliptic integrals K(m) and E(m), parameter m = k^2, by the
    arithmetic-geometric mean."""
    a = 1.0
    b = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    s = 0.5 * m # running sum of 2^(n-1) c_n^2
    p = 1.0
    for _ in range(60):
        a, b, c = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        s += 0.5 * p * c * c
        if abs(c) <= 1e-16 * a:
            break
    K = 0.5 * math.pi / a
    return K, K * (1.0 - s)


@njit(cache=True)
def _azimuth_piece(a, z2, R, h, lo, hi, x_gl, w_gl):
    """Integral over azimuth D in [lo, hi] of the spline pull of a particle at
    cylindrical radius a and squared height z2, times R (R - a cos D), on the
    circle of radius R (Gauss-Legendre)."""
    half = 0.5 * (hi - lo)
    total = 0.0
    for g in range(x_gl.size):
        D = lo + half * (x_gl[g] + 1.0)
        cos_D = math.cos(D)
        d = math.sqrt(max(a * a + R * R - 2.0 * a * R * cos_D + z2, 0.0))
        total += w_gl[g] * _spline_pull(d, h) * R * (R - a * cos_D)
    return half * total


@njit(cache=True)
def _ring_sum(R_cyl, z2, mass, h, comp, n_comp, radii, h_target, x_gl, w_gl):
    """pi / G times <R dPhi/dR> on each circle, per component."""
    out = np.zeros((radii.size, n_comp))
    for j in range(radii.size):
        R = radii[j]
        for i in range(R_cyl.size):
            a = R_cyl[i]
            hh = max(h[i], h_target) # softening of the pair, as in AREPO
            d2 = (a - R) * (a - R) + z2[i] # squared distance to the circle
            if d2 >= hh * hh:
                # Newtonian on the whole circle: exact ring formula
                s2 = (R + a) * (R + a) + z2[i]
                K, E = _ellipke(4.0 * a * R / s2)
                pull = (K - (a * a - R * R + z2[i]) / d2 * E) / math.sqrt(s2)
            else:
                # within h of the circle: integrate the spline pull over
                # azimuth, split at the angle D_h where the distance equals h
                c = (a * a + R * R + z2[i] - hh * hh) / (2.0 * a * R) if a > 0.0 else -1.0
                D_h = math.acos(min(1.0, max(-1.0, c)))
                pull = (_azimuth_piece(a, z2[i], R, hh, 0.0, D_h, x_gl, w_gl)
                        + _azimuth_piece(a, z2[i], R, hh, D_h, math.pi, x_gl, w_gl))
            out[j, comp[i]] += mass[i] * pull
    return out


def vcirc2_components(xyz, mass, h, comp, zhat, radii, n_comp, h_target=0.0,
                      n_gl=N_GL, G=G_AREPO):
    """Squared circular velocity in the disc plane per mass component, from the
    softened gravity of the particles.

    v_circ^2(R) = R dPhi/dR on the circle of radius R in the plane z = 0
    normal to `zhat`, averaged exactly over azimuth and summed directly over
    the particles. Each particle pulls with AREPO's cubic-spline softened
    gravity of support h (Newtonian beyond h). For a particle farther than h
    from the circle the azimuthal average is the ring formula
        R dPhi/dR = G m / (pi s) [K(k) - (a^2 - R^2 + z^2) / ((a - R)^2 + z^2) E(k)],
        s^2 = (R + a)^2 + z^2,  k^2 = 4 a R / s^2,
    with a and z its cylindrical radius and height; for a nearer particle the
    spline pull is integrated over azimuth by Gauss-Legendre quadrature, split
    at the angle where the distance equals h. Each pair is softened with
    max(h_target, h), as AREPO uses the larger of the two softenings;
    h_target = 0 gives the field of the particles alone.

    Parameters
    ----------
    xyz : array, shape (N, 3)
        Particle positions in kpc, centred on the galaxy.
    mass : array, shape (N,)
        Particle masses in Msun.
    h : array, shape (N,)
        Spline support of each particle's softening (2.8 x the
        Plummer-equivalent softening) in kpc.
    comp : array, shape (N,)
        Component of each particle, an integer from 0 to n_comp - 1.
    zhat : array, shape (3,)
        Normal of the disc plane.
    radii : array
        Radii of the circles in kpc.
    n_comp : int
        Number of components.
    h_target : float
        Softening (spline support) of the evaluation point in kpc.
    n_gl : int
        Gauss-Legendre nodes per piece of the azimuthal integral.
    G : float
        Gravitational constant in kpc (km/s)^2 / Msun.

    Returns
    -------
    v2 : array, shape (len(radii), n_comp)
        v_circ^2 of each component in (km/s)^2; negative where the
        component's net pull on the circle points outward.
    """
    xyz = np.asarray(xyz, dtype=float)
    mass = np.asarray(mass, dtype=float)
    h = np.asarray(h, dtype=float)
    comp = np.asarray(comp, dtype=np.int64)
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    zhat = np.asarray(zhat, dtype=float) / np.linalg.norm(zhat)
    n = len(xyz)
    if xyz.shape != (n, 3) or mass.shape != (n,) or h.shape != (n,) or comp.shape != (n,):
        raise ValueError("xyz (N, 3), mass, h and comp (N,) must describe the same particles")
    if n and (comp.min() < 0 or comp.max() >= n_comp):
        raise ValueError(f"component indices must lie in 0..{n_comp - 1}")
    if np.any(h < 0) or h_target < 0 or np.any(radii <= 0):
        raise ValueError("softenings must be >= 0 and radii > 0")

    z = xyz @ zhat # height above the disc plane
    R_cyl = np.sqrt(np.maximum(np.einsum("ij,ij->i", xyz, xyz) - z * z, 0.0)) # cylindrical radius
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_gl)
    return G / np.pi * _ring_sum(R_cyl, z * z, mass, h, comp, n_comp, radii,
                                 float(h_target), x_gl, w_gl)

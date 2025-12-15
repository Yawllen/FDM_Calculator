# thumbnail_core.py
# -*- coding: utf-8 -*-
"""
thumbnail_core — лёгкий модуль превью 3D-моделей для карточек заказов (server-side, CPU-only).

Задача:
- получить "нормальное" превью модели (узнаваемая 3D-форма)
- максимально быстро и стабильно на сервере без GPU / OpenGL
- безопасно для сервера (лимиты по размеру файла/сложности + деградация качества)
- один модуль, который можно вызывать из UI и CLI

Поддержка форматов:
- STL (.stl)
- 3MF (.3mf)  (через trimesh; если импорт 3MF не поддержан в окружении — вернём placeholder)

Зависимости:
  pip install numpy pillow trimesh
Опционально (очень рекомендуется для хорошего качества на тяжёлых моделях):
  pip install fast-simplification

Примечание:
Это НЕ "настоящий рендерер". Это быстрый генератор thumbnail'ов:
- для лёгких мешей: быстрый CPU "triangle render"
- для тяжёлых: decimation (упрощение) -> triangle render
- если decimation недоступен/упал: depthmap fallback (цельный объём, но менее детально)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, Union
import os
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import trimesh

try:
    import fast_simplification  # type: ignore
except Exception:
    fast_simplification = None  # type: ignore


# ---------------------------- Public API ----------------------------

@dataclass(frozen=True)
class ThumbLimits:
    """Лимиты безопасности (анти-DoS + предсказуемое время работы)."""
    max_file_mb: int = 200                 # максимальный размер файла модели (MB)
    max_faces_hard: int = 3_000_000         # выше — пропускаем decimation и идём в depthmap fallback
    max_faces_render: int = 120_000         # сколько граней максимум рендерим треугольниками
    target_faces_simplify: int = 100_000    # цель для decimation
    max_points_depthmap: int = 280_000      # максимум точек для depthmap (больше -> дороже)
    min_points_depthmap: int = 80_000       # минимум точек для depthmap (меньше -> дырки)
    # на практике можно добавить таймауты, но в чистом Python это требует сигналов/процессов


@dataclass(frozen=True)
class ThumbStyle:
    """Визуальные параметры (нейтральный стиль + читаемость)."""
    size_px: int = 320
    padding: float = 0.08
    bg_rgb: Tuple[int, int, int] = (248, 249, 251)

    # grayscale range for shaded object
    fg_min: int = 90
    fg_max: int = 235

    # outline/shadow
    outline_rgb: Tuple[int, int, int] = (55, 55, 55)
    shadow_strength: float = 0.35  # 0..1
    shadow_blur: int = 6
    shadow_offset: Tuple[int, int] = (6, 8)

    # light direction (for shading)
    light_dir: Tuple[float, float, float] = (-0.30, -0.55, 0.78)

    # "camera" tilt to make it look 3D (cheap)
    yaw_deg: float = 35.0
    pitch_deg: float = 22.0


@dataclass
class ThumbResult:
    """Результат генерации (для логов и UI)."""
    status: str                 # normal | simplified | fallback | placeholder | error
    png_bytes: bytes
    meta: Dict[str, Any]


def supported_exts() -> Tuple[str, ...]:
    return (".stl", ".3mf")


def generate_thumbnail_bytes(
    model_path: str,
    *,
    limits: ThumbLimits = ThumbLimits(),
    style: ThumbStyle = ThumbStyle(),
) -> ThumbResult:
    """
    Сгенерировать PNG thumbnail и вернуть bytes + метаданные.

    Политика:
      - Если меш <= max_faces_render: triangle-render (normal)
      - Иначе: real decimation (fast-simplification / trimesh) -> triangle-render (simplified)
      - Если decimation недоступен/сломался: depthmap fallback (fallback)
      - Если файл слишком большой/битый: placeholder или error
    """
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "path": model_path,
        "ext": os.path.splitext(model_path)[1].lower(),
        "file_mb": None,
        "faces_in": None,
        "faces_used": None,
        "mode": None,
        "ms": None,
        "error": None,
    }

    # Быстрый guard по размеру файла
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        meta["file_mb"] = round(size_mb, 3)
        if size_mb > limits.max_file_mb:
            png = _placeholder_png(style, text="FILE TOO LARGE")
            meta["mode"] = "placeholder"
            meta["ms"] = _ms(t0)
            return ThumbResult(status="placeholder", png_bytes=png, meta=meta)
    except OSError as e:
        meta["error"] = f"stat_failed: {e}"
        png = _placeholder_png(style, text="NO FILE")
        meta["mode"] = "placeholder"
        meta["ms"] = _ms(t0)
        return ThumbResult(status="error", png_bytes=png, meta=meta)

    # Загружаем меш (STL/3MF)
    mesh = _load_mesh(model_path, meta)
    if mesh is None:
        # meta уже заполнена ошибкой
        png = _placeholder_png(style, text="BAD MODEL")
        meta["mode"] = "placeholder"
        meta["ms"] = _ms(t0)
        return ThumbResult(status="error", png_bytes=png, meta=meta)

    # Минимальный cleanup (дёшево, но полезно)
    try:
        mesh = mesh.copy()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    faces_in = int(getattr(mesh.faces, "shape", [0])[0])
    meta["faces_in"] = faces_in

    # Слишком огромный меш: сразу depthmap (decimation может быть дорогим/нестабильным)
    if faces_in > limits.max_faces_hard:
        png = _render_depthmap_png(mesh, limits=limits, style=style)
        meta["mode"] = "fallback_depthmap"
        meta["ms"] = _ms(t0)
        return ThumbResult(status="fallback", png_bytes=png, meta=meta)

    # Лёгкий меш: triangle render
    if faces_in <= limits.max_faces_render:
        png = _render_triangles_png(mesh, style=style)
        meta["mode"] = "normal"
        meta["faces_used"] = faces_in
        meta["ms"] = _ms(t0)
        return ThumbResult(status="normal", png_bytes=png, meta=meta)

    # Тяжёлый меш: decimation -> triangle render
    target = min(limits.target_faces_simplify, limits.max_faces_render)
    dec = _try_decimate_best(mesh, target_faces=target)
    if dec is not None:
        used = int(dec.faces.shape[0])
        png = _render_triangles_png(dec, style=style)
        meta["mode"] = "simplified"
        meta["faces_used"] = used
        meta["ms"] = _ms(t0)
        return ThumbResult(status="simplified", png_bytes=png, meta=meta)

    # Если decimator не установлен или упал — depthmap fallback
    png = _render_depthmap_png(mesh, limits=limits, style=style)
    meta["mode"] = "fallback_depthmap"
    meta["ms"] = _ms(t0)
    return ThumbResult(status="fallback", png_bytes=png, meta=meta)


def generate_thumbnail_file(
    model_path: str,
    out_png_path: str,
    *,
    limits: ThumbLimits = ThumbLimits(),
    style: ThumbStyle = ThumbStyle(),
    make_dirs: bool = True,
) -> ThumbResult:
    """Сгенерировать thumbnail и сохранить PNG на диск."""
    if make_dirs:
        os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)

    res = generate_thumbnail_bytes(model_path, limits=limits, style=style)
    with open(out_png_path, "wb") as f:
        f.write(res.png_bytes)
    res.meta["out_png"] = out_png_path
    return res


def default_config_dict() -> Dict[str, Any]:
    """Экспорт дефолтов — удобно для логов/конфигов."""
    return {"limits": asdict(ThumbLimits()), "style": asdict(ThumbStyle())}


# ---------------------------- Mesh loading ----------------------------

def _load_mesh(path: str, meta: Dict[str, Any]) -> Optional[trimesh.Trimesh]:
    """
    Загрузить STL/3MF в один Trimesh.
    Возвращает None при ошибке (meta['error'] заполнится).

    Важное:
    - 3MF часто загружается как Scene -> нужно объединить.
    - trimesh сам парсит 3MF, но в некоторых окружениях могут понадобиться доп. зависимости.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in supported_exts():
        meta["error"] = f"unsupported_ext: {ext}"
        return None

    try:
        obj: Union[trimesh.Trimesh, trimesh.Scene] = trimesh.load(path, force=None)
    except Exception as e:
        meta["error"] = f"load_failed: {type(e).__name__}: {e}"
        return None

    try:
        if isinstance(obj, trimesh.Scene):
            # dump(concatenate=True) обычно даёт меш в мировых координатах
            if hasattr(obj, "dump"):
                dumped = obj.dump(concatenate=True)
                if isinstance(dumped, trimesh.Trimesh) and not dumped.is_empty:
                    return dumped
            # fallback: concat geometries (может проигнорировать трансформации, но лучше чем ничего)
            geoms = list(obj.geometry.values())
            if not geoms:
                meta["error"] = "scene_empty"
                return None
            return trimesh.util.concatenate(geoms)  # type: ignore
        if isinstance(obj, trimesh.Trimesh):
            if obj.is_empty:
                meta["error"] = "mesh_empty"
                return None
            return obj
    except Exception as e:
        meta["error"] = f"scene_merge_failed: {type(e).__name__}: {e}"
        return None

    meta["error"] = "unknown_load_type"
    return None


# ---------------------------- Decimation ----------------------------

def _try_decimate_best(mesh: trimesh.Trimesh, target_faces: int) -> Optional[trimesh.Trimesh]:
    """
    Лучший decimator:
    1) fast-simplification (быстро + качественно)
    2) trimesh simplify_quadratic_decimation (если доступен backend)
    """
    dec = _try_fast_simplification(mesh, target_faces=target_faces)
    if dec is not None:
        return dec
    return _try_trimesh_decimation(mesh, target_faces=target_faces)


def _try_fast_simplification(mesh: trimesh.Trimesh, target_faces: int) -> Optional[trimesh.Trimesh]:
    """
    fast-simplification: топ для наших задач.

    API библиотеки (в большинстве версий):
      v_out, f_out = fast_simplification.simplify(vertices, faces, target_reduction)

    Где target_reduction — доля граней, которые нужно удалить (0..1).
    """
    if fast_simplification is None:
        return None

    faces_in = int(mesh.faces.shape[0])
    if faces_in <= target_faces:
        return mesh

    # насколько нужно "сжать" меш
    target_reduction = 1.0 - (float(target_faces) / float(max(1, faces_in)))
    target_reduction = float(np.clip(target_reduction, 0.0, 0.99))

    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)

        out = fast_simplification.simplify(v, f, target_reduction)  # type: ignore
        # some versions return tuple (v,f); be defensive
        if isinstance(out, tuple) and len(out) == 2:
            v_out, f_out = out
        else:
            return None

        v_out = np.asarray(v_out, dtype=np.float64)
        f_out = np.asarray(f_out, dtype=np.int64)

        if v_out.size == 0 or f_out.size == 0:
            return None

        dec = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
        _cheap_cleanup(dec)
        if dec.is_empty or dec.faces.shape[0] <= 0:
            return None
        return dec
    except Exception:
        return None


def _try_trimesh_decimation(mesh: trimesh.Trimesh, target_faces: int) -> Optional[trimesh.Trimesh]:
    """Fallback: trimesh.simplify_quadratic_decimation (может не работать без backend)."""
    try:
        if hasattr(mesh, "simplify_quadratic_decimation"):
            dec = mesh.simplify_quadratic_decimation(target_faces)
            if isinstance(dec, trimesh.Trimesh) and not dec.is_empty and dec.faces.shape[0] > 0:
                _cheap_cleanup(dec)
                return dec
    except Exception:
        return None
    return None


def _cheap_cleanup(mesh: trimesh.Trimesh) -> None:
    """Дёшевый cleanup после decimation."""
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass


# ---------------------------- Rendering helpers ----------------------------

def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= 1e-12 else (v / n)


def _pca_axes(points: np.ndarray) -> np.ndarray:
    """PCA-оси для стабильного вида (уменьшают случайные повороты)."""
    X = points - points.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    _, eigvecs = np.linalg.eigh(cov)  # ascending
    axes = eigvecs[:, ::-1]          # descending

    # стабилизация знака (чтобы не флипало между запусками)
    if axes[0, 0] < 0:
        axes[:, 0] *= -1
    if axes[1, 1] < 0:
        axes[:, 1] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return axes


def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Центр + масштабирование к unit box (max dimension = 1)."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    v = vertices - center
    scale = float(np.max(vmax - vmin))
    if scale <= 1e-12:
        scale = 1.0
    return v / scale


def _apply_tilt(points: np.ndarray, *, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Дешёвый фиксированный наклон камеры для объёмности."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cp, -sp],
                   [0.0,  sp,  cp]], dtype=np.float64)
    return points @ (Rz @ Rx)


def _render_triangles_png(mesh: trimesh.Trimesh, *, style: ThumbStyle) -> bytes:
    """
    Быстрый CPU triangle-render:
    - PCA + наклон
    - ортографическая проекция
    - Painter's algorithm + простой шейдинг (Lambert)
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return _placeholder_png(style, text="EMPTY")

    verts = _normalize_vertices(verts)
    axes = _pca_axes(verts)
    vr = verts @ axes
    vr = _apply_tilt(vr, yaw_deg=style.yaw_deg, pitch_deg=style.pitch_deg)

    tri = vr[faces]  # (F,3,3)
    xy = tri[:, :, :2].copy()
    z = tri[:, :, 2].mean(axis=1)

    # normals for shading
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nlen == 0, 0, n / np.maximum(nlen, 1e-12))

    light = _unit(np.array(style.light_dir, dtype=np.float64))
    intensity = (n @ light).reshape(-1)
    intensity = np.clip(intensity, 0.0, 1.0)
    intensity = np.power(intensity, 0.75)  # contrast curve

    order = np.argsort(z)  # far -> near

    # fit to canvas
    xy_flat = xy.reshape(-1, 2)
    xy_min = xy_flat.min(axis=0)
    xy_max = xy_flat.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-9)

    S = int(style.size_px)
    pad = float(style.padding)
    k = (1.0 - 2.0 * pad) * (S - 1) / float(np.max(span))

    def to_px(p: np.ndarray) -> np.ndarray:
        q = (p - xy_min) * k
        q[:, 1] = (span[1] * k) - q[:, 1]  # flip Y
        q = q + np.array([pad * (S - 1), pad * (S - 1)])
        return q

    img = Image.new("RGB", (S, S), style.bg_rgb)
    draw = ImageDraw.Draw(img)

    fg_min = int(style.fg_min)
    fg_max = int(style.fg_max)

    for i in order:
        pts = to_px(xy[i])
        shade = fg_min + int((fg_max - fg_min) * float(intensity[i]))
        c = (shade, shade, shade)
        draw.polygon((float(pts[0, 0]), float(pts[0, 1]),
                      float(pts[1, 0]), float(pts[1, 1]),
                      float(pts[2, 0]), float(pts[2, 1])), fill=c)

    return _encode_png(img)


def _render_depthmap_png(mesh: trimesh.Trimesh, *, limits: ThumbLimits, style: ThumbStyle) -> bytes:
    """
    Fallback "солид" через depthmap (z-buffer по точкам поверхности).
    Это дешевле/стабильнее, чем пытаться рендерить миллионы треугольников.

    Визуально: цельный объём + outline + мягкая тень.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return _placeholder_png(style, text="EMPTY")

    faces_n = int(faces.shape[0])
    n_points = int(min(max(limits.min_points_depthmap, faces_n // 2), limits.max_points_depthmap))

    try:
        pts = mesh.sample(n_points)  # area-weighted
    except Exception:
        # fallback: vertices
        if verts.shape[0] > n_points:
            rng = np.random.default_rng(123)
            idx = rng.choice(verts.shape[0], size=n_points, replace=False)
            pts = verts[idx]
        else:
            pts = verts

    pts = np.asarray(pts, dtype=np.float64)
    pts = _normalize_vertices(pts)

    axes = _pca_axes(pts)
    pr = pts @ axes
    pr = _apply_tilt(pr, yaw_deg=style.yaw_deg, pitch_deg=style.pitch_deg)

    xy = pr[:, :2]
    z = pr[:, 2]

    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1e-9)

    S = int(style.size_px)
    pad = float(style.padding)
    k = (1.0 - 2.0 * pad) * (S - 1) / float(np.max(span))

    q = (xy - xy_min) * k
    q[:, 1] = (span[1] * k) - q[:, 1]
    q += np.array([pad * (S - 1), pad * (S - 1)])

    xi = np.clip(q[:, 0].astype(np.int32), 0, S - 1)
    yi = np.clip(q[:, 1].astype(np.int32), 0, S - 1)

    # z-buffer: front-most (max z)
    zbuf = np.full((S, S), -np.inf, dtype=np.float32)
    np.maximum.at(zbuf, (yi, xi), z.astype(np.float32))
    mask = np.isfinite(zbuf)

    if not mask.any():
        return _placeholder_png(style, text="PREVIEW")

    zmin = float(np.min(zbuf[mask]))
    zmax = float(np.max(zbuf[mask]))
    denom = (zmax - zmin) if (zmax - zmin) > 1e-9 else 1.0
    zn = (zbuf - zmin) / denom
    zn[~mask] = 0.0

    # fill small holes + smooth
    m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    m_img = m_img.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    mask2 = (np.array(m_img) > 0)

    d_img = Image.fromarray((zn * 255).astype(np.uint8), mode="L")
    d_img = d_img.filter(ImageFilter.BoxBlur(2))
    zn2 = (np.array(d_img).astype(np.float32) / 255.0)

    # pseudo normals from depth gradients
    dz_dy, dz_dx = np.gradient(zn2)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(nx)

    nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx /= nlen
    ny /= nlen
    nz /= nlen

    light = _unit(np.array(style.light_dir, dtype=np.float32))
    inten = nx * light[0] + ny * light[1] + nz * light[2]
    inten = np.clip(inten, 0.0, 1.0)
    inten = np.power(inten, 0.75)

    fg_min = float(style.fg_min)
    fg_max = float(style.fg_max)
    shade = (fg_min + (fg_max - fg_min) * inten).astype(np.uint8)

    # shadow: blurred mask offset
    shadow = m_img.filter(ImageFilter.GaussianBlur(int(style.shadow_blur)))
    shadow_np = (np.array(shadow).astype(np.float32) / 255.0) * float(style.shadow_strength)
    dx, dy = style.shadow_offset

    shadow_canvas = np.zeros((S, S), dtype=np.float32)
    y0 = max(0, dy)
    x0 = max(0, dx)
    shadow_canvas[y0:, x0:] = shadow_np[: S - y0, : S - x0]
    shadow_canvas = np.clip(shadow_canvas, 0.0, 0.8)

    bg = np.array(style.bg_rgb, dtype=np.float32)
    img = np.zeros((S, S, 3), dtype=np.float32)
    img[:, :] = bg

    # apply shadow
    img *= (1.0 - shadow_canvas[:, :, None])

    # paint object
    obj = np.stack([shade, shade, shade], axis=-1).astype(np.float32)
    img[mask2] = obj[mask2]

    # outline: edge = mask - eroded(mask)
    er = m_img.filter(ImageFilter.MinFilter(3))
    edge = (np.array(m_img) > 0) & (np.array(er) == 0)
    img[edge] = np.array(style.outline_rgb, dtype=np.float32)

    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    return _encode_png(out)


def _placeholder_png(style: ThumbStyle, *, text: str = "PREVIEW") -> bytes:
    """Всегда успешная заглушка."""
    S = int(style.size_px)
    img = Image.new("RGB", (S, S), style.bg_rgb)
    draw = ImageDraw.Draw(img)

    pad = max(8, S // 20)
    draw.rectangle((pad, pad, S - pad, S - pad), outline=(200, 200, 200), width=2)

    msg = (text or "PREVIEW")[:18]
    tw = 6 * len(msg)
    th = 10
    draw.text(((S - tw) / 2, (S - th) / 2), msg, fill=(120, 120, 120))

    return _encode_png(img)


def _encode_png(img: Image.Image) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

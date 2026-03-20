"""
ballDetector.py
───────────────
Non-AI ping-pong ball detector and APF movement-vector generator.

Key fixes vs previous version (informed by working reference):
  1. Gaussian blur applied to BGR frame BEFORE HSV conversion — critical for
     stable colour thresholding.
  2. Pre-computed remap maps (faster + alpha=0 crops black borders).
  3. Obstacle detection gates on sat >= min_saturation — white floor (S~0)
     is automatically excluded with no separate floor mask needed.
  4. Balls are NOT filtered by proximity to obstacles.  Instead, detected
     ball pixels are masked out of the obstacle detector.
  5. Looser thresholds (circularity 0.58, morph iterations=1, min area 90)
     matching the working reference.

Public API
----------
    from ballConfig import CONFIG
    from ballDetector import PingPongDetector, MovementVector

    detector = PingPongDetector(CONFIG)
    vectors  = detector.analyze_frame(bgr_frame)   # List[MovementVector]

    # Debug masks for /masks endpoint:
    masks = detector.get_debug_masks(bgr_frame)
    # masks["ball"]     — binary mask of ball-colour pixels
    # masks["obstacle"] — binary mask of obstacle pixels

Each MovementVector:
    .angle       — degrees from forward axis, range (-180, 180]
    .magnitude   — normalised APF magnitude in [min_magnitude, 1.0]
    .distance_cm — estimated real-world distance
    .target_px   — (x, y) ball centre in image pixels

List is sorted closest-first.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedBall:
    center:      Tuple[int, int]
    radius_px:   float
    distance_cm: float


@dataclass
class DetectedObstacle:
    center:  Tuple[int, int]
    contour: np.ndarray
    area_px: float


@dataclass
class MovementVector:
    angle:       float
    magnitude:   float
    distance_cm: float
    target_px:   Tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class PingPongDetector:
    """
    Detects orange ping-pong balls and coloured obstacles from BGR frames,
    then returns APF movement vectors toward each detected ball.
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, config: "dict | str") -> None:
        """
        Parameters
        ----------
        config : dict or str
            The CONFIG dict from ballConfig.py, or its module name.

        Examples
        --------
        from ballConfig import CONFIG
        det = PingPongDetector(CONFIG)

        det = PingPongDetector("ballConfig")
        """
        if isinstance(config, str):
            module = importlib.import_module(config)
            cfg: dict = module.CONFIG
        else:
            cfg = config

        self._load_camera(cfg["camera"])
        self._load_preprocess(cfg)
        self._load_ball_params(cfg["ping_pong"])
        self._load_obstacle_params(cfg["obstacles"])
        self._load_morph_params(cfg["morphology"])
        self._load_apf_params(cfg["apf"])

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> List[MovementVector]:
        """
        Detect balls and return APF vectors, sorted closest-first.

        Parameters
        ----------
        frame : np.ndarray  —  raw BGR frame from webcam

        Returns
        -------
        List[MovementVector]
        """
        if frame is None or frame.size == 0:
            return []

        undistorted = self._undistort(frame)

        # Blur BEFORE HSV — this is the critical fix.
        # Blurring smooths colour noise and makes inRange thresholding
        # far more stable across lighting variation.
        blurred = cv2.GaussianBlur(undistorted, self._blur_k, 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Ball detection + accepted-ball pixel mask
        balls, ball_px_mask = self._detect_balls(hsv)

        # Obstacle detection — ball pixels are excluded from the mask
        obstacles = self._detect_obstacles(hsv, ball_px_mask)

        vectors = self._compute_apf_vectors(balls, obstacles)
        vectors.sort(key=lambda v: v.distance_cm)
        return vectors

    def get_debug_masks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return intermediate binary masks for the /masks debug endpoint.

        Returns
        -------
        dict with keys "ball" and "obstacle" (single-channel uint8).
        """
        blank = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
        if frame is None or frame.size == 0:
            return {"ball": blank, "obstacle": blank.copy()}

        undistorted = self._undistort(frame)
        blurred = cv2.GaussianBlur(undistorted, self._blur_k, 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        ball_mask, ball_px_mask = self._ball_mask(hsv)
        obs_mask  = self._obstacle_mask(hsv, ball_px_mask)
        return {"ball": ball_mask, "obstacle": obs_mask}

    # ── Config loading ────────────────────────────────────────────────────

    def _load_camera(self, cam: dict) -> None:
        self._frame_w = int(cam["frame_width"])
        self._frame_h = int(cam["frame_height"])
        cam_mtx  = np.array(cam["camera_matrix"], dtype=np.float64)
        dist     = np.array(cam["dist_coeffs"],   dtype=np.float64)
        size     = (self._frame_w, self._frame_h)

        # Pre-compute remap maps once (faster than cv2.undistort per frame,
        # and alpha=0 crops out black undistortion borders)
        optimal, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, size, alpha=0.0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            cam_mtx, dist, None, optimal, size, cv2.CV_16SC2
        )
        self._fx = float(optimal[0, 0])
        self._cx = float(optimal[0, 2])

    def _load_preprocess(self, cfg: dict) -> None:
        k = cfg.get("gaussian_blur_kernel", [7, 7])
        self._blur_k = (int(k[0]), int(k[1]))

    def _load_ball_params(self, pp: dict) -> None:
        self._ball_diam_mm    = float(pp["diameter_mm"])
        self._ball_lower      = np.array(pp["ball_hsv_lower"], dtype=np.uint8)
        self._ball_upper      = np.array(pp["ball_hsv_upper"], dtype=np.uint8)
        self._min_area        = int(pp.get("min_area_px2",   90))
        self._max_area        = int(pp.get("max_area_px2",   80000))
        self._min_radius_px   = float(pp.get("min_radius_px", 7.0))
        self._max_radius_px   = float(pp.get("max_radius_px", 160.0))
        self._circularity_min = float(pp.get("circularity_threshold", 0.58))

    def _load_obstacle_params(self, obs: dict) -> None:
        self._obs_min_area   = int(obs.get("min_area_px2",    1200))
        self._obs_max_area   = int(obs.get("max_area_px2",   120000))
        self._obs_min_val    = int(obs.get("min_value",         15))
        self._obs_max_val    = int(obs.get("max_value",        210))
        self._obs_min_sat    = int(obs.get("min_saturation",    35))
        self._top_ignore     = float(obs.get("top_ignore_ratio", 0.12))

    def _load_morph_params(self, m: dict) -> None:
        ok = tuple(m.get("open_kernel",  [3, 3]))
        ck = tuple(m.get("close_kernel", [5, 5]))
        it = int(m.get("iterations", 1))
        self._open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ok)
        self._close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ck)
        self._morph_iter = it

    def _load_apf_params(self, apf: dict) -> None:
        self._k_att            = float(apf.get("k_att",            1.0))
        self._k_rep            = float(apf.get("k_rep",            0.75))
        self._rep_influence_cm = float(apf.get("rep_influence_cm", 30.0))
        self._min_magnitude    = float(apf.get("min_magnitude",    0.18))

        rp = apf.get("robot_position", "bottom_center")
        if rp == "bottom_center":
            self._robot_px: Tuple[int, int] = (self._frame_w // 2, self._frame_h - 1)
        elif rp == "center":
            self._robot_px = (self._frame_w // 2, self._frame_h // 2)
        else:
            self._robot_px = (int(rp[0]), int(rp[1]))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._open_k,  iterations=self._morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k, iterations=self._morph_iter)
        return mask

    # ── Ball detection ────────────────────────────────────────────────────

    def _ball_mask(self, hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (cleaned ball mask, accepted-ball pixel mask).

        The accepted-ball pixel mask paints a filled circle ~20% larger than
        each detected ball.  The obstacle detector zeros these pixels out,
        preventing balls from being classified as obstacles.
        """
        raw_mask  = cv2.inRange(hsv, self._ball_lower, self._ball_upper)
        ball_mask = self._clean_mask(raw_mask)

        contours, _ = cv2.findContours(
            ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        accepted_px = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1e-6:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < self._circularity_min:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (self._min_radius_px <= radius <= self._max_radius_px):
                continue
            # Paint a slightly enlarged circle so the obstacle detector
            # doesn't see the ball edge as a coloured object
            cv2.circle(accepted_px, (int(cx), int(cy)), int(radius * 1.2), 255, -1)

        return ball_mask, accepted_px

    def _detect_balls(self, hsv: np.ndarray) -> Tuple[List[DetectedBall], np.ndarray]:
        ball_mask, accepted_px = self._ball_mask(hsv)

        contours, _ = cv2.findContours(
            ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        balls: List[DetectedBall] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1e-6:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < self._circularity_min:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (self._min_radius_px <= radius <= self._max_radius_px):
                continue
            # Pinhole distance: d = (D_real_cm * fx) / (2 * r_px)
            dist_cm = (self._ball_diam_mm / 10.0 * self._fx) / (2.0 * radius)
            balls.append(DetectedBall(
                center=(int(cx), int(cy)),
                radius_px=radius,
                distance_cm=dist_cm,
            ))

        return balls, accepted_px

    # ── Obstacle detection ────────────────────────────────────────────────

    def _obstacle_mask(self, hsv: np.ndarray, ball_px_mask: np.ndarray) -> np.ndarray:
        """
        Build the obstacle binary mask.

        Logic (from working reference):
          obstacle = (V in [min_val, max_val])
                     AND (S >= min_saturation)   ← excludes white/grey floor
                     AND NOT ball_px_mask
          Top top_ignore_ratio% of frame zeroed out.
        """
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]

        mask = np.zeros(val.shape, dtype=np.uint8)
        mask[(val >= self._obs_min_val) & (val <= self._obs_max_val)] = 255
        mask[sat < self._obs_min_sat]  = 0   # removes white/grey floor
        mask[ball_px_mask > 0]         = 0   # removes ball pixels

        # Blank out sky / ceiling strip
        top_px = int(hsv.shape[0] * self._top_ignore)
        mask[:top_px, :] = 0

        return self._clean_mask(mask)

    def _detect_obstacles(
        self, hsv: np.ndarray, ball_px_mask: np.ndarray
    ) -> List[DetectedObstacle]:
        obs_mask = self._obstacle_mask(hsv, ball_px_mask)

        contours, _ = cv2.findContours(
            obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles: List[DetectedObstacle] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._obs_min_area or area > self._obs_max_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            obstacles.append(DetectedObstacle(center=(cx, cy), contour=cnt, area_px=area))

        return obstacles

    # ── APF vector computation ────────────────────────────────────────────

    def _compute_apf_vectors(
        self,
        balls: List[DetectedBall],
        obstacles: List[DetectedObstacle],
    ) -> List[MovementVector]:
        """
        Compute one APF vector per ball.

        Attractive force: constant unit vector toward ball (avoids local minima).
        Repulsive force:  Khatib 1986 — obstacles within rep_influence_cm push back.

        Angle convention (matches working reference):
            0deg   = straight ahead
            +90deg = right
            -90deg = left
        """
        rx, ry_img = self._robot_px
        ry = -ry_img   # flip to right-handed (Y up) frame

        vectors: List[MovementVector] = []

        for ball in balls:
            bx, by_img = ball.center
            by = -by_img

            # Attractive force
            att      = np.array([bx - rx, by - ry], dtype=float)
            att_dist = np.linalg.norm(att)
            if att_dist < 1e-6:
                continue
            f_att = self._k_att * att / att_dist

            # Repulsive forces from obstacles
            rho0_px = (self._rep_influence_cm * self._fx) / max(ball.distance_cm, 1.0)
            f_rep   = np.zeros(2, dtype=float)

            for obs in obstacles:
                ox, oy_img = obs.center
                oy   = -oy_img
                away = np.array([rx - ox, ry - oy], dtype=float)
                rho  = np.linalg.norm(away)
                if rho < 1e-6 or rho >= rho0_px:
                    continue
                unit_away = away / rho
                f_rep += (
                    self._k_rep
                    * (1.0 / rho - 1.0 / rho0_px)
                    * (1.0 / rho ** 2)
                    * unit_away
                )

            net      = f_att + f_rep
            net_norm = np.linalg.norm(net)
            if net_norm < 1e-6:
                continue

            angle_rad = math.atan2(float(net[0]), float(net[1]))
            angle_deg = math.degrees(angle_rad)
            if angle_deg <= -180.0:
                angle_deg = 180.0

            raw_mag   = net_norm / max(self._k_att, 1e-6)
            magnitude = float(np.clip(raw_mag, self._min_magnitude, 1.0))

            vectors.append(MovementVector(
                angle=round(angle_deg, 2),
                magnitude=round(magnitude, 4),
                distance_cm=round(ball.distance_cm, 2),
                target_px=ball.center,
            ))

        return vectors
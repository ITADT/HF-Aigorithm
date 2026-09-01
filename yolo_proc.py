"""yolo_proc - 共享 YOLO 检测 worker + 真假目标三态接口（agent 侧自研识别器）。

设计（spec 029 sensor 四态返回的落地）：
  * 3 架 UAV 的 agent 实例共用一个后台 worker（模块级单例），各自把
    obs.self.photo 推进内存队列，worker 推理后按 uid 存 latest；
    sensor() 只做内存操作（queue.put），满足隔离不变量 I-1。
  * 三态真假接口：YoloHypothesis.is_real = True(真) / False(诱饵) / None(未知)。
    按用户约定：官方单类权重（只含 TargetVehicle，未做真假区分）检出即视为
    真目标候选 -> is_real=True；换 2 类权重后按 names 动态判别
    TargetVehicle=True / DecoyVehicle=False；未知类名 -> None。FSM 只管读。
  * 降级链：photo 为 None / worker 未就绪 / 结果过期 -> agent.sensor() 返回
    None -> 走 SDK 默认识别器（train=AccuracySimulator，eval=YoloDetector），
    现有行为不回归。

几何：bbox 中心 -> pan/tilt 增量（本地纯函数，与 yolotrack 同口径）；
pan/tilt -> lat/lon 优先复用 SDK 的 pan_tilt_to_latlon，import 失败时用
内置同口径实现兜底（离线单测不依赖 SDK）。
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ultralytics 配置目录（默认 %APPDATA%\Ultralytics 常被权限挡 -> eval 崩）
_DEFAULT_CONFIG_DIR = r"D:\HF\.ultralytics"
if not os.environ.get("YOLO_CONFIG_DIR"):
    try:
        os.makedirs(_DEFAULT_CONFIG_DIR, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = _DEFAULT_CONFIG_DIR
    except Exception:
        pass

# 模型路径/推理参数可用环境变量覆盖（本地运行命令 / 平台 env 注入）
DEFAULT_MODEL_PATH = os.environ.get(
    "YOLO_MODEL",
    r"E:\osreadm\hf2026-sim-windows\examples\yolotrack\target_vehicle_yolov8s.pt")
DEFAULT_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
DEFAULT_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
DEFAULT_FOV = 50.0                      # 兜底 hfov（无 obs 时）；正常由 sensor 传真实 gimbal_fov
_MAX_AGE_MS = 1000                      # 结果新鲜度上限（sensor 判定用）
_IOU_MATCH = 0.25                       # 跨帧 IoU 关联阈值（mini 跟踪器）
_QUEUE_MAX = 12                         # 帧队列上限（latest-wins，丢旧帧）

_M_PER_DEG_LAT = 111320.0


@dataclass
class YoloHypothesis:
    """一个检测假设（三态真假接口的核心数据结构）。"""
    track_id: int                       # 跨帧跟踪 ID（per-UAV mini IoU 关联）
    conf: float                         # YOLO 置信度
    class_name: str                     # "TargetVehicle" / "DecoyVehicle" / ...
    is_real: Optional[bool]             # True=真 / False=诱饵 / None=未知（单类权重）
    bbox_center: Tuple[float, float]    # (cx, cy) 像素
    pan_delta: float                    # bbox 中心相对图像中心的 pan 增量（度）
    tilt_delta: float                   # 同上 tilt 增量（度）
    image_size: Tuple[int, int]         # (W, H)
    sim_seq: int = 0                    # 帧序号（调试用）


def bbox_to_pan_tilt_delta(bbox_center: Tuple[float, float],
                           image_size: Tuple[int, int],
                           hfov_deg: float, vfov_deg: float
                           ) -> Tuple[float, float]:
    """bbox 中心相对图像中心的 pan/tilt 增量（度），与 yolotrack 同口径。"""
    cx, cy = bbox_center
    W, H = image_size
    if W <= 0 or H <= 0:
        return 0.0, 0.0
    dx_norm = (cx - W / 2.0) / (W / 2.0)
    dy_norm = (cy - H / 2.0) / (H / 2.0)
    pan_delta = dx_norm * (hfov_deg / 2.0)
    tilt_delta = dy_norm * (vfov_deg / 2.0)
    return pan_delta, tilt_delta


def _meters_to_deg(meters: float, lat: float, is_lon: bool) -> float:
    if is_lon:
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        return meters / (_M_PER_DEG_LAT * cos_lat)
    return meters / _M_PER_DEG_LAT


def _pan_tilt_to_latlon_local(uav_lat: float, uav_lon: float, uav_alt: float,
                              gimbal_pan: float, gimbal_tilt: float,
                              pan_delta: float, tilt_delta: float
                              ) -> Tuple[float, float]:
    """SDK pan_tilt_to_latlon 的同口径内置实现（import 失败时兜底）。"""
    total_tilt = gimbal_tilt + tilt_delta
    total_pan = gimbal_pan + pan_delta
    abs_tilt = max(min(abs(total_tilt), 90.0), 1e-3)
    if 90.0 - abs_tilt <= 0:
        return uav_lat, uav_lon
    ground_dist = uav_alt / math.tan(math.radians(abs_tilt))
    pan_rad = math.radians(total_pan)
    d_north = ground_dist * math.cos(pan_rad)
    d_east = ground_dist * math.sin(pan_rad)
    d_lat = _meters_to_deg(d_north, uav_lat, is_lon=False)
    d_lon = _meters_to_deg(d_east, uav_lat, is_lon=True)
    return uav_lat + d_lat, uav_lon + d_lon


try:
    from competition.sdk.core.perception.bbox_to_latlon import \
        pan_tilt_to_latlon as _pan_tilt_to_latlon_sdk
except Exception:                       # 离线单测 / 独立环境
    _pan_tilt_to_latlon_sdk = None


def pan_tilt_to_latlon(uav_lat: float, uav_lon: float, uav_alt: float,
                       gimbal_pan: float, gimbal_tilt: float,
                       pan_delta: float, tilt_delta: float
                       ) -> Tuple[float, float]:
    if _pan_tilt_to_latlon_sdk is not None:
        return _pan_tilt_to_latlon_sdk(
            uav_lat, uav_lon, uav_alt, gimbal_pan, gimbal_tilt,
            pan_delta, tilt_delta)
    return _pan_tilt_to_latlon_local(
        uav_lat, uav_lon, uav_alt, gimbal_pan, gimbal_tilt,
        pan_delta, tilt_delta)


def project(h: YoloHypothesis, uav_lat: float, uav_lon: float, uav_alt: float,
            gimbal_pan: float, gimbal_tilt: float) -> Tuple[float, float]:
    """把检测假设投影为地面 lat/lon（用当前本机位姿，抵消推理延迟）。"""
    return pan_tilt_to_latlon(uav_lat, uav_lon, uav_alt,
                              gimbal_pan, gimbal_tilt,
                              h.pan_delta, h.tilt_delta)


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0.0 else 0.0


class YoloProc:
    """共享 YOLO 检测 worker：单模型实例 + 每 UAV 独立 latest/跟踪状态。

    线程模型：push() 可从任意 agent 线程调用（sensor 10Hz）；唯一 worker
    线程串行推理（避免 ultralytics 多线程并发同一模型），每 UAV 只保留
    最新结果（latest-wins，队列满丢最旧帧）。
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 imgsz: int = DEFAULT_IMGSZ, conf: float = DEFAULT_CONF):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self._q: "queue.Queue[Tuple[str, int, bytes, float]]" = \
            queue.Queue(maxsize=_QUEUE_MAX)
        self._lock = threading.Lock()
        self._latest: Dict[str, Tuple[int, List[YoloHypothesis], float]] = {}
        self._push_seq: Dict[str, int] = {}
        self._prev: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        self._next_track: Dict[str, int] = {}
        self._model = None
        self._names: Dict[int, str] = {}
        self._healthy = False
        self._setup_done = False
        self._setup_err: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- 生命周期 --------------------------------------------------------

    def ensure_started(self) -> None:
        """幂等启动 worker 线程（模型在 worker 内延迟加载，不阻塞 agent）。"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="yolo-proc", daemon=True)
            self._thread.start()

    def warmup(self) -> Optional[str]:
        """同步加载模型（离线单测 / 预加载用）。返回错误信息或 None。"""
        if self._setup_done:
            return self._setup_err
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._names = dict(self._model.names)
            self._setup_done = True
            self._setup_err = None
            with self._lock:
                self._healthy = True
            return None
        except Exception as e:           # noqa: BLE001
            self._setup_done = True
            self._setup_err = f"model load failed: {e}"
            self._healthy = False
            return self._setup_err

    def stop(self) -> None:
        self._stop.set()

    def healthy(self) -> bool:
        return self._healthy

    def setup_err(self) -> Optional[str]:
        return self._setup_err

    def model_names(self) -> Dict[int, str]:
        return dict(self._names)

    # -- 入口（agent.sensor 调用，纯内存操作）----------------------------

    def push(self, uid: str, photo: bytes, fov_deg: float) -> None:
        """把本 UAV 最新相机帧入队（sensor 内调用，不阻塞）。"""
        if not self._healthy:
            return
        with self._lock:
            seq = self._push_seq.get(uid, 0) + 1
            self._push_seq[uid] = seq
        try:
            self._q.put_nowait((uid, seq, photo, fov_deg))
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((uid, seq, photo, fov_deg))
            except queue.Full:
                pass

    def latest(self, uid: str, max_age_ms: int = _MAX_AGE_MS
               ) -> Optional[List[YoloHypothesis]]:
        """取该 UAV 最新检测假设（过期返回 None -> 调用方走降级链）。"""
        with self._lock:
            item = self._latest.get(uid)
        if item is None:
            return None
        _, hyps, wall_ms = item
        if time.time() * 1000.0 - wall_ms > max_age_ms:
            return None
        return hyps

    def reset(self, uid: str) -> None:
        """每局开始清空该 UAV 的跟踪状态（防跨局串扰）。"""
        with self._lock:
            self._latest.pop(uid, None)
            self._prev.pop(uid, None)
            self._next_track.pop(uid, None)

    # -- worker 线程 ------------------------------------------------------

    def _run(self) -> None:
        err = self.warmup()
        if err is not None:
            import sys
            print(f"[yolo_proc] {err} -> 使用默认识别器降级", file=sys.stderr)
            return
        print(f"[yolo_proc] 已启动: model={self.model_path} "
              f"imgsz={self.imgsz} conf={self.conf} names={self._names}")
        while not self._stop.is_set():
            try:
                uid, seq, photo, fov = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                hyps = self._infer(uid, photo, fov, seq)
            except Exception as e:       # noqa: BLE001
                print(f"[yolo_proc] infer error: {e}")
                continue
            if hyps is None:
                continue
            with self._lock:
                cur = self._latest.get(uid)
                if cur is not None and cur[0] >= seq:
                    continue            # 已有更新结果
                self._latest[uid] = (seq, hyps, time.time() * 1000.0)

    def _infer(self, uid: str, photo: bytes, fov_deg: float, seq: int
               ) -> Optional[List[YoloHypothesis]]:
        import cv2
        import numpy as np
        arr = np.frombuffer(photo, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        results = self._model(
            img, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results or results[0].boxes is None:
            return None
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        hfov = fov_deg if fov_deg and fov_deg > 0 else DEFAULT_FOV
        vfov = hfov * (H / W) if W > 0 else hfov
        new_xyxy = []
        new_hyps = []
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy().tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            pd, td = bbox_to_pan_tilt_delta((cx, cy), (W, H), hfov, vfov)
            name = self._names.get(cls_id, str(cls_id))
            new_xyxy.append((x1, y1, x2, y2))
            new_hyps.append(YoloHypothesis(
                track_id=-1, conf=conf, class_name=name,
                is_real=self._classify(cls_id, name),
                bbox_center=(cx, cy), pan_delta=pd, tilt_delta=td,
                image_size=(W, H), sim_seq=seq))
        # per-UAV mini IoU 跟踪关联（保 track_id 连续性，供 FSM 未来使用）
        with self._lock:
            prev = self._prev.get(uid, [])
            nxt = self._next_track.get(uid, 1)
        ids = self._associate(prev, new_xyxy, nxt)
        with self._lock:
            self._prev[uid] = list(zip(ids, new_xyxy))
            self._next_track[uid] = max(ids) + 1 if ids else nxt
        for h, tid in zip(new_hyps, ids):
            h.track_id = tid
        new_hyps.sort(key=lambda h: h.conf, reverse=True)
        return new_hyps

    def _classify(self, cls_id: int, class_name: str) -> Optional[bool]:
        """真假判定（用户约定）：官方单类权重 -> 恒 True；2 类权重按类名判别。

        官方权重只在 TargetVehicle 一个类上训练（未做真假区分），所以
        "检出即视为真目标候选"（is_real=True）。后续换 2 类权重
        （TargetVehicle/DecoyVehicle）后自动按类名返回 True/False；
        未知类名 -> None（交给 FSM 保守处理）。
        """
        n_cls = len(self._names)
        if n_cls >= 2:
            low = (class_name or "").lower()
            if low in ("targetvehicle", "target_vehicle", "target"):
                return True
            if low in ("decoyvehicle", "decoy_vehicle", "decoy"):
                return False
            return None
        return True

    @staticmethod
    def _associate(prev: List[Tuple[int, Tuple[float, float, float, float]]],
                   new: List[Tuple[float, float, float, float]],
                   nxt: int) -> List[int]:
        """贪心 IoU 关联：命中续 id，未命中发新 id。"""
        ids = []
        used_prev = set()
        for box in new:
            best_tid, best_iou, best_pi = -1, _IOU_MATCH, -1
            for pi, (tid, pbox) in enumerate(prev):
                if pi in used_prev:
                    continue
                iou = _iou(pbox, box)
                if iou > best_iou:
                    best_iou, best_tid, best_pi = iou, tid, pi
            if best_pi >= 0:
                used_prev.add(best_pi)
                ids.append(best_tid)
            else:
                ids.append(nxt)
                nxt += 1
        return ids


# 模块级共享单例：3 架 UAV 的 agent 实例共用
proc = YoloProc()
_YOLO_MAX_AGE_MS = _MAX_AGE_MS

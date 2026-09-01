"""
V4.2 协同收敛版
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

from competition.sdk.core.commands import (
    Command, broadcast, fly_to, point_gimbal, report_target, set_gimbal_fov,
)
from competition.sdk.scenarios.coop_decoy import CoopAgent
from competition.sdk.scenarios.coop_decoy.observation import CoopObs
from competition.sdk.core.observation import Detection

try:
    from . import yolo_proc              # 平台/包内：同一目录或同包
except Exception:
        yolo_proc = None                 # 都没有 -> YOLO 关闭（走默认识别器）

# ── 任务几何常量 ─────────────────────────────────────────────────────────
_M_PER_DEG = 111320.0                      # 纬度 1° ≈ 111.32 km
_ALT = 500.0                               # 巡航高度（m，引擎锁 500 不可改）
_SEARCH_SPD = 25.0                         # 搜索巡航速度（UAV min 15 / max 40）
_TRACK_SPD = 15.0                          # 盯防速度（低速保证 gimbal 跟上绕飞）
_FOV_SEARCH = 50.0                         # 搜索大视场（引擎上限 50°）
_FOV_TRACK = 50.0                          # 盯防视场（窄一点，锁定更稳）
# 巡逻分区（mission_area，lat 26.98–27.02 / lon 124.98–125.02）
_BBOX = ((26.98, 124.98), (27.02, 125.02))
# 安全框 = terrain bbox 内缩，避免 >500m 越界罚分
_SAFE_BBOX = ((26.986, 124.984), (27.021, 125.016))
_SEARCH_RADIUS = 600.0                     # 绕飞半径（m）
_SEARCH_ANG_SPD = 2.5                      # 绕飞角速度（°/s，线速度≈24m/s）
_SECTOR_SWAP_S = 75.0                      # 每 75s 轮换下一个分区中心
# 盯防站位
_HELP_OFFSET_M = 450.0                     # 辅助机与目标水平偏移（>300 保证与主机 >200m）
_MAIN_SWING_M = 200.0                      # 主机摆锤半幅：目标侧向 200m 切弦往返（不穿目标）
_MAIN_TANGENT_M = 80.0                     # 切弦两端点沿切向的间距半宽
_HELP_LOITER = 100.0                       # 辅机绕飞半径
_HELP_SPEED = 35.0                         # 支援机转场速度（m/s，越快越早形成双机锁定）
# 状态机时间参数
_VERIFY_MIN_S = 3.0                        # 确认最短时长
_VERIFY_SAMPLES = 3                        # 确认最少样本数
_LATE_VERIFY_S = 6.0                       # 晚窗（t>32s）候选确认时长（速度已无判别力）
_BC_INTERVAL_S = 1.0                       # MAIN 广播目标周期（越快收敛 K=2 越早）
_LOST_TIMEOUT_S = 3.0                      # 连续无检出视为丢失
_REACQ_DET_S = 6.0                         # ENGAGE 连续无检出多久进入 REACQ 重捕
_REACQ_RADIUS_M = 500.0                    # REACQ 绕飞半径（m，防多机同点绕飞<200m）
_ENGAGE_TIMEOUT_S = 60.0                   # 盯防超时（真目标需 ≥2 机 20s dwell + 支援到达）
_TIMEOUT_BASE_S = 22.0                     # 无双机锁定的盯防超时（诱饵快速释放）
_VERIFY_OFFSET_M = 520.0                   # VERIFY 接近点偏移（防多机同点 <200m 罚分）
_MATCH_M = 250.0                           # 检测点与候选的关联半径（m）
_BLACKLIST_M = 200.0                       # 黑名单跳过半径（m）
_TRACK_M = 400.0                           # 盯防期检测点与候选的关联半径（m）
_HELP_MATCH_M = 150.0                      # HELP 期检测点与共享目标的紧关联半径（防被邻车带偏）
_PAN_RATE_LIMIT = 60.0                     # gimbal pan 角速度上限（°/s，引擎限速）
_TILT_RATE_LIMIT = 30.0                    # gimbal tilt 限速（°/s，保守值）
# 诱饵快速判别（真目标 Start WaitTime=30s 静止起步；诱饵 WaitTime=0 立即移动）
_VERIFY_BC_S = 3.0                         # VERIFY 确认后立即广播求援（支援提前转场）
_FAST_TRIAL_S = 14.0                       # 快速判别观察窗口（秒）
_FAST_TRIAL_WINDOW_T = 32.0                # 仅 t<32s 时“运动=诱饵”成立（真目标 30s 后才动）
_DECOY_SPEED = 6.0                         # 速度估计 > 此值 → 诱饵（m/s，放宽防误杀静止真目标）
_REAL_SPEED = 3.5                          # 速度估计 < 此值 → 真目标（m/s）
_ABANDON_M = 600.0                         # HELP 收到 D: 时放弃候选的关联半径
_BLACKLIST_TTL = 40.0                      # 黑名单有效期（防误判真目标被永久拉黑）


def _hav(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 水平距离（米）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def _bear(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """绝对方位角（0=N，顺时针）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _offset_pt(lat: float, lon: float, dist_m: float, az_deg: float
               ) -> Tuple[float, float]:
    """从 (lat,lon) 沿方位 az 平移 dist_m 米。"""
    dlat = dist_m * math.cos(math.radians(az_deg)) / _M_PER_DEG
    dlon = (dist_m * math.sin(math.radians(az_deg))
            / (_M_PER_DEG * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def _clamp(lat: float, lon: float) -> Tuple[float, float]:
    (lat_min, lon_min), (lat_max, lon_max) = _SAFE_BBOX
    return min(max(lat, lat_min), lat_max), min(max(lon, lon_min), lon_max)


def _parse_ll(payload: str) -> Optional[Tuple[float, float]]:
    try:
        la, lo = payload.split(",")
        return float(la), float(lo)
    except Exception:
        return None


def _parse_msg(p: str) -> Optional[Tuple[str, Tuple[float, float]]]:
    """解析 T:<uid>:<lat>,<lon> / D:<uid>:<lat>,<lon> → (uid, (lat,lon))。"""
    try:
        kind, rest = p[0], p[2:]
        uid, ll = rest.split(":", 1)
        la, lo = ll.split(",")
        return uid, (float(la), float(lo))
    except Exception:
        return None


class _EMA:
    """位置 EMA 平滑 + 匀速外推。

    检测位置噪声 σ≈75m（Snow_Light 天气），单帧不可用；EMA 平滑后上报，
    外推用于盯防时预判目标移动（目标速度 ~10m/s）。
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.vlat = 0.0
        self.vlon = 0.0
        self.last_dt = 0.1

    def reset(self) -> None:
        self.lat = None
        self.lon = None
        self.vlat = 0.0
        self.vlon = 0.0

    def seed(self, lat: float, lon: float) -> None:
        self.lat, self.lon = lat, lon

    def update(self, lat: float, lon: float, dt: float) -> None:
        if self.lat is None:
            self.lat, self.lon = lat, lon
            self.last_dt = max(dt, 0.05)
            return
        dt = max(dt, 0.05)
        # 速度估计（跨帧差分，噪声大但可用）
        self.vlat = (1 - self.alpha) * self.vlat + \
            self.alpha * (lat - self.lat) / dt
        self.vlon = (1 - self.alpha) * self.vlon + \
            self.alpha * (lon - self.lon) / dt
        self.lat = (1 - self.alpha) * self.lat + self.alpha * lat
        self.lon = (1 - self.alpha) * self.lon + self.alpha * lon
        self.last_dt = max(dt, 0.05)

    def predict(self, ahead: float = 0.0) -> Tuple[float, float]:
        if self.lat is None:
            return (0.0, 0.0)
        return (self.lat + self.vlat * ahead,
                self.lon + self.vlon * ahead)


class _SpeedEst:
    """检测位置最小二乘速度估计（10Hz 原始点）。

    σ≈50-75m 噪声下 8-10s 窗口（80-100 点）可将 5-10m/s 的真实运动与
    静止目标区分开（静止时估计值噪声 σ≈2-4m/s）。
    """

    def __init__(self, history: int = 150):
        self._buf: Deque[Tuple[float, float, float]] = deque(maxlen=history)

    def reset(self) -> None:
        self._buf.clear()

    def n(self) -> int:
        return len(self._buf)

    def append(self, lat: float, lon: float, t: float) -> None:
        self._buf.append((t, lat, lon))

    def speed_mps(self) -> Optional[float]:
        n = len(self._buf)
        if n < 15:
            return None
        ts = [p[0] for p in self._buf]
        lats = [p[1] for p in self._buf]
        lons = [p[2] for p in self._buf]

        def _slope(xs: List[float], ys: List[float]) -> float:
            nn = float(len(xs))
            sx = sum(xs)
            sy = sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            d = nn * sxx - sx * sx
            if abs(d) < 1e-12:
                return 0.0
            return (nn * sxy - sx * sy) / d

        vlat = _slope(ts, lats) * _M_PER_DEG
        vlon = _slope(ts, lons) * _M_PER_DEG * math.cos(math.radians(lats[-1]))
        return math.hypot(vlat, vlon)


class V1CoopAgent(CoopAgent):
    """V1 简单可行版：分区绕飞搜索 + 识别后双机协同盯防"""

    def _log(self, msg: str) -> None:
        print(f"[{self.my_uid}] {msg}", flush=True)

    # 状态
    SEARCH = 0
    VERIFY = 1
    ENGAGE_MAIN = 2
    ENGAGE_HELP = 3

    def __init__(self, my_uid: str):
        super().__init__(my_uid)
        self._t = 0.0
        self._tick = 0
        self._state = self.SEARCH
        # 分区中心列表（3 个，按 lon 三等分）
        self._centers = self._partition_centers()
        self._region_idx = self._uid_slot(my_uid)
        self._phase = (int(my_uid) % 10) * 2.0 if my_uid.isdigit() else 0.0
        # 搜索状态
        self._swap_t = 0.0
        # 目标处理状态
        self._cand: Optional[Tuple[float, float]] = None
        self._ema = _EMA()
        self._verify_t = 0.0
        self._samples = 0
        self._last_det_t = -1e9
        self._engage_t = 0.0
        self._nd_on_engage = 0          # 进入盯防时的 n_destroyed
        self._last_report_t = -1e9
        self._last_bc_t = -1e9
        self._help_az = self._help_azimuth(my_uid)
        self._blacklist: List[Tuple[float, float, float]] = []
        self._n_destroyed = 0
        self._det_win = 0
        self._win_n = 0
        self._last_det: Optional[Tuple[float, float]] = None
        self._gimbal_pan: Optional[float] = None   # 最近下发的云台指令（限幅基准）
        self._gimbal_tilt: Optional[float] = None
        self._pendulum_side = 1                    # 摆锤当前侧（+1/-1）
        self._seen_msgs: Deque[Tuple[str, str]] = deque(maxlen=256)
        self._helper_lock_t: Dict[str, float] = {}  # uid → 首次 L: 锁定时刻
        self._helper_ack_t: Dict[str, float] = {}   # uid → A: 支援应答时刻
        self._last_l_t = -1e9                        # 本机最近 L: 广播时刻
        self._speed = _SpeedEst()            # 诱饵速度判别（运动=诱饵）
        self._verify_start_t = 0.0           # 候选首次观察时刻（任务时间）
        self._cand_real = False              # 候选已确认真目标（静止）
        self._help_from: Optional[str] = None  # HELP 跟随的 MAIN（uid，用于 D: 匹配）
        self._reacq = False                  # ENGAGE 掉锁重捕模式
        self._reacq_pt: Optional[Tuple[float, float]] = None  # REACQ 绕飞中心
        self._pend_center: Optional[Tuple[float, float]] = None  # 摆锤冻结中心
        self._pend_refresh_t = -1e9
        self._help_station: Optional[Tuple[float, float]] = None  # 支援站位冻结中心
        self._help_refresh_t = -1e9
        # ── YOLO 自研识别器（sensor 填充；官方单类权重 -> is_real=None）──
        self._hypotheses: List["yolo_proc.YoloHypothesis"] = []
        if yolo_proc is not None:
            yolo_proc.proc.ensure_started()

    # ── 初始化辅助 ─────────────────────────────────────────────────────

    @staticmethod
    def _partition_centers() -> List[Tuple[float, float]]:
        (lat_min, lon_min), (lat_max, lon_max) = _BBOX
        lat_mid = (lat_min + lat_max) / 2.0
        n = 3
        sub_w = (lon_max - lon_min) / n
        return [(lat_mid, lon_min + sub_w * (i + 0.5)) for i in range(n)]

    @staticmethod
    def _uid_slot(uid: str) -> int:
        if uid.isdigit():
            return int(uid) % 3
        return abs(hash(uid)) % 3

    @staticmethod
    def _help_azimuth(uid: str) -> float:
        """辅助机站位方位（按分区槽位 0/1/2 → 90/210/330°，互不重叠）。"""
        slot = V1CoopAgent._uid_slot(uid)
        return 90.0 + slot * 120.0

    def reset(self) -> None:
        self._t = 0.0
        self._tick = 0
        self._state = self.SEARCH
        self._region_idx = self._uid_slot(self.my_uid)
        self._swap_t = 0.0
        self._cand = None
        self._ema.reset()
        self._verify_t = 0.0
        self._samples = 0
        self._last_det_t = -1e9
        self._engage_t = 0.0
        self._nd_on_engage = 0
        self._last_report_t = -1e9
        self._last_bc_t = -1e9
        self._blacklist = []
        self._n_destroyed = 0
        self._gimbal_pan = None
        self._gimbal_tilt = None
        self._pendulum_side = 1
        self._seen_msgs.clear()
        self._helper_lock_t = {}
        self._helper_ack_t = {}
        self._last_l_t = -1e9
        self._speed.reset()
        self._verify_start_t = 0.0
        self._cand_real = False
        self._reacq = False
        self._reacq_pt = None
        self._pend_center = None
        self._pend_refresh_t = -1e9
        self._help_station = None
        self._help_refresh_t = -1e9
        self._hypotheses = []
        if yolo_proc is not None:
            yolo_proc.proc.reset(self.my_uid)

    # ── 工具 ───────────────────────────────────────────────────────────

    def _near_blacklist(self, lat: float, lon: float) -> bool:
        return any(self._t - b[2] < _BLACKLIST_TTL
                   and _hav(lat, lon, b[0], b[1]) < _BLACKLIST_M
                   for b in self._blacklist)

    def _add_blacklist(self, lat: float, lon: float) -> None:
        self._blacklist.append((lat, lon, self._t))
        if len(self._blacklist) > 24:
            self._blacklist = self._blacklist[-24:]

    def _gimbal_to(self, uav_lat: float, uav_lon: float, uav_heading: float,
                   tlat: float, tlon: float, uav_alt: float = _ALT
                   ) -> Tuple[float, float]:
        """云台对准目标：pan 相对机头，tilt 按斜下几何（用真实高度）。"""
        brg = _bear(uav_lat, uav_lon, tlat, tlon)
        pan = _norm180(brg - uav_heading)
        ground = max(1.0, _hav(uav_lat, uav_lon, tlat, tlon))
        tilt = -math.degrees(math.atan2(max(uav_alt, 50.0), ground))
        return pan, tilt

    def _smooth_gimbal(self, pan: float, tilt: float, dt: float
                       ) -> Tuple[float, float]:
        """云台指令角速度限幅（v2 关键修复）。

        引擎对 gimbal pan 有 60°/s 限速；ENGAGE 阶段若每帧下发大步进
        （EMA 外推位置噪声 + 绕飞机动），实际光轴会追不上目标、目标脱出
        FOV 锥 → 引擎检测不到 → dwell 不累积 → kill=0。
        以"上次已下发指令"为基准做限幅，保证指令流始终可被执行。
        """
        if self._gimbal_pan is None:
            self._gimbal_pan = float(pan)
            self._gimbal_tilt = float(tilt)
            return self._gimbal_pan, self._gimbal_tilt
        dpan = _norm180(pan - self._gimbal_pan)
        dpan = max(-_PAN_RATE_LIMIT * dt, min(_PAN_RATE_LIMIT * dt, dpan))
        dtilt = tilt - self._gimbal_tilt
        dtilt = max(-_TILT_RATE_LIMIT * dt, min(_TILT_RATE_LIMIT * dt, dtilt))
        self._gimbal_pan = _norm180(self._gimbal_pan + dpan)
        self._gimbal_tilt += dtilt
        return self._gimbal_pan, self._gimbal_tilt

    def _track_pt(self) -> Tuple[float, float]:
        """盯防基准点：慢 EMA 位置（不外推速度）。

        外推 ahead=0.5s 会把 σ75m 噪声放大成 ~300m 的每帧跳变；这里只用
        滤波位置，目标静止/慢速（真目标前 30s 静止）时收敛到真值。
        """
        return _clamp(*self._ema.predict(0.0))

    # ── 路线规划：搜索巡逻 ─────────────────────────────────────────────

    def _search_route(self) -> Tuple[float, float, float, float]:
        """当前时刻的搜索目标点 + 云台扫描姿态。

        以分区中心为圆心绕飞（半径 550m，角速度 2.5°/s），每 _SECTOR_SWAP_S
        轮换到下一个分区中心；云台左右摆动扫描（pan ±45° 正弦），俯角固定
        看斜下方（tilt -55°，光轴落点 ~245m，覆盖半径 ~163m）。
        """
        self._swap_t += self._dt
        if self._swap_t >= _SECTOR_SWAP_S:
            self._swap_t = 0.0
            self._region_idx = (self._region_idx + 1) % len(self._centers)
        c_lat, c_lon = self._centers[self._region_idx]
        t = self._t + self._phase
        bearing = (self._search_ang_speed() * t) % 360.0
        revs = (self._search_ang_speed() * t) / 360.0
        # spiral: radius 200m -> 600m, ring gap < 2x FOV coverage
        radius = min(_SEARCH_RADIUS, 200.0 + 200.0 * revs)
        # 绕飞点（相移保证三机不重叠）
        lat = c_lat + (radius * math.cos(math.radians(bearing))) / _M_PER_DEG
        lon = c_lon + (_SEARCH_RADIUS * math.sin(math.radians(bearing))) / \
            (_M_PER_DEG * math.cos(math.radians(lat)))
        lat, lon = _clamp(lat, lon)
        # 云台摆动扫描
        pan = 45.0 * math.sin(2.0 * math.pi * t / 10.0)
        tilt = -72.5 - 17.5 * math.cos(2.0 * math.pi * t / 6.0)
        return lat, lon, pan, tilt

    def _search_ang_speed(self) -> float:
        # 线速度 = r × ω ≈ 24 m/s，贴近巡航速度避免追点失真
        return _SEARCH_ANG_SPD

    def _search_cmds(self) -> List[Command]:
        lat, lon, pan, tilt = self._search_route()
        return [fly_to(lat, lon, alt=_ALT, speed=_SEARCH_SPD),
                point_gimbal(pan, tilt),
                set_gimbal_fov(_FOV_SEARCH)]

    # ── 识别后的处理 ───────────────────────────────────────────────────

    def _verify_cmds(self, obs: CoopObs) -> List[Command]:
        """确认阶段：接近候选 + 云台锁定，收集检测样本。"""
        tlat, tlon = _clamp(*self._cand)
        # 接近点带个人方位偏移（~120m），防止多机同时扑向同一候选而 <200m 罚分
        vlat, vlon = _offset_pt(tlat, tlon, _VERIFY_OFFSET_M, self._help_az)
        vlat, vlon = _clamp(vlat, vlon)
        pan, tilt = self._gimbal_to(
            obs.self.lat, obs.self.lon, obs.self.heading_deg, tlat, tlon,
            obs.self.alt)
        pan, tilt = self._smooth_gimbal(pan, tilt, self._dt)
        return [fly_to(vlat, vlon, alt=_ALT, speed=_TRACK_SPD,
                       loiter_radius=80.0),
                point_gimbal(pan, tilt),
                set_gimbal_fov(_FOV_TRACK)]

    def _engage_main_cmds(self, obs: CoopObs) -> List[Command]:
        """主盯防：目标两侧 ±130m 摆锤往返 + 云台锁定 + 1Hz 上报 + 1s 广播。

        用固定往返点代替动态绕飞中心：绕飞中心每 tick 随 EMA 噪声移动会让
        引擎的 set_destination 持续重规划、UAV 始终追不到点（v4.1 d_cand
        在 144~463m 间漂移）；摆锤只给静止的 A/B 两点，UAV 稳定在目标
        ~130m 内，云台角速度需求 <15°/s，远低于 60°/s 限速。
        """
        cmds: List[Command] = []
        if self._reacq:
            return self._reacq_cmds(obs)
        tlat, tlon = self._track_pt()
        # v4.2g：摆锤中心冻结，5s 刷新一次（实时 EMA 每 tick 移动 → A/B 点
        # 跟着跳 → UAV 追点不落位，dwell 永远累计不起来）
        if (self._pend_center is None
                or self._t - self._pend_refresh_t >= 5.0):
            self._pend_center = (tlat, tlon)
            self._pend_refresh_t = self._t
        # 周期广播目标（用最近检测位置，比滤波位置更接近真值）
        bc = self._last_det if self._last_det is not None else (tlat, tlon)
        if self._t - self._last_bc_t >= _BC_INTERVAL_S:
            cmds.append(broadcast(f"T:{self.my_uid}:{bc[0]:.5f},{bc[1]:.5f}"))
            self._last_bc_t = self._t
        # 1Hz 上报慢 EMA 滤波位置（v4.2d：仅双机锁定期间上报——有 L: 说明
        # HELP 与 MAIN 盯同一目标，上报位置才可能承接真目标；诱饵锁定的孤立
        # 上报污染 accuracy 分桶，v42c1 实测 298 报 → rmse 632）
        if (self._helper_lock_t
                and self._t - self._last_report_t >= 1.0
                and self._t - self._last_det_t <= 3.0):
            cmds.append(report_target(tlat, tlon))
            self._last_report_t = self._t
        # gimbal 指向最近检测位置（2s 内新鲜才用；目标保持在光轴中心 →
        # 引擎持续 detected → 自稳定锁定）
        aim = tlat, tlon
        if self._last_det is not None and self._t - self._last_det_t <= 2.0:
            aim = self._last_det
        pan, tilt = self._gimbal_to(
            obs.self.lat, obs.self.lon, obs.self.heading_deg, aim[0], aim[1],
            obs.self.alt)
        pan, tilt = self._smooth_gimbal(pan, tilt, self._dt)
        ptx, pty = self._pendulum_pt(
            self._pend_center[0], self._pend_center[1],
            obs.self.lat, obs.self.lon)
        cmds.append(fly_to(ptx, pty, alt=_ALT, speed=_TRACK_SPD,
                           loiter_radius=40.0))
        cmds.append(point_gimbal(pan, tilt))
        cmds.append(set_gimbal_fov(_FOV_TRACK))
        return cmds

    def _pendulum_pt(self, tlat: float, tlon: float,
                     uav_lat: float, uav_lon: float
                     ) -> Tuple[float, float]:
        """摆锤切弦往返点：两端点都在目标侧向 _MAIN_SWING_M 处，沿切向相距
        2×_MAIN_TANGENT_M。路径是目标外围一条弧段，不穿过目标正上方——
        v4.2 的对称摆锤会在每次折返时从目标头顶掠过（az 噪声 → 掉锁 →
        coop dwell 重置，实测 10001 有 35s 累计被中断归零 2 次）。
        """
        brg = _bear(uav_lat, uav_lon, tlat, tlon)
        # v4.2e：摆锤方向按 UID 错开（help_az=90/210/330°）——双 MAIN 盯
        # 同一目标时摆锤弧互不重叠（min 间距 ~346m > 200m），消除同弧罚分
        perp = brg + self._help_az
        a = _offset_pt(*_offset_pt(tlat, tlon, _MAIN_SWING_M, perp),
                       _MAIN_TANGENT_M, brg)
        b = _offset_pt(*_offset_pt(tlat, tlon, _MAIN_SWING_M, perp),
                       -_MAIN_TANGENT_M, brg)
        cur = a if self._pendulum_side > 0 else b
        if _hav(uav_lat, uav_lon, cur[0], cur[1]) < 40.0:
            self._pendulum_side = -self._pendulum_side
            cur = a if self._pendulum_side > 0 else b
        return _clamp(*cur)

    def _engage_help_cmds(self, obs: CoopObs) -> List[Command]:
        """辅助盯防：目标侧向 400m 站位绕飞 + 云台锁定。

        站位几何保证与主机（目标上空）最小间距 >300m，规避 <200m 罚分。
        若到达后尚未重新检出目标（<4s），先小半径绕搜锁定。
        v4.2：云台始终瞄准共享目标（_track_pt），不再用自己过期的
        _last_det（v4.1 里 HELP 会被自己的旧检测带偏到别的车）。
        """
        cmds: List[Command] = []
        if self._reacq:
            return self._reacq_cmds(obs)
        tlat, tlon = self._track_pt()
        # v4.2g：站位中心冻结，5s 刷新（T: 每 1s 到、原始位置 σ75m 噪声，
        # 每 tick 重算站位 → 支援机追着跳动的站圈永远落不了位）
        if (self._help_station is None
                or self._t - self._help_refresh_t >= 5.0):
            self._help_station = (tlat, tlon)
            self._help_refresh_t = self._t
        if self._engage_t < 4.0 and self._last_det is None:
            # 到达初期：绕候选点小圈搜索，云台正下扫
            tlat2, tlon2 = _offset_pt(tlat, tlon, 120.0, self._help_az)
            tlat2, tlon2 = _clamp(tlat2, tlon2)
            pan2 = 45.0 * math.sin(2.0 * math.pi * self._t / 6.0)
            return [fly_to(tlat2, tlon2, alt=_ALT, speed=_TRACK_SPD,
                           loiter_radius=90.0),
                    point_gimbal(pan2, -80.0),
                    set_gimbal_fov(_FOV_TRACK)]
        hlat, hlon = _offset_pt(
            self._help_station[0], self._help_station[1],
            _HELP_OFFSET_M, self._help_az)
        hlat, hlon = _clamp(hlat, hlon)
        # v4.2b：已重新检出共享目标（2s 内新鲜）→ 2s 周期广播 L:，让 MAIN
        # 知道双机已就位、延长 dwell 窗口到 20s 完成。
        if (self._t - self._last_det_t <= 2.0
                and self._t - self._last_l_t >= 2.0):
            cmds.append(broadcast(f"L:{self.my_uid}:{self._t:.1f}"))
            self._last_l_t = self._t
        pan, tilt = self._gimbal_to(
            obs.self.lat, obs.self.lon, obs.self.heading_deg, tlat, tlon,
            obs.self.alt)
        pan, tilt = self._smooth_gimbal(pan, tilt, self._dt)
        cmds.append(fly_to(hlat, hlon, alt=_ALT, speed=_HELP_SPEED,
                           loiter_radius=_HELP_LOITER))
        cmds.append(point_gimbal(pan, tilt))
        cmds.append(set_gimbal_fov(_FOV_TRACK))
        return cmds

    def _reacq_cmds(self, obs: CoopObs) -> List[Command]:
        """掉锁重捕：围绕最后已知位置绕飞 + 云台扫描，重检出即续锁。

        v4.2c：LOST 不再弃守/拉黑（out9 里真目标被 D: 黑名单永久忽略），
        改为在目标最后已知位置附近小圈绕飞、云台搜索式摆动重新捕获；
        MAIN 期间仍周期广播 T:，支援机继续向最新位置收敛。
        """
        cmds: List[Command] = []
        tlat, tlon = self._ema.predict(0.0)
        if self._reacq_pt is None:
            self._reacq_pt = (tlat, tlon)
        rlat, rlon = _clamp(*_offset_pt(
            self._reacq_pt[0], self._reacq_pt[1], _REACQ_RADIUS_M,
            self._help_az + 60.0))
        if self._state == self.ENGAGE_MAIN:
            bc = self._last_det if self._last_det is not None else (tlat, tlon)
            if self._t - self._last_bc_t >= _BC_INTERVAL_S:
                cmds.append(broadcast(f"T:{self.my_uid}:{bc[0]:.5f},{bc[1]:.5f}"))
                self._last_bc_t = self._t
        pan = 45.0 * math.sin(2.0 * math.pi * self._t / 10.0)
        tilt = -72.5 - 17.5 * math.cos(2.0 * math.pi * self._t / 6.0)
        cmds.append(fly_to(rlat, rlon, alt=_ALT, speed=_TRACK_SPD,
                           loiter_radius=120.0))
        cmds.append(point_gimbal(pan, tilt))
        cmds.append(set_gimbal_fov(_FOV_TRACK))
        return cmds

    def _handle_near_det(self, det_lat: float, det_lon: float,
                         radius: float = _MATCH_M) -> None:
        """用新检测更新候选 EMA（检测点在候选附近才算同一目标）。"""
        if self._cand is None:
            return
        if _hav(det_lat, det_lon, self._cand[0], self._cand[1]) < radius:
            self._ema.update(det_lat, det_lon, self._dt)
            self._cand = self._ema.predict(0.0)
            self._samples += 1
            self._last_det_t = self._t

    # ── 主决策 ─────────────────────────────────────────────────────────

    # ── 自研识别器（spec 029）：YOLO(photo) -> Detection；无结果 -> None 走默认识别器 ──

    def sensor(self, obs: CoopObs, dt: float) -> Optional[List[Detection]]:
        """YOLO 视觉识别 + 三态真假接口（官方单类权重 -> is_real=None）。

        返回 List[Detection] 表示自研结果（runner 填入 obs.self.detection）；
        返回 None 表示本帧无结果 -> 走默认识别器（train=AccuracySimulator /
        eval=YoloDetector），无 UE / 模型未就绪时不回归。
        """
        if yolo_proc is None:
            return None
        if obs.self.photo is not None:
            yolo_proc.proc.push(self.my_uid, obs.self.photo,
                                obs.self.gimbal_fov_deg)
        hyps = yolo_proc.proc.latest(
            self.my_uid, max_age_ms=yolo_proc._YOLO_MAX_AGE_MS)
        if not hyps:
            self._hypotheses = []
            return None
        self._hypotheses = hyps
        best = self._pick_hypothesis(hyps, obs)
        tlat, tlon = yolo_proc.project(
            best, obs.self.lat, obs.self.lon, obs.self.alt,
            obs.self.gimbal_pan, obs.self.gimbal_tilt)
        return [Detection(detected=True, confidence=best.conf,
                          target_lat=tlat, target_lon=tlon,
                          target_type=best.class_name)]

    def _pick_hypothesis(self, hyps: List["yolo_proc.YoloHypothesis"],
                         obs: CoopObs) -> "yolo_proc.YoloHypothesis":
        """挑一个检测提交给 FSM：盯防/确认期优先候选附近的，否则取最高分。

        官方单类权重阶段：is_real 恒 None，FSM 维持现有 22s 释放 / 双机锁
        上报门控（防止把诱饵当真目标死盯 -> accuracy 分桶被污染）。
        """
        if (self._cand is not None
                and self._state in (self.VERIFY, self.ENGAGE_MAIN,
                                    self.ENGAGE_HELP)):
            near = []
            for h in hyps:
                tlat, tlon = yolo_proc.project(
                    h, obs.self.lat, obs.self.lon, obs.self.alt,
                    obs.self.gimbal_pan, obs.self.gimbal_tilt)
                if _hav(tlat, tlon, self._cand[0], self._cand[1]) < _MATCH_M:
                    near.append(h)
            if near:
                return max(near, key=lambda h: h.conf)
        return max(hyps, key=lambda h: h.conf)

    def decide(self, obs: CoopObs, dt: float) -> List[Command]:
        self._dt = dt
        self._tick += 1
        self._t += dt
        det = obs.self.detection
        sv = obs.briefing.score_view
        n_destroyed = sv.n_destroyed if sv is not None else 0
        self._n_destroyed = max(self._n_destroyed, n_destroyed)
        if self._tick % 10 == 1:
            self._det_win = 0
            self._win_n = 0
        self._win_n += 1
        if det.detected:
            self._det_win += 1
        # ENGAGE 诊断日志（每 2s：距离/云台回读/检出状态）
        if self._state in (self.ENGAGE_MAIN, self.ENGAGE_HELP) and self._tick % 20 == 0 \
                and self._cand is not None:
            d = _hav(obs.self.lat, obs.self.lon, self._cand[0], self._cand[1])
            az2 = det.azimuth_error_deg if det.detected else None
            spe = self._speed.speed_mps()
            sps = f"{spe:.1f}" if spe is not None else "-"
            self._log(f"ENGAGE det={det.detected} az={az2} d_cand={d:.0f} "
                      f"pan={obs.self.gimbal_pan:.0f} tilt={obs.self.gimbal_tilt:.0f} "
                      f"fov={obs.self.gimbal_fov_deg:.0f} alt={obs.self.alt:.0f} spd={obs.self.speed:.0f} "
                      f"spe={sps} real={int(self._cand_real)}")
        if self._tick % 100 == 0:
            az = det.azimuth_error_deg if det.detected else None
            self._log(f"t={self._t:.0f} st={self._state} nd={self._n_destroyed} dw={self._det_win}/{self._win_n} az={az} "
                      f"pan={obs.self.gimbal_pan:.0f} fov={obs.self.gimbal_fov_deg:.0f} hd={obs.self.heading_deg:.0f} "
                      f"yolo={len(self._hypotheses)} "
                      f"pos=({obs.self.lat:.5f},{obs.self.lon:.5f})")
        if self._tick % 10 == 0:
            self._log(f"POS t={self._t:.1f} st={self._state} ({obs.self.lat:.5f},{obs.self.lon:.5f}) spd={obs.self.speed:.0f}")

        # ── 通信：解析队友消息（T:<uid>:lat,lon 求援 / D:<uid>:lat,lon 诱饵）──
        help_calls: List[Tuple[float, float]] = []
        help_uid: Optional[str] = None
        abandoned = False
        for m in obs.comm_inbox:
            if m.sender_uid == self.my_uid:
                continue  # 忽略自己的广播
            # 引擎 comm inbox 不清空：同一消息会被永久重复投递。
            # 按 (sender, payload) 去重，只处理一次（payload 含位置，目标
            # 移动时 payload 变化 → 新消息仍会处理）。
            key = (m.sender_uid, m.payload)
            if key in self._seen_msgs:
                continue
            self._seen_msgs.append(key)
            # L:<uid>:<t> — HELP 已锁定共享目标（MAIN 用来延长 dwell 窗口）
            if m.payload.startswith("L:"):
                parts = m.payload[2:].split(":", 1)
                if (len(parts) == 2 and parts[0] and parts[1]
                        and self._state == self.ENGAGE_MAIN
                        and parts[0] not in self._helper_lock_t):
                    self._helper_lock_t[parts[0]] = self._t
                continue
            # A:<uid>:<t> — 支援机已响应 T:（在途）；MAIN 据此延长超时等它到位
            if m.payload.startswith("A:"):
                parts = m.payload[2:].split(":", 1)
                if (len(parts) == 2 and parts[0] and parts[1]
                        and self._state == self.ENGAGE_MAIN):
                    self._helper_ack_t[parts[0]] = self._t
                continue
            parsed = _parse_msg(m.payload)
            if parsed is None:
                continue
            uid, (llat, llon) = parsed
            if m.payload.startswith("T:"):
                help_calls.append((llat, llon))
                if help_uid is None:
                    help_uid = uid
                # v4.2e：双 MAIN 同目标（同时 VERIFY 提交、互未见对方 T:）→
                # uid 大者让位当 HELP：确定性不抖动，仅盯防初期（<8s）且尚无
                # 支援锁定；消除双机摆锤同弧 <200m 罚分 + 第二架浪费。
                if (self._state == self.ENGAGE_MAIN
                        and uid > self.my_uid and self._engage_t < 8.0
                        and not self._helper_lock_t
                        and self._cand is not None
                        and _hav(llat, llon, self._cand[0], self._cand[1])
                        < _MATCH_M):
                    self._log(f"ENGAGE_MAIN yield -> HELP ({uid})")
                    self._state = self.ENGAGE_HELP
                    self._help_from = uid
                    self._cand = (llat, llon)
                    self._ema.reset()
                    self._ema.seed(llat, llon)
                    self._last_det = (llat, llon)
                    self._last_det_t = self._t
                    self._engage_t = 0.0
                    self._nd_on_engage = self._n_destroyed
                    self._reacq = False
                    self._reacq_pt = None
                # HELP 正在盯同一 MAIN 的候选：用最新求援位置重新对准（目标会移动）
                if (self._state == self.ENGAGE_HELP
                        and uid == self._help_from and self._cand is not None
                        and not self._near_blacklist(llat, llon)
                        and _hav(llat, llon, self._cand[0], self._cand[1])
                        < 2 * _ABANDON_M):
                    # v4.2g：EMA 平滑吸收 T:（不再裸重播种——原始检测 σ75m，
                    # 直接换会让站位/候选跟着噪声跳，HELP 永远落不了位）
                    self._ema.update(llat, llon, self._dt)
                    self._cand = self._ema.predict(0.0)
                    self._last_det = (llat, llon)
                    self._last_det_t = self._t
            elif m.payload.startswith("D:"):
                # v4.2d：D: 不再拉黑（共享黑名单曾让真目标被永久忽略）；
                # 只有自己跟随的 MAIN 宣判诱饵才出队放弃（精确匹配，避免误伤）
                if (not self._cand_real and self._engage_t > 2.0
                        and self._state == self.ENGAGE_HELP
                        and uid == self._help_from
                        and self._cand is not None
                        and _hav(llat, llon, self._cand[0], self._cand[1])
                        < _ABANDON_M):
                    self._log("abandon decoy (D:) -> SEARCH")
                    self._state = self.SEARCH
                    self._cand = None
                    self._speed.reset()
                    self._help_from = None
                    self._helper_lock_t = {}
                    self._helper_ack_t = {}
                    abandoned = True

        # 收到诱饵宣判并放弃 → 立即回搜索，避免本 tick 又被求援拉走（抖动根因）
        if abandoned and self._state == self.SEARCH:
            return self._search_cmds()

        # 收到求援 → 空闲/确认中响应协助（3 机收敛同一目标；MAIN 一律不弃守，
        # 避免团队在不同候选间来回换目标 → K=2 dwell 永远累积不起来）
        if (help_calls and self._state in (self.SEARCH, self.VERIFY)):
            hlat, hlon = help_calls[0]
            if not self._near_blacklist(hlat, hlon):
                if self._state != self.SEARCH:
                    self._log(f"{self._state}->HELP assist ({hlat:.5f},{hlon:.5f})")
                self._state = self.ENGAGE_HELP
                self._help_from = help_uid
                self._cand = (hlat, hlon)
                self._ema.reset()
                self._ema.seed(hlat, hlon)
                self._speed.reset()
                self._verify_start_t = self._t
                self._cand_real = False
                self._engage_t = 0.0
                self._nd_on_engage = self._n_destroyed
                self._last_det_t = self._t
                self._reacq = False
                self._reacq_pt = None
                self._helper_lock_t = {}
                self._helper_ack_t = {}
                self._help_station = None
                # v4.2f：响应求援即广播 A:（一次）并自记 ack——本机转场需要
                # 时间，MAIN 据此把超时从 22s 延长到 75s，避免半路被 D: 放弃
                self._helper_ack_t[self.my_uid] = self._t
                return ([broadcast(f"A:{self.my_uid}:{self._t:.1f}")]
                        + self._engage_help_cmds(obs))

        # ── 状态机 ──
        if self._state == self.SEARCH:
            return self._decide_search(obs, det, help_calls)
        if self._state == self.VERIFY:
            return self._decide_verify(obs, det)
        if self._state in (self.ENGAGE_MAIN, self.ENGAGE_HELP):
            return self._decide_engage(obs, det, n_destroyed)
        return self._search_cmds()

    # ── SEARCH：巡逻，响应检测与求援 ──────────────────────────────────

    def _decide_search(self, obs: CoopObs, det, help_calls) -> List[Command]:
        # 1) 检测到候选 → 转入确认（求援已在 decide 统一处理）
        if det.detected and det.target_lat is not None:
            if not self._near_blacklist(det.target_lat, det.target_lon):
                self._log(f"SEARCH->VERIFY det=({det.target_lat:.5f},{det.target_lon:.5f})")
                self._state = self.VERIFY
                self._cand = (det.target_lat, det.target_lon)
                self._ema.reset()
                self._ema.seed(det.target_lat, det.target_lon)
                self._speed.reset()
                self._speed.append(det.target_lat, det.target_lon, self._t)
                self._verify_start_t = self._t
                self._cand_real = False
                self._verify_t = 0.0
                self._samples = 1
                self._last_det_t = self._t
                return self._verify_cmds(obs)
        # 3) 继续巡逻
        return self._search_cmds()

    # ── VERIFY：确认目标持续存在 ──────────────────────────────────────

    def _decide_verify(self, obs: CoopObs, det) -> List[Command]:
        self._verify_t += self._dt
        if det.detected and det.target_lat is not None:
            self._handle_near_det(det.target_lat, det.target_lon)
            if self._cand is not None and _hav(
                    det.target_lat, det.target_lon,
                    self._cand[0], self._cand[1]) < _MATCH_M:
                self._speed.append(det.target_lat, det.target_lon, self._t)
        lost = self._t - self._last_det_t > _LOST_TIMEOUT_S
        # v4.2b：真目标与诱饵均以 ~10m/s 移动（A* 注入丢弃 WaitTime，
        # points.json 的 Start.WaitTime=30 不生效）→ 速度判别作废。
        # 对任意候选 4s 持续检出即提交，让全队收敛盯防（K=2 唯一路径）。
        commit = (self._verify_t >= 4.0
                  and self._samples >= 20)
        if commit:
            self._log(f"VERIFY->MAIN cand=({self._cand[0]:.5f},{self._cand[1]:.5f}) "
                      f"samples={self._samples}")
            self._cand_real = False
            self._help_from = None
            self._state = self.ENGAGE_MAIN
            self._engage_t = 0.0
            self._nd_on_engage = self._n_destroyed
            self._last_bc_t = -1e9
            self._last_det_t = self._t
            self._reacq = False
            self._reacq_pt = None
            self._helper_lock_t = {}
            self._helper_ack_t = {}
            return self._engage_main_cmds(obs)
        if lost or self._verify_t > 15.0:
            self._state = self.SEARCH
            self._cand = None
            self._speed.reset()
            self._help_from = None
            return self._search_cmds()
        # 不提前广播：避免大量诱饵候选把支援机拉来拉去（抖动）。MAIN 提交时才广播。
        return self._verify_cmds(obs)

    # ── ENGAGE：双机盯防 20s 直到摧毁或判定诱饵 ───────────────────────

    def _decide_engage(self, obs: CoopObs, det, n_destroyed) -> List[Command]:
        self._engage_t += self._dt
        # 摧毁检测：评分器判定成功（≥2 机连续锁定 20s）
        if n_destroyed > self._nd_on_engage:
            self._log(f"ENGAGE KILL SUCCESS nd={n_destroyed} engage_t={self._engage_t:.1f}")
            self._add_blacklist(*self._cand)   # 已处理，勿再盯
            self._reacq = False
            self._reacq_pt = None
            self._helper_lock_t = {}
            self._helper_ack_t = {}
            self._state = self.SEARCH
            self._cand = None
            self._speed.reset()
            self._cand_real = False
            self._help_from = None
            return self._search_cmds()
        # 更新目标位置（持续检出；限候选附近，防止被相邻诱饵带偏）
        if det.detected and det.target_lat is not None:
            if self._reacq:
                # 重捕成功：旧候选已过期，直接跳到新检出点续锁
                self._cand = (det.target_lat, det.target_lon)
                self._ema.reset()
                self._ema.seed(det.target_lat, det.target_lon)
                self._last_det = (det.target_lat, det.target_lon)
                self._last_det_t = self._t
                self._reacq = False
                self._reacq_pt = None
                self._log(f"ENGAGE REACQ->LOCK "
                          f"({det.target_lat:.5f},{det.target_lon:.5f})")
            else:
                match_m = (_HELP_MATCH_M if self._state == self.ENGAGE_HELP
                           else _TRACK_M)
                self._handle_near_det(det.target_lat, det.target_lon, match_m)
                if self._cand is not None and _hav(
                        det.target_lat, det.target_lon,
                        self._cand[0], self._cand[1]) < match_m:
                    self._last_det_t = self._t
                    self._last_det = (det.target_lat, det.target_lon)
        elif self._t - self._last_det_t > 2.0:
            self._last_det = None  # 检测过期，回退 EMA
        # 盯防中长时间看不到目标 → 原地 REACQ 重捕，不弃守、不拉黑。
        # v4.2c：旧逻辑 LOST→SEARCH+D: 广播 → 全队黑名单，真目标被永久
        # 忽略（out9 10003 只 coop 5.6s 后不再被盯的根因）。HELP 转场中
        # （距候选 >站位距离+100m）不判丢失，到达后再试。
        enroute = (self._state == self.ENGAGE_HELP and self._cand is not None
                   and _hav(obs.self.lat, obs.self.lon,
                            self._cand[0], self._cand[1]) > _HELP_OFFSET_M + 100.0)
        if (not enroute and self._engage_t > 3.0
                and self._t - self._last_det_t > _REACQ_DET_S):
            # v4.2d：REACQ 只保留给有 HELP 锁定（L:）的 MAIN——双机已就位，
            # 原地重捕不浪费收敛成果；其余掉锁直接回搜索（对诱饵死磕是净负
            # 优化，v42c1 实测 REACQ 全程把 3 目标 coop 拖成 0）
            if self._helper_lock_t and self._state == self.ENGAGE_MAIN:
                if not self._reacq:
                    self._log(f"ENGAGE REACQ (no det {self._t - self._last_det_t:.0f}s)")
                    self._reacq = True
                    self._reacq_pt = self._ema.predict(0.0)
            else:
                self._state = self.SEARCH
                self._cand = None
                self._speed.reset()
                self._cand_real = False
                self._help_from = None
                self._reacq = False
                self._reacq_pt = None
                self._helper_lock_t = {}
                self._helper_ack_t = {}
                return self._search_cmds()
        # 超时未摧毁 → 按诱饵拉黑+广播。v4.2f：无 L: 且无人应答（A:）时 22s
        # 快速释放（诱饵占多数）；有支援在途（A:）延长到 75s 等转场到位；有
        # HELP 锁定（L:）延长到锁定后 +25s（给足 20s 双机 dwell + 余量）。
        base = _TIMEOUT_BASE_S
        if self._helper_lock_t:
            base = max(base, max(self._helper_lock_t.values()) + 25.0)
        elif (self._helper_ack_t
                and self._t - max(self._helper_ack_t.values()) < 75.0):
            base = 75.0
        timeout_s = min(base, 75.0)
        if self._engage_t > timeout_s:
            tlat, tlon = _clamp(*self._ema.predict(0.0))
            cmds: List[Command] = []
            self._log(f"ENGAGE TIMEOUT -> decoy at cand={self._cand}")
            self._add_blacklist(tlat, tlon)
            self._reacq = False
            self._reacq_pt = None
            self._helper_lock_t = {}
            self._helper_ack_t = {}
            # v4.2d：仅 MAIN 宣判 D:（HELP 超时静默退出，避免双机各自广播
            # 宣判、消息互相干扰）
            if self._state == self.ENGAGE_MAIN:
                cmds.append(broadcast(
                    f"D:{self.my_uid}:{tlat:.5f},{tlon:.5f}"))
            self._state = self.SEARCH
            self._cand = None
            self._speed.reset()
            self._cand_real = False
            self._help_from = None
            cmds += self._search_cmds()
            return cmds
        # 持续盯防
        if self._state == self.ENGAGE_MAIN:
            return self._engage_main_cmds(obs)
        return self._engage_help_cmds(obs)




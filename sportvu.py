import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from typing import Optional

# ── Court constants ────────────────────────────────────────────────────────────
HOOP_LEFT    = (5.25,  25, 10)
HOOP_RIGHT   = (88.75, 25, 10)
HALF_COURT_X = 47.0

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_BALL_HEIGHT_FT        = 7.0
MAX_PLAYER_BALL_DIST_FT   = 1.5
JUMP_SHOT_MIN_HEIGHT_FT   = 10.5
HOOP_Y_TOLERANCE_FT       = 1.5
HOOP_Y_TOLERANCE_3PT_FT   = 2.0
FRAMES_PER_SEC            = 25
MOVING_AWAY_FRAMES        = 4
RESUME_AFTER_SHOT_FRAMES  = 5
RESUME_AFTER_LAYUP_FRAMES = 5
LAYUP_SHOOTER_HOOP_DIST   = 8.0    # shooter must be within 8ft of hoop when ball hits 10ft
DUNK_INITIAL_SHOOTER_DIST = 10.0   # shooter must be within 10ft of hoop at potential shot moment
DUNK_BALL_HOOP_DIST       = 1.0    # ball must get within 1ft of hoop
DUNK_BALL_MIN_HEIGHT      = 10.0   # ball must be >= 10ft when within 1ft of hoop
DUNK_SHOOTER_HOOP_DIST    = 3.5    # shooter must be within 3.5ft of hoop at that moment
DUNK_MAX_WAIT_FRAMES      = 25     # cap search at 1 second
DUNK_MAX_HEIGHT_FT        = 11.25  # ball must not exceed this height within 3ft of hoop
DUNK_MAX_HEIGHT_RADIUS    = 3.0    # radius within which the height cap applies
LAYUP_BALL_HOOP_DIST      = 4.0    # ball must be within 4ft of hoop when it hits 10ft

# ── Read in the SportVU tracking data ─────────────────────────────────────────
sportvu = []
with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)


# ── Build player lookup from roster at top of JSON ────────────────────────────
def build_player_lookup(sportvu_data: dict) -> dict:
    """
    Returns {player_id: {"name": "First Last", "team_abbr": "BOS", "team_id": ...}}
    by reading the visitor/home rosters in the first event.
    """
    lookup = {}
    first_event = sportvu_data["events"][0]
    for side in ("visitor", "home"):
        team_info = first_event[side]
        team_abbr = team_info["abbreviation"]
        team_id   = team_info["teamid"]
        for player in team_info["players"]:
            pid  = player["playerid"]
            name = f"{player['firstname']} {player['lastname']}"
            lookup[pid] = {
                "name":      name,
                "team_abbr": team_abbr,
                "team_id":   team_id,
            }
    return lookup


# ── Helpers ────────────────────────────────────────────────────────────────────
def xy_distance(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def shot_point_value(shooter_x: float, shooter_y: float, attacking_hoop: tuple) -> int:
    """
    Returns 3 if the shot is a 3-pointer, 2 otherwise.

    - If the shooter is 'above the hoop' (past the baseline side), it's a 3
      if they are > 23.75ft from the hoop center.
    - If the shooter is 'below the hoop' (corner), it's a 3 if their y is
      outside the 3-6 range (i.e. y < 3 or y > 47).
    """
    hoop_x, hoop_y, _ = attacking_hoop
    THREE_POINT_RADIUS = 23.75
    CORNER_Y_MIN       = 3.0
    CORNER_Y_MAX       = 47.0

    # Is the shooter on the same side of the hoop as the paint (i.e. not in the corner)?
    # "Above the hoop" means they're further from the baseline than the hoop itself.
    if attacking_hoop[0] < HALF_COURT_X:
        # Attacking left hoop (x=5.25): above the hoop means shooter_x > hoop_x
        above_hoop = shooter_x > hoop_x
    else:
        # Attacking right hoop (x=88.75): above the hoop means shooter_x < hoop_x
        above_hoop = shooter_x < hoop_x

    if above_hoop:
        dist = xy_distance(shooter_x, shooter_y, hoop_x, hoop_y)
        return 3 if dist > THREE_POINT_RADIUS else 2
    else:
        # Corner three: shooter is behind/level with the hoop — use y position
        return 3 if (shooter_y < CORNER_Y_MIN or shooter_y > CORNER_Y_MAX) else 2


def classify_shot_zone(shooter_x: float, shooter_y: float,
                       attacking_hoop: tuple, shot_type: str) -> Optional[str]:
    """
    Classifies a shot into its court zone based on shooter position.
    Returns None for 3pt jump shots (not yet zoned).
    Layups and dunks -> low_paint.

    All zones defined relative to distance from the attacking baseline (dx),
    so the same logic works for both left and right hoops.

    Zones:
      low_paint:    dx  0-14,  y 17-33
      high_paint:   dx 14-19,  y 17-33
      straight_up_2: dx > 19,  y 17-33
      baseline_2:   dx  0-14,  y <17 or y >33
      wing_2:       dx > 14,   y <17 or y >33
    """
    if shot_type == "3pt_jump_shot":
        hoop_x = attacking_hoop[0]
        if hoop_x < HALF_COURT_X:
            dx = shooter_x
        else:
            dx = 94.0 - shooter_x

        if dx < 14:
            return "corner_3"
        elif 17 <= shooter_y <= 33:
            return "straight_up_3"
        else:
            return "wing_3"

    if shot_type in ("layup", "dunk"):
        return "low_paint"

    hoop_x = attacking_hoop[0]
    if hoop_x < HALF_COURT_X:
        dx = shooter_x           # left hoop: baseline is x=0
    else:
        dx = 94.0 - shooter_x   # right hoop: baseline is x=94

    in_paint_y = 17 <= shooter_y <= 33

    if in_paint_y:
        if dx < 14:
            return "low_paint"
        elif dx < 19:
            return "high_paint"
        else:
            return "straight_up_2"
    else:
        if dx < 14:
            return "baseline_2"
        else:
            return "wing_2"


ZONE_SCORES = {
    "corner_3":      10.0,
    "low_paint":      9.7,
    "straight_up_3":  7.5,
    "wing_3":         4.5,
    "straight_up_2":  4.0,
    "high_paint":     3.9,
    "baseline_2":     3.3,
    "wing_2":         1.0,
}


def zone_score(zone: Optional[str]) -> Optional[float]:
    """Returns the base shot quality score (1-10) for a given zone."""
    if zone is None:
        return None
    return ZONE_SCORES.get(zone)


def defender_multiplier(defender_dist: float) -> float:
    """
    Returns a multiplier based on nearest defender distance from shooter.
      <= 2ft  -> 0.75 (heavily contested)
      >= 6ft  -> 1.00 (open)
      2-6ft   -> linear scale between 0.75 and 1.00
    """
    if defender_dist <= 2.0:
        return 0.75
    if defender_dist >= 6.0:
        return 1.00
    return 0.75 + (defender_dist - 2.0) / (6.0 - 2.0) * (1.00 - 0.75)



def format_game_clock(quarter: int, time_left: float) -> str:
    """Return a human-readable game clock string e.g. 'Q2 03:14'."""
    mins = int(time_left) // 60
    secs = time_left % 60
    return f"Q{quarter} {mins:01d}:{secs:05.2f}"


def parse_moment(raw: list) -> dict:
    quarter, timestamp_ms, time_left_quarter, time_left_shot_clock, _, positions = raw
    ball_raw = positions[0]
    ball = {
        "team_id": ball_raw[0], "player_id": ball_raw[1],
        "x": ball_raw[2], "y": ball_raw[3], "z": ball_raw[4]
    }
    players = [
        {"team_id": p[0], "player_id": p[1], "x": p[2], "y": p[3], "z": p[4]}
        for p in positions[1:]
    ]
    return {
        "quarter": quarter,
        "timestamp_ms": timestamp_ms,
        "time_left_quarter": time_left_quarter,
        "time_left_shot_clock": time_left_shot_clock,
        "ball": ball,
        "players": players,
    }


def flatten_moments(events: list) -> list:
    """Flatten all moments across all events into a single ordered list,
    deduplicating by timestamp since the same moment can appear in multiple events."""
    seen_timestamps = set()
    all_moments = []
    for event in events:
        for raw_moment in event["moments"]:
            ts = raw_moment[1]  # index 1 is the timestamp_ms
            if ts in seen_timestamps:
                continue
            seen_timestamps.add(ts)
            all_moments.append(parse_moment(raw_moment))
    return all_moments


# ── Tip-off: determine which hoop each team attacks in Q1 ─────────────────────
def determine_attacking_hoops(events: list) -> dict:
    centre_x, centre_y = HALF_COURT_X, 25.0

    for event in events:
        for raw_moment in event["moments"]:
            moment = parse_moment(raw_moment)
            if moment["quarter"] != 1:
                continue

            sorted_players = sorted(
                moment["players"],
                key=lambda p: xy_distance(p["x"], p["y"], centre_x, centre_y)
            )
            tipper = sorted_players[0]

            def attacking_hoop_for(player):
                dist_left  = xy_distance(player["x"], player["y"], HOOP_LEFT[0],  HOOP_LEFT[1])
                dist_right = xy_distance(player["x"], player["y"], HOOP_RIGHT[0], HOOP_RIGHT[1])
                return HOOP_RIGHT if dist_left < dist_right else HOOP_LEFT

            tipper_hoop = attacking_hoop_for(tipper)
            other_hoop  = HOOP_LEFT if tipper_hoop == HOOP_RIGHT else HOOP_RIGHT

            # Find the other team's id
            other_team_id = next(
                p["team_id"] for p in moment["players"]
                if p["team_id"] != tipper["team_id"]
            )

            attacking_hoops = {
                tipper["team_id"]: tipper_hoop,
                other_team_id:     other_hoop,
            }

            for team_id, hoop in attacking_hoops.items():
                side = "RIGHT" if hoop[0] > HALF_COURT_X else "LEFT"
                print(f"[TIP-OFF] Team {team_id} attacks {side} hoop")

            return attacking_hoops

    raise RuntimeError("Could not determine attacking hoops from tip-off data.")


def get_attacking_hoop(team_id: int, quarter: int, q1_hoops: dict) -> tuple:
    q1_hoop = q1_hoops[team_id]
    if quarter % 2 == 1:
        return q1_hoop
    return HOOP_RIGHT if q1_hoop == HOOP_LEFT else HOOP_LEFT


def offensive_team_for_ball(ball_x: float, quarter: int, q1_hoops: dict) -> Optional[int]:
    if ball_x == HALF_COURT_X:
        return None
    ball_on_right = ball_x > HALF_COURT_X
    for team_id in q1_hoops:
        hoop = get_attacking_hoop(team_id, quarter, q1_hoops)
        if (hoop[0] > HALF_COURT_X) == ball_on_right:
            return team_id
    return None


# ── Jump shot confirmation ─────────────────────────────────────────────────────
def check_dunk(moments: list, release_idx: int, attacking_hoop: tuple,
              shooter_pos: dict) -> tuple:
    """
    Checks if the potential shot is a dunk.

    First checks shooter is within 10ft of hoop at potential shot moment.
    Then scans up to 25 frames for the ball to get within 1ft x/y of the hoop.
    At that moment checks:
      1. Ball is >= 10ft high
      2. Shooter is within 3.5ft of the hoop

    Returns (is_dunk: bool, end_idx: int)
    """
    hoop_x, hoop_y = attacking_hoop[0], attacking_hoop[1]

    # ── Initial check: shooter within 10ft of hoop at potential shot moment ───
    shooter_dist = xy_distance(shooter_pos["x"], shooter_pos["y"], hoop_x, hoop_y)
    if shooter_dist > DUNK_INITIAL_SHOOTER_DIST:
        return False, release_idx

    # ── Scan for ball to get within 1ft of hoop ───────────────────────────────
    close_idx = None
    for j in range(release_idx, release_idx + DUNK_MAX_WAIT_FRAMES + 1):
        if j >= len(moments):
            return False, release_idx
        ball = moments[j]["ball"]
        # If ball exceeds height cap while within 3ft of hoop → not a dunk
        if (xy_distance(ball["x"], ball["y"], hoop_x, hoop_y) <= DUNK_MAX_HEIGHT_RADIUS
                and ball["z"] > DUNK_MAX_HEIGHT_FT):
            return False, release_idx
        if xy_distance(ball["x"], ball["y"], hoop_x, hoop_y) <= DUNK_BALL_HOOP_DIST:
            close_idx = j
            break

    if close_idx is None:
        return False, release_idx

    # ── Condition 1: ball >= 10ft high at that moment ─────────────────────────
    if moments[close_idx]["ball"]["z"] < DUNK_BALL_MIN_HEIGHT:
        return False, release_idx

    # ── Condition 2: ball z is decreasing immediately after the 1ft threshold ─
    next_idx = close_idx + 1
    if next_idx < len(moments):
        if moments[next_idx]["ball"]["z"] >= moments[close_idx]["ball"]["z"]:
            return False, release_idx

    # ── Condition 3: shooter within 3.5ft of hoop at that moment ─────────────
    # Use the shooter's position from the closest frame that has them
    shooter_now = next(
        (p for p in moments[close_idx]["players"]
         if p["player_id"] == shooter_pos["player_id"]),
        shooter_pos  # fall back to release position if not found
    )
    if xy_distance(shooter_now["x"], shooter_now["y"], hoop_x, hoop_y) > DUNK_SHOOTER_HOOP_DIST:
        return False, release_idx

    # ── Confirmed dunk — find retreat index ───────────────────────────────────
    _, end_idx = check_height_before_retreat(moments, release_idx, attacking_hoop)
    return True, end_idx


def check_layup(moments: list, release_idx: int, attacking_hoop: tuple,
               shooter_pos: dict) -> tuple:
    """
    Checks if the potential shot is a layup.

    Waits until the ball first reaches >= 10ft, then checks:
      1. Shooter was within 8ft of the hoop at that moment
      2. Ball was within 4ft of the hoop at that moment

    Returns (is_layup: bool, end_idx: int) where end_idx is the frame where
    the ball was determined to be moving away from the hoop (for resume logic).
    Uses same 5-consecutive-frames retreat detection as jump shots.
    """
    MAX_WAIT_FRAMES = int(1.5 * FRAMES_PER_SEC)  # cap search at 1.5s
    hoop_x, hoop_y = attacking_hoop[0], attacking_hoop[1]

    # ── Find the first frame where ball hits 10ft ──────────────────────────────
    ten_ft_idx = None
    for j in range(release_idx, release_idx + MAX_WAIT_FRAMES + 1):
        if j >= len(moments):
            return False, release_idx
        if moments[j]["ball"]["z"] >= 10.0:
            ten_ft_idx = j
            break

    if ten_ft_idx is None:
        return False, release_idx

    # ── Condition 1: shooter within 8ft of hoop at that frame ─────────────────
    shooter_dist = xy_distance(shooter_pos["x"], shooter_pos["y"], hoop_x, hoop_y)
    if shooter_dist > LAYUP_SHOOTER_HOOP_DIST:
        return False, release_idx

    # ── Condition 2: ball within 4ft of hoop at that frame ────────────────────
    ball = moments[ten_ft_idx]["ball"]
    ball_dist = xy_distance(ball["x"], ball["y"], hoop_x, hoop_y)
    if ball_dist > LAYUP_BALL_HOOP_DIST:
        return False, release_idx

    # ── Confirmed layup — find retreat index for resume logic ─────────────────
    _, end_idx = check_height_before_retreat(moments, release_idx, attacking_hoop)
    return True, end_idx


def check_trajectory(moments: list, release_idx: int, attacking_hoop: tuple,
                     shooter_pos: dict) -> bool:
    """
    Wait until the ball is > 1.5ft (x/y) from the shooter's release position,
    then fit a best fit line through the next 10 frames of ball x/y positions
    and project to the hoop's x coordinate to check if y lands within 1.5ft
    of the hoop's y.

    If the ball never leaves the shooter's 1.5ft radius within 1.5s (37 frames),
    returns False (pump fake / not a real release).
    """
    LEAVE_RADIUS_FT    = 1.5
    MAX_WAIT_FRAMES    = int(1.5 * FRAMES_PER_SEC)   # 37 frames
    COLLECT_FRAMES     = 10

    sx, sy = shooter_pos["x"], shooter_pos["y"]

    # ── Phase 1: wait for ball to leave the 1.5ft radius ──────────────────────
    leave_idx = None
    for j in range(release_idx + 1, release_idx + MAX_WAIT_FRAMES + 1):
        if j >= len(moments):
            return False
        ball = moments[j]["ball"]
        if xy_distance(ball["x"], ball["y"], sx, sy) > LEAVE_RADIUS_FT:
            leave_idx = j
            break

    if leave_idx is None:
        return False   # ball never left — pump fake or held ball

    # ── Phase 2: collect next 10 frames and fit best fit line ─────────────────
    end = min(leave_idx + COLLECT_FRAMES, len(moments))
    xs  = [moments[k]["ball"]["x"] for k in range(leave_idx, end)]
    ys  = [moments[k]["ball"]["y"] for k in range(leave_idx, end)]

    if len(xs) < 2:
        return False

    coeffs = np.polyfit(xs, ys, 1)
    m, b   = coeffs

    # Perpendicular distance from hoop center to the best fit line (mx - y + b = 0)
    hoop_x, hoop_y = attacking_hoop[0], attacking_hoop[1]
    perp_dist = abs(m * hoop_x - hoop_y + b) / math.sqrt(m ** 2 + 1)

    # Use a wider tolerance for 3pt attempts since line errors are magnified at distance
    shooter_dist = xy_distance(shooter_pos["x"], shooter_pos["y"], hoop_x, hoop_y)
    tolerance = HOOP_Y_TOLERANCE_3PT_FT if shooter_dist > 23.75 else HOOP_Y_TOLERANCE_FT
    return perp_dist <= tolerance


def check_height_before_retreat(moments: list, release_idx: int, attacking_hoop: tuple) -> tuple:
    max_z_seen       = moments[release_idx]["ball"]["z"]
    increasing_count = 0
    prev_dist        = xy_distance(
        moments[release_idx]["ball"]["x"], moments[release_idx]["ball"]["y"],
        attacking_hoop[0], attacking_hoop[1]
    )

    i = release_idx + 1
    while i < len(moments):
        ball = moments[i]["ball"]
        curr_dist = xy_distance(ball["x"], ball["y"], attacking_hoop[0], attacking_hoop[1])
        max_z_seen = max(max_z_seen, ball["z"])

        if curr_dist > prev_dist:
            increasing_count += 1
        else:
            increasing_count = 0

        if increasing_count >= MOVING_AWAY_FRAMES:
            hit_height = max_z_seen >= JUMP_SHOT_MIN_HEIGHT_FT
            return hit_height, i

        prev_dist = curr_dist
        i += 1

    return max_z_seen >= JUMP_SHOT_MIN_HEIGHT_FT, len(moments) - 1


# ── Potential shot check ───────────────────────────────────────────────────────
def is_potential_shot_moment(moment: dict, q1_hoops: dict) -> tuple:
    """
    Returns (is_potential, offensive_team_id, attacking_hoop, nearest_player)
    """
    ball = moment["ball"]

    if ball["z"] < MIN_BALL_HEIGHT_FT:
        return False, None, None, None

    offensive_team_id = offensive_team_for_ball(ball["x"], moment["quarter"], q1_hoops)
    if offensive_team_id is None:
        return False, None, None, None

    offensive_players = [p for p in moment["players"] if p["team_id"] == offensive_team_id]
    if not offensive_players:
        return False, None, None, None

    nearest = min(offensive_players, key=lambda p: xy_distance(ball["x"], ball["y"], p["x"], p["y"]))
    dist = xy_distance(ball["x"], ball["y"], nearest["x"], nearest["y"])

    if dist > MAX_PLAYER_BALL_DIST_FT:
        return False, None, None, None

    attacking_hoop = get_attacking_hoop(offensive_team_id, moment["quarter"], q1_hoops)
    return True, offensive_team_id, attacking_hoop, nearest


# ── Core shot detection ────────────────────────────────────────────────────────
def detect_shots(sportvu_data: dict) -> list:
    events    = sportvu_data["events"]
    q1_hoops  = determine_attacking_hoops(events)
    moments   = flatten_moments(events)

    confirmed_shots = []
    i = 0

    while i < len(moments):
        moment = moments[i]

        is_potential, offensive_team_id, attacking_hoop, nearest_player = \
            is_potential_shot_moment(moment, q1_hoops)

        if not is_potential:
            i += 1
            continue

        # ── Check 1: dunk ─────────────────────────────────────────────────────
        is_dunk, end_idx = check_dunk(moments, i, attacking_hoop, nearest_player)

        if is_dunk:
            shot_type = "dunk"
        else:
            # ── Check 2: layup ─────────────────────────────────────────────────
            is_layup, end_idx = check_layup(moments, i, attacking_hoop, nearest_player)

        if is_dunk:
            pass  # shot_type already set
        elif is_layup:
            shot_type = "layup"
        else:
            # ── Check 2: jump shot trajectory ─────────────────────────────────
            if not check_trajectory(moments, i, attacking_hoop, nearest_player):
                i += 1
                continue

            # ── Check 3: jump shot height + distance ──────────────────────────
            hit_height, end_idx = check_height_before_retreat(moments, i, attacking_hoop)
            if not hit_height:
                i += 1
                continue
            point_value = shot_point_value(
                nearest_player["x"], nearest_player["y"], attacking_hoop
            )
            shot_type = "3pt_jump_shot" if point_value == 3 else "2pt_jump_shot"

        # ── Nearest defender distance ──────────────────────────────────────────
        defensive_players = [p for p in moment["players"] if p["team_id"] != offensive_team_id]
        if defensive_players:
            nearest_def_dist = min(
                xy_distance(nearest_player["x"], nearest_player["y"], p["x"], p["y"])
                for p in defensive_players
            )
        else:
            nearest_def_dist = 6.0  # no defender found — treat as open

        shot_zone  = classify_shot_zone(nearest_player["x"], nearest_player["y"], attacking_hoop, shot_type)
        base_score = zone_score(shot_zone)
        multiplier = defender_multiplier(nearest_def_dist)

        # ── Confirmed shot ─────────────────────────────────────────────────────
        quarter      = moment["quarter"]
        time_left    = moment["time_left_quarter"]
        time_elapsed = (quarter - 1) * 720 + (720 - time_left)

        confirmed_shots.append({
            "quarter":              quarter,
            "game_clock":           format_game_clock(quarter, time_left),
            "time_elapsed_s":       round(time_elapsed, 2),
            "time_left_shot_clock": moment["time_left_shot_clock"],
            "shooter_player_id":    nearest_player["player_id"],
            "shooter_team_id":      offensive_team_id,
            "ball_x":               round(moment["ball"]["x"], 3),
            "ball_y":               round(moment["ball"]["y"], 3),
            "ball_z":               round(moment["ball"]["z"], 3),
            "shooter_x":            round(nearest_player["x"], 3),
            "shooter_y":            round(nearest_player["y"], 3),
            "attacking_hoop":       attacking_hoop,
            "shot_type":            shot_type,
            "zone":                 shot_zone,
            "zone_score":           base_score,
            "nearest_def_dist":     round(nearest_def_dist, 3),
            "defender_multiplier":  round(multiplier, 4),
            "shot_quality":         round(base_score * multiplier, 4) if base_score is not None else None,
        })

        resume_frames = RESUME_AFTER_LAYUP_FRAMES if is_layup else RESUME_AFTER_SHOT_FRAMES
        i = end_idx + resume_frames

    # ── Deduplicate shots ─────────────────────────────────────────────────────
    # If two shots share the same shooter, are within 1 second of each other,
    # and the shooter position is within 2ft, keep only the first occurrence.
    deduped = []
    for shot in confirmed_shots:
        is_dup = False
        for kept in deduped:
            if (shot["shooter_player_id"] == kept["shooter_player_id"]
                    and abs(shot["time_elapsed_s"] - kept["time_elapsed_s"]) <= 1.0
                    and xy_distance(shot["shooter_x"], shot["shooter_y"],
                                    kept["shooter_x"], kept["shooter_y"]) <= 2.0):
                is_dup = True
                break
        if not is_dup:
            deduped.append(shot)

    print(f"[DEDUP] Removed {len(confirmed_shots) - len(deduped)} duplicate shot(s)")
    return deduped


# ── Write shot record CSV ──────────────────────────────────────────────────────
def write_shot_record(confirmed_shots: list, player_lookup: dict,
                      filepath: str = "shot_record.csv"):
    """
    Writes one row per confirmed jump shot with game clock, shooter name,
    team, and position info.
    """
    fieldnames = [
        "shot_num",
        "quarter",
        "game_clock",
        "time_elapsed_s",
        "shot_clock",
        "shooter_name",
        "shooter_team",
        "shooter_player_id",
        "shot_type",
        "zone",
        "shot_quality",
        "nearest_def_dist",
        "ball_x",
        "ball_y",
        "ball_z_at_release",
        "shooter_x",
        "shooter_y",
    ]

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, shot in enumerate(confirmed_shots, start=1):
            pid    = shot["shooter_player_id"]
            pinfo  = player_lookup.get(pid, {"name": f"Unknown ({pid})", "team_abbr": "???"})

            writer.writerow({
                "shot_num":           idx,
                "quarter":            shot["quarter"],
                "game_clock":         shot["game_clock"],
                "time_elapsed_s":     shot["time_elapsed_s"],
                "shot_clock":         shot["time_left_shot_clock"],
                "shooter_name":       pinfo["name"],
                "shooter_team":       pinfo["team_abbr"],
                "shooter_player_id":  pid,
                "shot_type":          shot["shot_type"],
                "zone":               shot["zone"],
                "shot_quality":       shot["shot_quality"],
                "nearest_def_dist":   shot["nearest_def_dist"],
                "ball_x":             shot["ball_x"],
                "ball_y":             shot["ball_y"],
                "ball_z_at_release":  shot["ball_z"],
                "shooter_x":          shot["shooter_x"],
                "shooter_y":          shot["shooter_y"],
            })

    print(f"\n[RECORD] Shot log written to '{filepath}' ({len(confirmed_shots)} shots)")


# ── YOUR SOLUTION ──────────────────────────────────────────────────────────────
player_lookup   = build_player_lookup(sportvu)
confirmed_shots = detect_shots(sportvu)

print(f"\nFound {len(confirmed_shots)} confirmed shot(s)")
dunks      = sum(1 for s in confirmed_shots if s["shot_type"] == "dunk")
layups     = sum(1 for s in confirmed_shots if s["shot_type"] == "layup")
two_pt     = sum(1 for s in confirmed_shots if s["shot_type"] == "2pt_jump_shot")
three_pt   = sum(1 for s in confirmed_shots if s["shot_type"] == "3pt_jump_shot")
print(f"  Dunks:             {dunks}")
print(f"  Layups:            {layups}")
print(f"  2pt jump shots:    {two_pt}")
print(f"  3pt jump shots:    {three_pt}")

write_shot_record(confirmed_shots, player_lookup, filepath="shot_record.csv")

shot_times   = np.array([s["time_elapsed_s"] for s in confirmed_shots])
shot_quality = np.array([s["shot_quality"] if s["shot_quality"] is not None else 0.0 for s in confirmed_shots])


# ── Timeline display ───────────────────────────────────────────────────────────
# DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_quality, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot quality')

ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.tick_params(axis='x', length=20)
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,720,1440,2160,2880]))
ax.set_yticks([])

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax+5)
ax.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')

plt.tight_layout()

plt.savefig("Shot_Timeline.png")
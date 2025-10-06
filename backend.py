# -*- coding: utf-8 -*-
import os
import json
import math
import traceback
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Objective, minimize,
    Constraint, Any as PyAny, value, SolverFactory, Reals
)

# =========================================================
# Genel Ayarlar
# =========================================================
DATASET_PATH = Path(os.environ.get("DATASET_PATH", str(Path(__file__).parent / "dataset.json")))
MAX_SOLVE_SECONDS = 26

# Paket döngülerini caydırmak için küçük epsilon maliyeti (TL/km)
PACKAGE_EPS_TL_PER_KM = 1e-3

app = Flask(__name__, static_url_path="", static_folder=".")

# ---------------- Azure OpenAI (Chat & Görüntüden okuma) ----------------
from openai import AzureOpenAI
AZURE_OPENAI_API_KEY   = os.environ.get("AZURE_OPENAI_API_KEY", "d0167637046c4443badc4920cc612abb")
AZURE_OPENAI_ENDPOINT  = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-fnss.openai.azure.com")
AZURE_DEPLOYMENT_NAME  = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VER   = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

try:
    aoai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VER,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception:
    aoai_client = None


# =========================================================
# Yardımcılar
# =========================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_dataset_from_file() -> Dict[str, Any]:
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = [
            "cities", "main_depot",
            "vehicle_types", "vehicle_count",
            "distances", "packages", "minutil_penalty"
        ]
        for k in required:
            if k not in data:
                raise KeyError(f"dataset.json alan eksik: {k}")

        for tname, tinfo in data["vehicle_types"].items():
            if "hiz_kmh" not in tinfo:
                raise KeyError(f"vehicle_types['{tname}'] için 'hiz_kmh' eksik")

        for p in data["packages"]:
            for kk in ["id", "baslangic", "hedef", "agirlik", "ready_hour", "deadline_hour", "ceza"]:
                if kk not in p:
                    raise KeyError(f"Paket {p.get('id','?')} için alan eksik: {kk}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset.json bulunamadı: {DATASET_PATH}")
    except Exception as e:
        raise RuntimeError(f"dataset.json okunamadı: {e}")


def _normalize_initial_locations(vehicles: List[str], cities: List[str], main_depot: str, payload: Dict[str, Any]) -> Dict[str, str]:
    raw = payload.get("vehicle_initial_locations", {}) or {}
    init = {v: main_depot for v in vehicles}
    if isinstance(raw, dict):
        for k, city in raw.items():
            if k in vehicles and city in cities:
                init[k] = city
        for k, city in raw.items():
            if k not in vehicles and city in cities:
                pref = f"{k}_"
                for v in vehicles:
                    if v.startswith(pref):
                        init[v] = city
    for v in vehicles:
        if init.get(v) not in cities:
            init[v] = main_depot
    return init


def pick_solver():
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        s = AppsiHighs()
        try: s.config.time_limit = MAX_SOLVE_SECONDS
        except Exception: pass
        return "appsi_highs", s, True
    except Exception:
        pass
    for cand in ["highs", "cbc", "glpk", "cplex"]:
        try:
            s = SolverFactory(cand)
            if s is not None and s.available():
                try:
                    if cand == "highs": s.options["time_limit"] = MAX_SOLVE_SECONDS
                    elif cand == "cbc": s.options["seconds"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "glpk": s.options["tmlim"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "cplex":
                        s.options["timelimit"] = MAX_SOLVE_SECONDS
                        s.options["mipgap"]   = 0.05
                        s.options["threads"]  = 2
                except Exception: pass
                return cand, s, False
        except Exception:
            continue
    return None, None, False


# =========================================================
# Model (kopukluğu engelleyen kısıtlarla)
# =========================================================
def build_model(payload: Dict[str, Any]) -> Tuple[ConcreteModel, Dict[str, Any]]:
    cities: List[str] = payload["cities"]
    main_depot: str = payload["main_depot"]
    vt_info: Dict[str, Dict[str, Any]] = payload["vehicle_types"]
    vcount: Dict[str, int] = payload["vehicle_count"]

    vehicles: List[str] = [f"{vt}_{i}" for vt, cnt in vcount.items() for i in range(1, int(cnt) + 1)]
    init_loc = _normalize_initial_locations(vehicles, cities, main_depot, payload)

    distances: Dict[Tuple[str, str], float] = {}
    for i, j, d in payload["distances"]:
        distances[(i, j)] = float(d); distances[(j, i)] = float(d)
    for c in cities: distances[(c, c)] = 0.0

    packages_list: List[Dict[str, Any]] = payload["packages"]
    packages = {}
    for rec in packages_list:
        pid = str(rec["id"])
        packages[pid] = {
            "baslangic": rec["baslangic"],
            "hedef": rec["hedef"],
            "agirlik": float(rec["agirlik"]),
            "ready_hour": float(rec["ready_hour"]),
            "deadline_hour": float(rec["deadline_hour"]),
            "ceza_maliyeti": float(rec["ceza"]),
        }

    MINUTIL_PENALTY = safe_float(payload.get("minutil_penalty", 10.0), 10.0)

    model = ConcreteModel()
    model.Cities   = Set(initialize=cities)
    model.Vehicles = Set(initialize=vehicles)
    model.Packages = Set(initialize=list(packages.keys()))

    def vtype(v): return v.rsplit("_", 1)[0]

    model.Distance        = Param(model.Cities, model.Cities, initialize=lambda m, i, j: distances[(i, j)])
    model.VehicleCapacity = Param(model.Vehicles, initialize=lambda m, v: vt_info[vtype(v)]["kapasite"])
    model.TransportCost   = Param(model.Vehicles, initialize=lambda m, v: vt_info[vtype(v)]["maliyet_km"])
    model.FixedCost       = Param(model.Vehicles, initialize=lambda m, v: vt_info[vtype(v)]["sabit_maliyet"])
    model.MinUtilization  = Param(model.Vehicles, initialize=lambda m, v: vt_info[vtype(v)]["min_doluluk"])
    model.Speed           = Param(model.Vehicles, initialize=lambda m, v: vt_info[vtype(v)]["hiz_kmh"])

    model.PackageWeight   = Param(model.Packages, initialize=lambda m, p: packages[p]["agirlik"])
    model.PackageOrigin   = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["baslangic"])
    model.PackageDest     = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["hedef"])
    model.PackageReady    = Param(model.Packages, initialize=lambda m, p: packages[p]["ready_hour"])
    model.PackageDeadline = Param(model.Packages, initialize=lambda m, p: packages[p]["deadline_hour"])
    model.LatePenalty     = Param(model.Packages, initialize=lambda m, p: packages[p]["ceza_maliyeti"])

    # Süreler (saat)
    travel_time = {}
    for v in vehicles:
        spd = float(vt_info[vtype(v)]["hiz_kmh"]) if float(vt_info[vtype(v)]["hiz_kmh"]) > 0 else 1e-6
        for i in cities:
            for j in cities:
                dkm = distances[(i, j)]
                travel_time[(v, i, j)] = dkm / spd

    # ---------- Değişkenler ----------
    # Araç kenarı
    model.x = Var(model.Vehicles, model.Cities, model.Cities, domain=Binary)
    # Paket kenarı
    model.y = Var(model.Packages, model.Vehicles, model.Cities, model.Cities, domain=Binary)
    # Araç kullanımı
    model.z = Var(model.Vehicles, domain=Binary)
    # Gecikme
    model.lateness = Var(model.Packages, domain=NonNegativeReals)
    # Min. doluluk açığı
    model.minutil_shortfall = Var(model.Vehicles, domain=NonNegativeReals)

    # Araç-şehir ziyaret bayrağı
    model.w = Var(model.Vehicles, model.Cities, domain=Binary)
    # Bağlanırlık akışı (araç bazında)
    model.f = Var(model.Vehicles, model.Cities, model.Cities, domain=NonNegativeReals)

    # ---------- Amaç ----------
    def objective_rule(m):
        transport = sum(m.TransportCost[v] * m.Distance[i, j] * m.x[v, i, j]
                        for v in m.Vehicles for i in m.Cities for j in m.Cities if i != j)
        fixed     = sum(m.FixedCost[v] * m.z[v] for v in m.Vehicles)
        late      = sum(m.LatePenalty[p] * m.lateness[p] for p in m.Packages)
        minutil   = MINUTIL_PENALTY * sum(m.minutil_shortfall[v] for v in m.Vehicles)

        # Paket kenarlarına küçük mesafe maliyeti -> gereksiz paket döngüleri açılmasın
        eps_cost  = PACKAGE_EPS_TL_PER_KM * sum(m.Distance[i, j] * m.y[p, v, i, j]
                        for p in m.Packages for v in m.Vehicles for i in m.Cities for j in m.Cities if i != j)
        return transport + fixed + late + minutil + eps_cost
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # ---------- Kısıtlar ----------
    # Araç kullanıldı bayrağı
    def used_vehicle_flag_rule(m, v, i, j):
        if i == j: return Constraint.Skip
        return m.z[v] >= m.x[v, i, j]
    model.used_vehicle_flag = Constraint(model.Vehicles, model.Cities, model.Cities, rule=used_vehicle_flag_rule)

    # Paket başlangıçtan tam 1 çıkış
    def package_origin_rule(m, p):
        o = m.PackageOrigin[p]
        return sum(m.y[p, v, o, j] for v in m.Vehicles for j in m.Cities if j != o) == 1
    model.package_origin_constraint = Constraint(model.Packages, rule=package_origin_rule)

    # Paket hedefe tam 1 varış
    def package_destination_rule(m, p):
        d = m.PackageDest[p]
        return sum(m.y[p, v, i, d] for v in m.Vehicles for i in m.Cities if i != d) == 1
    model.package_destination_constraint = Constraint(model.Packages, rule=package_destination_rule)

    # Paket – ara şehirlerde akış korunumu
    def package_flow_conserv(m, p, k):
        o, d = m.PackageOrigin[p], m.PackageDest[p]
        if k == o or k == d: return Constraint.Skip
        inflow  = sum(m.y[p, v, i, k] for v in m.Vehicles for i in m.Cities if i != k)
        outflow = sum(m.y[p, v, k, j] for v in m.Vehicles for j in m.Cities if j != k)
        return inflow == outflow
    model.package_flow_conserv = Constraint(model.Packages, model.Cities, rule=package_flow_conserv)

    # Paket – her şehirde en fazla bir giriş ve bir çıkış (disjoint cycle'ları kısıtlar)
    def pkg_in_deg_le1(m, p, k):
        return sum(m.y[p, v, i, k] for v in m.Vehicles for i in m.Cities if i != k) <= 1
    def pkg_out_deg_le1(m, p, k):
        return sum(m.y[p, v, k, j] for v in m.Vehicles for j in m.Cities if j != k) <= 1
    model.pkg_in_deg_le1  = Constraint(model.Packages, model.Cities, rule=pkg_in_deg_le1)
    model.pkg_out_deg_le1 = Constraint(model.Packages, model.Cities, rule=pkg_out_deg_le1)

    # Paket kenarı ancak araç kenarı varsa (y <= x)
    def y_le_x_rule(m, p, v, i, j):
        if i == j: return Constraint.Skip
        return m.y[p, v, i, j] <= m.x[v, i, j]
    model.y_le_x = Constraint(model.Packages, model.Vehicles, model.Cities, model.Cities, rule=y_le_x_rule)

    # Kapasite (her kenarda)
    def capacity_rule(m, v, i, j):
        if i == j: return Constraint.Skip
        load = sum(m.PackageWeight[p] * m.y[p, v, i, j] for p in m.Packages)
        return load <= m.VehicleCapacity[v]
    model.capacity_constraint = Constraint(model.Vehicles, model.Cities, model.Cities, rule=capacity_rule)

    # Min. doluluk (ana depodan çıkışlar için soft)
    def min_utilization_soft_rule(m, v):
        start_city = init_loc[v]  # her aracın gerçek başlangıç şehri
        departures = sum(m.x[v, start_city, j] for j in m.Cities if j != start_city)  # 0..N
        loaded = sum(
            m.PackageWeight[p] * m.y[p, v, start_city, j]
            for p in m.Packages for j in m.Cities if j != start_city
        )
        target = m.MinUtilization[v] * m.VehicleCapacity[v] * departures
        return loaded + m.minutil_shortfall[v] >= target
    model.min_utilization_soft = Constraint(model.Vehicles, rule=min_utilization_soft_rule)    

    # Araç şehir ziyaret bayrakları: eğer bir kente giriyor/çıkıyorsa w=1
    def visit_out_imp(m, v, n):
        return sum(m.x[v, n, j] for j in m.Cities if j != n) <= m.w[v, n]
    def visit_in_imp(m, v, n):
        return sum(m.x[v, i, n] for i in m.Cities if i != n) <= m.w[v, n]
    model.visit_out_imp = Constraint(model.Vehicles, model.Cities, rule=visit_out_imp)
    model.visit_in_imp  = Constraint(model.Vehicles, model.Cities, rule=visit_in_imp)

    # Araç – giriş çıkış denklikleri (init'te çıkış = giriş + r, diğerlerinde giriş = çıkış)
    def vehicle_degree_balance(m, v, n):
        outn = sum(m.x[v, n, j] for j in m.Cities if j != n)
        inn  = sum(m.x[v, i, n] for i in m.Cities if i != n)
        if n == init_loc[v]:
            # Başlangıçta: çıkış - giriş >= 0 (yol başlayabilir), ayrıca bağlanırlık akışı bunu düzenler
            return outn >= inn
        return outn == inn
    model.vehicle_degree_balance = Constraint(model.Vehicles, model.Cities, rule=vehicle_degree_balance)

    # Tek zincir için bağlanırlık akışı:
    # M sabiti
    BIGM = len(cities)
    # f <= M * x
    def flow_cap(m, v, i, j):
        if i == j: return Constraint.Skip
        return m.f[v, i, j] <= BIGM * m.x[v, i, j]
    model.flow_cap = Constraint(model.Vehicles, model.Cities, model.Cities, rule=flow_cap)

    # Akış korunumları: init node kaynak, diğer ziyaret edilenler tüketici
    def flow_conserv(m, v, n):
        out_flow = sum(m.f[v, n, j] for j in m.Cities if j != n)
        in_flow  = sum(m.f[v, i, n] for i in m.Cities if i != n)
        if n == init_loc[v]:
            # kaynak = ziyaret edilen düğüm sayısı (init hariç)
            rhs = sum(m.w[v, k] for k in m.Cities if k != n)
            return out_flow - in_flow == rhs
        else:
            # ziyaret edildiyse 1 birim tüket
            return out_flow - in_flow == - m.w[v, n]
    model.flow_conserv = Constraint(model.Vehicles, model.Cities, rule=flow_conserv)

    # Paket varış ve gecikme
    def lateness_rule(m, p):
        total_time = sum(m.y[p, v, i, j] * travel_time[(v, i, j)]
                         for v in m.Vehicles for i in m.Cities for j in m.Cities if i != j)
        arrival = m.PackageReady[p] + total_time
        return m.lateness[p] >= arrival - m.PackageDeadline[p]
    model.lateness_calc = Constraint(model.Packages, rule=lateness_rule)

    meta = {
        "cities": cities,
        "vehicles": vehicles,
        "packages": packages,
        "distances": distances,
        "vehicle_types": vt_info,
        "main_depot": main_depot,
        "MINUTIL_PENALTY": MINUTIL_PENALTY,
        "initial_locations": init_loc,
        "travel_time": travel_time
    }
    return model, meta


# =========================================================
# Çözümden insan-okur sonuç üretimi
# =========================================================
def _order_path_from_edges(edges: List[Tuple[str, str]], start: str) -> List[Tuple[str, str]]:
    """edges: (i,j) listesi, tekli yol varsayımıyla sıralı hale getirir."""
    if not edges: return []
    nxt = {}
    indeg, outdeg = {}, {}
    for i, j in edges:
        nxt[i] = j
        outdeg[i] = outdeg.get(i, 0) + 1
        indeg[j]  = indeg.get(j, 0) + 1
    # başlangıcı bul
    cur = start
    if cur not in nxt:
        # eğer başlangıç listede yoksa, herhangi bir ucu başa al
        candidates = [i for (i, _) in edges if i not in indeg]
        if candidates: cur = candidates[0]
        else: cur = edges[0][0]
    ordered = []
    visited = set()
    while cur in nxt and (cur, nxt[cur]) not in visited:
        step = (cur, nxt[cur])
        ordered.append(step)
        visited.add(step)
        cur = nxt[cur]
    return ordered


def extract_results(model: ConcreteModel, meta: Dict[str, Any]) -> Dict[str, Any]:
    cities      = meta["cities"]
    vehicles    = meta["vehicles"]
    packages    = meta["packages"]
    distances   = meta["distances"]
    travel_time = meta["travel_time"]
    MINUTIL     = meta["MINUTIL_PENALTY"]

    res = {"objective": float(value(model.obj))}

    # Maliyet kırılımı
    transport = 0.0
    for v in vehicles:
        for i in cities:
            for j in cities:
                if i != j and value(model.x[v, i, j]) > 0.5:
                    transport += float(value(model.TransportCost[v])) * float(value(model.Distance[i, j]))
    fixed = sum(float(value(model.FixedCost[v])) for v in vehicles if value(model.z[v]) > 0.5)
    lateness_cost = sum(float(value(model.LatePenalty[p])) * float(value(model.lateness[p])) for p in model.Packages)
    minutil_pen = MINUTIL * sum(float(value(model.minutil_shortfall[v])) for v in vehicles)

    res["cost_breakdown"] = {
        "transport": transport,
        "fixed": fixed,
        "lateness": lateness_cost,
        "min_util_gap": float(minutil_pen),
    }

    # Araç rotaları (sıralı)
    vehicle_routes = []
    for v in sorted(vehicles):
        # v için kullanılan kenarlar
        v_edges = [(i, j) for i in cities for j in cities if i != j and value(model.x[v, i, j]) > 0.5]
        if not v_edges: continue

        # sıraya diz: init'ten başlayarak
        ordered = _order_path_from_edges(v_edges, meta["initial_locations"][v])

        legs = []
        for (i, j) in ordered:
            moved = []
            load  = 0.0
            for p in model.Packages:
                if value(model.y[p, v, i, j]) > 0.5:
                    moved.append(p)
                    load += float(value(model.PackageWeight[p]))
            legs.append({
                "from": i, "to": j,
                "km": float(distances[(i, j)]),
                "travel_hours": float(travel_time[(v, i, j)]),
                "packages": moved,
                "load_kg": load,
                "utilization_pct": (100.0 * load / float(value(model.VehicleCapacity[v]))) if load > 0 else 0.0
            })
        vehicle_routes.append({"vehicle": v, "capacity": float(value(model.VehicleCapacity[v])), "legs": legs})
    res["vehicle_routes"] = vehicle_routes

    # Paket özetleri (sıralı rota ve zamanlar)
    pkg_summaries = []
    for pid, pdata in packages.items():
        o, d = pdata["baslangic"], pdata["hedef"]
        ready, deadline = pdata["ready_hour"], pdata["deadline_hour"]

        # Paket kenarları (v bazında birleştir)
        edges = []
        steps = []
        for v in model.Vehicles:
            for i in cities:
                for j in cities:
                    if i != j and value(model.y[pid, v, i, j]) > 0.5:
                        edges.append((i, j))
        ordered = _order_path_from_edges(edges, o)

        ttot = 0.0
        timeline = []
        cur_time = ready
        for (i, j) in ordered:
            # bu kenarı hangi araç taşıdıysa onu bul
            veh = None; dur = None; km = float(distances[(i, j)])
            for v in model.Vehicles:
                if value(model.y[pid, v, i, j]) > 0.5:
                    veh = v
                    dur = float(travel_time[(v, i, j)])
                    break
            if dur is None:  # güvenlik
                # hız fark etmeksizin km/speed ~ süre
                anyv = vehicles[0]
                dur = float(travel_time[(anyv, i, j)])
            start_t = cur_time
            end_t   = cur_time + dur
            timeline.append({"vehicle": veh, "from": i, "to": j, "km": km, "duration_h": dur,
                             "start_h": round(start_t, 3), "end_h": round(end_t, 3)})
            cur_time = end_t
            ttot += dur

        arrival = ready + ttot
        late_h  = max(0.0, arrival - deadline)
        pkg_summaries.append({
            "id": pid,
            "origin": o, "dest": d,
            "weight": pdata["agirlik"],
            "ready_hour": ready,
            "deadline_by": deadline,
            "total_travel_hours": round(ttot, 3),
            "delivered_at": round(arrival, 3),
            "on_time": (arrival <= deadline + 1e-9),
            "lateness_hours": late_h,
            "lateness_penalty": late_h * pdata["ceza_maliyeti"],
            "route": timeline
        })
    res["packages"] = pkg_summaries
    return res


# =========================================================
# HTTP Routes
# =========================================================
@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/dataset", methods=["GET", "PUT", "POST"])
def dataset_endpoint():
    try:
        if request.method == "GET":
            if not DATASET_PATH.exists():
                return jsonify({"ok": False, "error": "dataset.json bulunamadı"}), 404
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"ok": True, "dataset": data})

        try:
            payload = request.get_json(force=True)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400

        required_keys = ["cities","main_depot","vehicle_types","vehicle_count","distances","packages","minutil_penalty"]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return jsonify({"ok": False, "error": f"Eksik alanlar: {', '.join(missing)}"}), 400

        for tname, tinfo in payload["vehicle_types"].items():
            if "hiz_kmh" not in tinfo:
                return jsonify({"ok": False, "error": f"'{tname}' tipi için 'hiz_kmh' alanı zorunlu."}), 400

        for p in payload["packages"]:
            for kk in ["id", "baslangic", "hedef", "agirlik", "ready_hour", "deadline_hour", "ceza"]:
                if kk not in p:
                    return jsonify({"ok": False, "error": f"Paket {p.get('id','?')} için alan eksik: {kk}"}), 400

        tmp_path = DATASET_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(DATASET_PATH)

        return jsonify({"ok": True, "message": "dataset.json güncellendi"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        if aoai_client is None:
            return jsonify({"ok": False, "error": "Azure OpenAI istemcisi oluşturulamadı (anahtar/endpoint)."}), 500

        payload = request.get_json(force=True) or {}
        user_messages = payload.get("messages", [])
        model_context = payload.get("context", {})

        sys_prompt = f"""
Sen bir lojistik optimizasyon asistanısın. Kullanıcıdan gelen çok duruşlu taşımacılık
parametrelerini (şehirler, ana depo, araç tip/sayıları, hızlar ve mesafeler, paketler:
hazır olma saati, termin saati, ceza; min. doluluk cezası) kullanarak net cevap ver.
Zaman modelinde dönem yok; süre = mesafe / hız. Geç teslim mümkündür (cezalı).
Adın: VRP Assist 2.0

Model JSON:
{model_context}

Kurallar:
- Sayısal/lojistik sorularda net hesap yap.
- Tutarsızlık görürsen hangi alanın düzelmesi gerektiğini söyle.
- Kısa ve anlaşılır cevap ver.
""".strip()

        completion = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": sys_prompt}, *user_messages],
            temperature=0.2, max_tokens=600,
        )
        answer = completion.choices[0].message.content
        return jsonify({"ok": True, "answer": answer})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


@app.route("/ai-pack", methods=["POST"])
def ai_pack():
    try:
        if aoai_client is None:
            return jsonify({"ok": False, "error": "Azure OpenAI istemcisi yok."}), 500

        payload = request.get_json(force=True) or {}
        image_b64 = payload.get("image_b64")
        cities    = payload.get("cities") or []
        defaults  = payload.get("defaults") or {"ready_hour": 0, "deadline_hour": 36, "ceza": 120}

        if not image_b64 or not isinstance(image_b64, str) or "base64," not in image_b64:
            return jsonify({"ok": False, "error": "Geçerli data URL (base64) bekleniyor: image_b64"}), 400

        sys_prompt = (
            "Görselde sevkiyat etiketi/fatura olabilir. "
            "Tek bir paket için şu JSON'u üret:\n"
            '{ "id":"P?", "baslangic":"<Şehir>", "hedef":"<Şehir>", '
            '"agirlik":<kg>, "ready_hour":<saat>, "deadline_hour":<saat>, "ceza":<TL/saat> }\n'
            f"Sadece şu şehirlerden birini kullan: {cities}. Saat/yoksa ready=0, deadline=36, ceza defaults.ceza. "
            "Yalnızca geçerli JSON döndür."
        )

        completion = aoai_client.chat.completions.create(
            model=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
            temperature=0.0, max_tokens=500,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Görselden tek paket bilgisi çıkar ve JSON döndür."},
                    {"type": "image_url", "image_url": {"url": image_b64}}
                ]}
            ],
        )

        raw = completion.choices[0].message.content.strip()
        try:
            pkg = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m: return jsonify({"ok": False, "error": "Modelden geçerli JSON alınamadı.", "raw": raw}), 400
            pkg = json.loads(m.group(0))

        out = {
            "id": str(pkg.get("id") or "P?").strip(),
            "baslangic": (pkg.get("baslangic") or (cities[0] if cities else "")).strip(),
            "hedef": (pkg.get("hedef") or (cities[0] if cities else "")).strip(),
            "agirlik": float(pkg.get("agirlik") or 0),
            "ready_hour": float(pkg.get("ready_hour") if pkg.get("ready_hour") is not None else defaults.get("ready_hour", 0)),
            "deadline_hour": float(pkg.get("deadline_hour") if pkg.get("deadline_hour") is not None else defaults.get("deadline_hour", 36)),
            "ceza": float(pkg.get("ceza") if pkg.get("ceza") is not None else defaults.get("ceza", 120)),
        }
        if cities:
            if out["baslangic"] not in cities: out["baslangic"] = cities[0]
            if out["hedef"]   not in cities: out["hedef"]   = cities[0]

        return jsonify({"ok": True, "package": out, "raw": raw})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e}", "trace": traceback.format_exc()}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


# =========================================================
# /solve
# =========================================================
@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            data = load_dataset_from_file()

        model, meta = build_model(data)
        solver_name, solver, is_appsi = pick_solver()
        if solver is None:
            return jsonify({"ok": False, "error": "Uygun MILP çözücüsü bulunamadı."}), 400

        if is_appsi:
            results = solver.solve(model)
            term = getattr(results, "termination_condition", None)
        else:
            try:
                results = solver.solve(model, tee=False, load_solutions=True)
            except TypeError:
                results = solver.solve(model, load_solutions=True)
            term = getattr(getattr(results, "solver", None), "termination_condition", None)

        def has_incumbent(m):
            try:
                for _, v in m.x.items():
                    if v.value is not None:
                        return True
            except Exception:
                pass
            return False

        diag = {
            "termination": str(term),
            "solver": solver_name,
            "wallclock_time": getattr(getattr(results, "solver", None), "wallclock_time", None),
            "gap": getattr(getattr(results, "solver", None), "gap", None),
            "status": getattr(getattr(results, "solver", None), "status", None),
        }

        if has_incumbent(model):
            out = extract_results(model, meta)
            return jsonify({"ok": True, "solver": solver_name, "result": out, "diagnostics": diag})

        return jsonify({
            "ok": False,
            "error": f"{MAX_SOLVE_SECONDS} sn içinde uygulanabilir çözüm bulunamadı veya yüklenemedi. Durum: {term}",
            "diagnostics": diag
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


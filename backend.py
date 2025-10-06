# -*- coding: utf-8 -*-
import os
import json
from openai import AzureOpenAI
import traceback
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, send_from_directory
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals, Objective, minimize,
    Constraint, Any as PyAny, value, SolverFactory
)
from pathlib import Path

# Dataset yolu (env ile özelleştirilebilir)
DATASET_PATH = Path(os.environ.get("DATASET_PATH", str(Path(__file__).parent / "dataset.json")))

MAX_SOLVE_SECONDS = 26  # işlemciye verdiğimiz süre
app = Flask(__name__, static_url_path="", static_folder=".")

# ---------------- Azure OpenAI Ayarları ----------------
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "d0167637046c4443badc4920cc612abb")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-fnss.openai.azure.com")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

try:
    aoai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception:
    aoai_client = None
# -------------------------------------------------------


# ------------------------------
# Yardımcılar
# ------------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_dataset_from_file() -> Dict[str, Any]:
    """
    dataset.json dosyasını okur ve dict döner.
    Zaman dilimi (period) yok; hız tabanlı süre modeli var.
    """
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # minimum alan kontrolü
        required = [
            "cities", "main_depot",
            "vehicle_types", "vehicle_count",
            "distances", "packages", "minutil_penalty"
        ]
        for k in required:
            if k not in data:
                raise KeyError(f"dataset.json alan eksik: {k}")
        # hız alanı kontrolü
        for tname, tinfo in data["vehicle_types"].items():
            if "hiz_kmh" not in tinfo:
                raise KeyError(f"vehicle_types['{tname}'] için 'hiz_kmh' eksik")
        # paket alanları
        for p in data["packages"]:
            for kk in ["id", "baslangic", "hedef", "agirlik", "ready_hour", "deadline_hour", "ceza"]:
                if kk not in p:
                    raise KeyError(f"Paket {p.get('id','?')} için alan eksik: {kk}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset.json bulunamadı: {DATASET_PATH}")
    except Exception as e:
        raise RuntimeError(f"dataset.json okunamadı: {e}")


def _normalize_initial_locations(
    vehicles: List[str],
    cities: List[str],
    main_depot: str,
    payload: Dict[str, Any]
) -> Dict[str, str]:
    """
    vehicle_initial_locations aşağıdaki iki formdan birini alabilir:
      A) { "Küçük_1": "Ankara", "Orta_1": "İzmir", ... }  # doğrudan araç adına göre
      B) { "Küçük": "Ankara", "Orta": "İzmir" }           # tipe göre, o tipteki tüm araçlara uygular
    Verilmezse tüm araçlar main_depot'ta başlatılır.
    Şehir doğrulaması yapar; hatalı şehirler sessizce main_depot'a döner.
    """
    raw = payload.get("vehicle_initial_locations", {}) or {}
    init = {v: main_depot for v in vehicles}

    if isinstance(raw, dict):
        # araç adına göre eşleşenler
        for k, city in raw.items():
            if k in vehicles and city in cities:
                init[k] = city
        # tip adına göre eşleşenler
        for k, city in raw.items():
            if k not in vehicles and city in cities:
                prefix = f"{k}_"
                for v in vehicles:
                    if v.startswith(prefix):
                        init[v] = city

    # son güvenlik
    for v in vehicles:
        if init.get(v) not in cities:
            init[v] = main_depot

    return init


def pick_solver():
    """
    Önce Pyomo APPsi-HiGHS (highspy) denenir.
    Ardından klasik arayüzler (highs/cbc/glpk/cplex).
    Geriye (solver_adı, solver_nesnesi, is_appsi_bool) döner.
    """
    # 1) APPsi-HiGHS (Python wrapper)
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as AppsiHighs
        s = AppsiHighs()
        try:
            s.config.time_limit = MAX_SOLVE_SECONDS
        except Exception:
            pass
        return "appsi_highs", s, True
    except Exception:
        pass

    # 2) Klasik arayüzler
    for cand in ["highs", "cbc", "glpk", "cplex"]:
        try:
            s = SolverFactory(cand)
            if s is not None and s.available():
                try:
                    if cand == "highs":
                        s.options["time_limit"] = MAX_SOLVE_SECONDS
                    elif cand == "cbc":
                        s.options["seconds"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "glpk":
                        s.options["tmlim"] = int(MAX_SOLVE_SECONDS)
                    elif cand == "cplex":
                        s.options["timelimit"] = MAX_SOLVE_SECONDS
                        s.options["mipgap"] = 0.05
                        s.options["threads"] = 2
                except Exception:
                    pass
                return cand, s, False
        except Exception:
            continue

    return None, None, False


# ------------------------------
# Model Kurulumu (hız/süre tabanlı, dönem yok)
# ------------------------------
def build_model(payload: Dict[str, Any]) -> Tuple[ConcreteModel, Dict[str, Any]]:
    """
    Beklenen payload anahtarları:
      cities, main_depot, vehicle_types, vehicle_count,
      distances, packages, minutil_penalty, (opsiyonel) vehicle_initial_locations

    vehicle_types örn:
      "Küçük": {"kapasite":200,"maliyet_km":2.5,"sabit_maliyet":150,"min_doluluk":0.4,"hiz_kmh":70}
    packages örn (saat cinsinden):
      {"id":"P1","baslangic":"Ankara","hedef":"İzmir","agirlik":120,"ready_hour":0,"deadline_hour":36,"ceza":120}
    """
    # --- Girdiler
    cities: List[str] = payload["cities"]
    main_depot: str = payload["main_depot"]

    vehicle_types: Dict[str, Dict[str, Any]] = payload["vehicle_types"]
    vehicle_count: Dict[str, int] = payload["vehicle_count"]

    # Araç isimleri: Tip_index
    vehicles: List[str] = [f"{vt}_{i}" for vt, cnt in vehicle_count.items() for i in range(1, int(cnt) + 1)]

    # Başlangıç konumları
    init_loc = _normalize_initial_locations(vehicles, cities, main_depot, payload)

    # Mesafeler (simetrik tamamla)
    distances: Dict[Tuple[str, str], float] = {}
    for i, j, d in payload["distances"]:
        distances[(i, j)] = float(d)
        distances[(j, i)] = float(d)
    for c in cities:
        distances[(c, c)] = 0.0

    # Paketler
    packages_input: List[Dict[str, Any]] = payload["packages"]
    packages = {}
    for rec in packages_input:
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

    # --- Pyomo Modeli
    model = ConcreteModel()

    # Kümeler
    model.Cities = Set(initialize=cities)
    model.Vehicles = Set(initialize=vehicles)
    model.Packages = Set(initialize=list(packages.keys()))

    def vtype(v):  # "Küçük_1" -> "Küçük"
        return v.rsplit("_", 1)[0]

    # Parametreler
    model.Distance = Param(model.Cities, model.Cities, initialize=lambda m, i, j: distances[(i, j)])
    model.VehicleCapacity = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["kapasite"])
    model.TransportCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["maliyet_km"])
    model.FixedCost = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["sabit_maliyet"])
    model.MinUtilization = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["min_doluluk"])
    model.Speed = Param(model.Vehicles, initialize=lambda m, v: vehicle_types[vtype(v)]["hiz_kmh"])

    model.PackageWeight = Param(model.Packages, initialize=lambda m, p: packages[p]["agirlik"])
    model.PackageOrigin = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["baslangic"])
    model.PackageDest = Param(model.Packages, within=PyAny, initialize=lambda m, p: packages[p]["hedef"])
    model.PackageReady = Param(model.Packages, initialize=lambda m, p: packages[p]["ready_hour"])
    model.PackageDeadline = Param(model.Packages, initialize=lambda m, p: packages[p]["deadline_hour"])
    model.LatePenalty = Param(model.Packages, initialize=lambda m, p: packages[p]["ceza_maliyeti"])

    # Kenar/süre ön-hesabı (v,i,j) -> saat
    travel_time = {}
    for v in vehicles:
        spd = float(vehicle_types[vtype(v)]["hiz_kmh"])
        for i in cities:
            for j in cities:
                dkm = distances[(i, j)]
                travel_time[(v, i, j)] = (dkm / spd) if spd > 0 else 1e6

    # Değişkenler
    # x[v,i,j]: araç v i->j kenarını kullanır mı
    model.x = Var(model.Vehicles, model.Cities, model.Cities, domain=Binary)
    # y[p,v,i,j]: paket p, araç v üzerinde i->j kenarını kullanır mı
    model.y = Var(model.Packages, model.Vehicles, model.Cities, model.Cities, domain=Binary)
    # z[v]: araç kullanıldı mı (en az bir kenar)
    model.z = Var(model.Vehicles, domain=Binary)
    # gecikme (saat)
    model.lateness = Var(model.Packages, domain=NonNegativeReals)
    # min. doluluk açığı (kg) – araç başına (yalnız ana depodan çıkışlarda)
    model.minutil_shortfall = Var(model.Vehicles, domain=NonNegativeReals)

    # Amaç
    def objective_rule(m):
        transport = sum(
            m.TransportCost[v] * m.Distance[i, j] * m.x[v, i, j]
            for v in m.Vehicles for i in m.Cities for j in m.Cities if i != j
        )
        fixed = sum(m.FixedCost[v] * m.z[v] for v in m.Vehicles)
        # Varış zamanı = ready + toplam yol süresi; gecikme = max(0, arrival - deadline) -> m.lateness
        late = sum(m.LatePenalty[p] * m.lateness[p] for p in m.Packages)
        minutil = MINUTIL_PENALTY * sum(m.minutil_shortfall[v] for v in m.Vehicles)
        return transport + fixed + late + minutil

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # ------------------------------
    # KISITLAR
    # ------------------------------

    # 0) z[v] >= x[v,i,j] (kullanıldı bayrağı)
    def used_vehicle_flag_rule(m, v, i, j):
        if i == j:
            return Constraint.Skip
        return m.z[v] >= m.x[v, i, j]
    model.used_vehicle_flag = Constraint(model.Vehicles, model.Cities, model.Cities, rule=used_vehicle_flag_rule)

    # 1) Paket origin'den tam 1 çıkış
    def package_origin_rule(m, p):
        o = m.PackageOrigin[p]
        return sum(m.y[p, v, o, j] for v in m.Vehicles for j in m.Cities if j != o) == 1
    model.package_origin_constraint = Constraint(model.Packages, rule=package_origin_rule)

    # 2) Paket hedefe tam 1 varış
    def package_destination_rule(m, p):
        d = m.PackageDest[p]
        return sum(m.y[p, v, i, d] for v in m.Vehicles for i in m.Cities if i != d) == 1
    model.package_destination_constraint = Constraint(model.Packages, rule=package_destination_rule)

    # 3) Ana depodan en az bir kez geçiş (origin/dest depo değilse)
    def main_depot_rule(m, p):
        o, d = m.PackageOrigin[p], m.PackageDest[p]
        if o == main_depot or d == main_depot:
            return Constraint.Skip
        through = sum(m.y[p, v, i, main_depot] for v in m.Vehicles for i in m.Cities if i != main_depot) \
                + sum(m.y[p, v, main_depot, j] for v in m.Vehicles for j in m.Cities if j != main_depot)
        return through >= 1
    model.main_depot_constraint = Constraint(model.Packages, rule=main_depot_rule)

    # 4) Paket ancak araç kenarı kullanıyorsa taşınır (y <= x)
    def y_le_x_rule(m, p, v, i, j):
        if i == j:
            return Constraint.Skip
        return m.y[p, v, i, j] <= m.x[v, i, j]
    model.package_vehicle_link = Constraint(model.Packages, model.Vehicles, model.Cities, model.Cities, rule=y_le_x_rule)

    # 5) Kapasite (her kenarda)
    def capacity_rule(m, v, i, j):
        if i == j:
            return Constraint.Skip
        return sum(m.PackageWeight[p] * m.y[p, v, i, j] for p in m.Packages) <= m.VehicleCapacity[v]
    model.capacity_constraint = Constraint(model.Vehicles, model.Cities, model.Cities, rule=capacity_rule)

    # 6) SOFT min. doluluk (sadece ana depodan çıkan kenarlar, araç başına)
    def min_utilization_soft_rule(m, v):
        departures = sum(m.x[v, main_depot, j] for j in m.Cities if j != main_depot)  # 0..N
        loaded = sum(m.PackageWeight[p] * m.y[p, v, main_depot, j] for p in m.Packages for j in m.Cities if j != main_depot)
        target = m.MinUtilization[v] * m.VehicleCapacity[v] * departures
        return loaded + m.minutil_shortfall[v] >= target
    model.min_utilization_soft = Constraint(model.Vehicles, rule=min_utilization_soft_rule)

    # 7) Paket akış korunumu (ara şehirler)
    def flow_conservation_rule(m, p, k):
        o, d = m.PackageOrigin[p], m.PackageDest[p]
        if k == o or k == d:
            return Constraint.Skip
        inflow = sum(m.y[p, v, i, k] for v in m.Vehicles for i in m.Cities if i != k)
        outflow = sum(m.y[p, v, k, j] for v in m.Vehicles for j in m.Cities if j != k)
        return inflow == outflow
    model.flow_conservation = Constraint(model.Packages, model.Cities, rule=flow_conservation_rule)

    # 8) Araç rotası tutarlılığı (başlangıç şehrinden çıkış serbest, diğer şehirlerde giriş=çıkış)
    def vehicle_flow_rule(m, v, n):
        init = init_loc[v]
        outn = sum(m.x[v, n, j] for j in m.Cities if j != n)
        inn = sum(m.x[v, i, n] for i in m.Cities if i != n)
        if n == init:
            # init için: çıkış >= giriş (başlayabilir, gerekirse döngü de kurabilir)
            return outn >= inn
        else:
            # diğer şehirlerde giriş = çıkış (kopuk kenarları azaltır)
            return outn == inn
    model.vehicle_route_balance = Constraint(model.Vehicles, model.Cities, rule=vehicle_flow_rule)

    # 9) Paket varış zamanı ve gecikme tanımı
    #    arrival = ready + sum(y[p,v,i,j] * time[v,i,j])
    #    lateness >= arrival - deadline
    def lateness_rule(m, p):
        total_time = sum(m.y[p, v, i, j] * travel_time[(v, i, j)] for v in m.Vehicles for i in m.Cities for j in m.Cities if i != j)
        arrival = m.PackageReady[p] + total_time
        deadline = m.PackageDeadline[p]
        return m.lateness[p] >= arrival - deadline
    model.lateness_calc = Constraint(model.Packages, rule=lateness_rule)

    # meta
    meta = {
        "cities": cities,
        "vehicles": vehicles,
        "packages": packages,
        "distances": distances,
        "vehicle_types": vehicle_types,
        "main_depot": main_depot,
        "MINUTIL_PENALTY": MINUTIL_PENALTY,
        "initial_locations": init_loc,
        "travel_time": travel_time  # (v,i,j) saat
    }
    return model, meta


# ------------------------------
# Sonuçları çıkarma (UI için)
# ------------------------------
def extract_results(model: ConcreteModel, meta: Dict[str, Any]) -> Dict[str, Any]:
    cities = meta["cities"]
    vehicles = meta["vehicles"]
    packages = meta["packages"]
    distances = meta["distances"]
    travel_time = meta["travel_time"]
    MINUTIL_PENALTY = meta["MINUTIL_PENALTY"]

    results = {}
    total_obj = float(value(model.obj))
    results["objective"] = total_obj

    # Maliyet dağılımı
    transport_cost = 0.0
    for v in vehicles:
        for i in cities:
            for j in cities:
                if i != j and value(model.x[v, i, j]) > 0.5:
                    transport_cost += float(value(model.TransportCost[v])) * float(value(model.Distance[i, j]))

    fixed_cost = 0.0
    for v in vehicles:
        if value(model.z[v]) > 0.5:
            fixed_cost += float(value(model.FixedCost[v]))

    penalty_cost = sum(float(value(model.LatePenalty[p])) * float(value(model.lateness[p])) for p in model.Packages)
    minutil_pen = MINUTIL_PENALTY * sum(float(value(model.minutil_shortfall[v])) for v in vehicles)

    results["cost_breakdown"] = {
        "transport": transport_cost,
        "fixed": fixed_cost,
        "lateness": penalty_cost,
        "min_util_gap": float(minutil_pen),
    }

    # Araç rotaları
    vehicle_routes = []
    for v in sorted(vehicles):
        entries = []
        for i in cities:
            for j in cities:
                if i != j and value(model.x[v, i, j]) > 0.5:
                    moved = []
                    totw = 0.0
                    for p in model.Packages:
                        if value(model.y[p, v, i, j]) > 0.5:
                            moved.append(p)
                            totw += float(value(model.PackageWeight[p]))
                    entries.append({
                        "from": i,
                        "to": j,
                        "km": float(distances[(i, j)]),
                        "travel_hours": float(travel_time[(v, i, j)]),
                        "packages": moved,
                        "load_kg": totw,
                        "utilization_pct": (100.0 * totw / float(value(model.VehicleCapacity[v]))) if totw > 0 else 0.0
                    })
        if entries:
            vehicle_routes.append({"vehicle": v, "capacity": float(value(model.VehicleCapacity[v])), "legs": entries})
    results["vehicle_routes"] = vehicle_routes

    # Paketler
    package_summaries = []
    for p in sorted(packages.keys()):
        o = packages[p]["baslangic"]
        d = packages[p]["hedef"]
        r = packages[p]["ready_hour"]
        deadline = packages[p]["deadline_hour"]

        # toplam yol süresi
        tt = 0.0
        segs = []
        for v in model.Vehicles:
            for i in cities:
                for j in cities:
                    if i != j and value(model.y[p, v, i, j]) > 0.5:
                        tt += travel_time[(v, i, j)]
                        segs.append({"from": i, "to": j, "vehicle": v, "travel_hours": float(travel_time[(v, i, j)])})

        arrival = r + tt
        late = float(value(model.lateness[p]))
        passed_main = False
        main_depot = meta["main_depot"]
        for v in model.Vehicles:
            if passed_main:
                break
            for i in cities:
                if i == main_depot:
                    for j in cities:
                        if j != i and value(model.y[p, v, i, j]) > 0.5:
                            passed_main = True; break
                if passed_main:
                    break
                if i != main_depot:
                    if value(model.y[p, v, i, main_depot]) > 0.5:
                        passed_main = True; break

        summary = {
            "id": p,
            "origin": o,
            "dest": d,
            "weight": packages[p]["agirlik"],
            "ready_hour": r,
            "deadline_by": deadline,
            "total_travel_hours": round(tt, 3),
            "delivered_at": round(arrival, 3),
            "on_time": (arrival <= deadline + 1e-9),
            "passed_main_depot": passed_main,
            "route": segs,
            "lateness_hours": late,
            "lateness_penalty": float(value(model.LatePenalty[p])) * late
        }
        package_summaries.append(summary)

    results["packages"] = package_summaries
    return results


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/dataset", methods=["GET", "PUT", "POST"])
def dataset_endpoint():
    """
    GET  -> dataset.json'u oku ve döndür
    PUT  -> gelen JSON'u doğrula ve dataset.json'a yaz
    POST -> bazı ortamlarda PUT engelli olabilir; POST'u PUT gibi kullan
    """
    try:
        if request.method == "GET":
            if not DATASET_PATH.exists():
                return jsonify({"ok": False, "error": "dataset.json bulunamadı"}), 404
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"ok": True, "dataset": data})

        # PUT veya POST -> kaydet
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400

        # Basit şema kontrolü (periods kaldırıldı)
        required_keys = [
            "cities","main_depot",
            "vehicle_types","vehicle_count",
            "distances","packages","minutil_penalty"
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return jsonify({"ok": False, "error": f"Eksik alanlar: {', '.join(missing)}"}), 400

        # Araç tipi hız alanı kontrolü
        for tname, tinfo in payload["vehicle_types"].items():
            if "hiz_kmh" not in tinfo:
                return jsonify({"ok": False, "error": f"'{tname}' tipi için 'hiz_kmh' alanı zorunlu."}), 400

        # Paket alanları kontrolü
        for p in payload["packages"]:
            for kk in ["id", "baslangic", "hedef", "agirlik", "ready_hour", "deadline_hour", "ceza"]:
                if kk not in p:
                    return jsonify({"ok": False, "error": f"Paket {p.get('id','?')} için alan eksik: {kk}"}), 400

        # Dosyaya yaz (atomic)
        tmp_path = DATASET_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(DATASET_PATH)

        return jsonify({"ok": True, "message": "dataset.json güncellendi"})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {e}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Önizleme/forma girilen JSON parametrelerini 'context' olarak alır,
    'messages' listesindeki sohbet geçmişiyle birlikte Azure OpenAI'ye gönderir.
    """
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
Her şehir bir depodur ama bir tane ana depo vardır. Zaman modelinde dönem yok; süreler
= mesafe / hız. Geç teslim mümkündür ve ceza ile modellenir. Alakasız şeyleri yanıtlama.
Adın: VRP Assist 2.0

Model için kullanılan JSON parametreleri:
{model_context}

Kurallar:
- Sayısal/lojistik sorularda net hesap yap ve kısaca açıkla.
- Tutarsızlık görürsen hangi alanın düzelmesi gerektiğini söyle.
- Gereksiz ayrıntıya girme; anlaşılır ve kısa yanıt üret.
        """.strip()

        completion = aoai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                *user_messages
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = completion.choices[0].message.content
        return jsonify({"ok": True, "answer": answer})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.errorhandler(500)
def handle_500(e):
    return jsonify({"ok": False, "error": "Internal Server Error"}), 500


# ==== /solve ====
@app.route("/solve", methods=["POST"])
def solve():
    try:
        # Önce gelen payload'ı dene; yoksa dataset.json'u yükle
        data = request.get_json(silent=True) or {}
        if not data:
            data = load_dataset_from_file()

        model, meta = build_model(data)

        solver_name, solver, is_appsi = pick_solver()
        if solver is None:
            return jsonify({"ok": False, "error": "Uygun MILP çözücüsü bulunamadı."}), 400

        # --- ÇÖZ ---
        if is_appsi:
            results = solver.solve(model)
            term = getattr(results, "termination_condition", None)
        else:
            try:
                results = solver.solve(model, tee=False, load_solutions=True)
            except TypeError:
                results = solver.solve(model, load_solutions=True)
            term = None
            if hasattr(results, "solver") and hasattr(results.solver, "termination_condition"):
                term = results.solver.termination_condition
            else:
                term = getattr(results, "termination_condition", None)

        # --- İnkümbent var mı? ---
        def has_incumbent(m):
            try:
                # x değişkenlerinden herhangi birinin değeri gelmiş mi?
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
            "error": f"{MAX_SOLVE_SECONDS} sn içinde uygulanabilir çözüm bulunamadı veya çözüm yüklenemedi. Durum: {term}",
            "diagnostics": diag
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}", "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    # Lokal test için:
    app.run(host="0.0.0.0", port=5000, debug=True)

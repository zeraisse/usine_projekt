"""
Générateur de données synthétiques pour le Détecteur Intelligent de Causes Racines.

Produit deux fichiers :
    1. training_data..csv     -> dataset supervisé (log_block, root_cause_index)
    2. demo_production.log   -> flux de logs réaliste pour la démo Web

Usage :
    python log_generator.py
"""

import csv
import random
from datetime import datetime, timedelta

import pandas as pd

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None


# ---------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE
# ---------------------------------------------------------------------------

MODULES = ["SYS", "LOG", "API", "DB", "WORKER", "AUTH", "UI"]
USERS = ["j.doe@domain.com", "s.smith@domain.com", "admin@domain.com",
         "system@domain.com", "m.curie@domain.com"]

# ---------------------------------------------------------------------------
# 2. LOGS "NORMAUX" (bruit de fond)
# ---------------------------------------------------------------------------
# Chaque entrée = (module, level, event, message_template)
NORMAL_LOGS = [
    ("SYS",    "INFO",    "Cache Refresh",            "General system operation executed successfully."),
    ("SYS",    "INFO",    "Health Check",             "All endpoints reported 200 OK. Latency: {lat}ms."),
    ("SYS",    "INFO",    "Database Sync",            "Synchronized {n} records with the primary PostgreSQL instance."),
    ("SYS",    "INFO",    "User Login",               "User '{user}' authenticated via OAuth2 provider."),
    ("SYS",    "INFO",    "File Upload",              "Resource 'data_{rid}.ext' stored successfully in Azure Blob Storage."),
    ("LOG",    "INFO",    "Service Started",          "Worker service 'QueueListener' initialized on .NET 9.0.1."),
    ("LOG",    "INFO",    "Background Job Completed", "General system operation executed successfully."),
    ("LOG",    "INFO",    "Middleware Pipeline",      "General system operation executed successfully."),
    ("LOG",    "INFO",    "Token Refresh",            "General system operation executed successfully."),
    ("LOG",    "INFO",    "API Request Success",      "General system operation executed successfully."),
    ("SYS",    "WARNING", "Session Timeout Warning",  "General system operation executed successfully."),
    ("SYS",    "WARNING", "Disk Space Low",           "General system operation executed successfully."),
    ("SYS",    "WARNING", "Deprecated API Call",      "Usage of obsolete method 'GetLegacyData()'. Switch to V2."),
    ("LOG",    "WARNING", "Memory Pressure",          "LOH (Large Object Heap) fragmentation detected. GC triggered."),
    ("LOG",    "WARNING", "Unusual Traffic Pattern",  "General system operation executed successfully."),
]


# ---------------------------------------------------------------------------
# 3. SCÉNARIOS D'ANOMALIES (CASCADES)
# ---------------------------------------------------------------------------
# Chaque cascade = liste ORDONNÉE de logs.
# L'ÉLÉMENT 0 = la CAUSE RACINE. Les éléments suivants = conséquences.
# Pour ajouter une cascade : append un nouveau dict ici. Aucun autre changement requis.

ANOMALY_SCENARIOS = [
    {
        "name": "DB_TIMEOUT_CASCADE",
        "steps": [
            ("DB",     "ERROR",   "Database Timeout",        "Connection to SQL Server '10.0.0.5' timed out after 30000ms."),
            ("API",    "ERROR",   "API Latency Spike",       "Endpoint /orders/list response time > 15000ms."),
            ("WORKER", "ERROR",   "Job Queue Backpressure",  "Hangfire queue 'default' size exceeded threshold (2500)."),
            ("UI",     "ERROR",   "UI Crash",                "Frontend React app reported unhandled promise rejection."),
        ],
    },
    {
        "name": "AUTH_FAILURE_CASCADE",
        "steps": [
            ("AUTH",   "ERROR",   "JWT Signing Key Missing", "KeyVault secret 'jwt-prod' not found. Auth middleware aborted."),
            ("API",    "ERROR",   "Unauthorized Access",     "IP 185.x.x.x blocked after 5 failed JWT attempts."),
            ("SYS",    "ERROR",   "Dependency Injection Failure", "IAuthService could not be resolved at runtime."),
            ("UI",     "ERROR",   "Session Lost",            "All active sessions invalidated. Users redirected to /login."),
        ],
    },
    {
        "name": "DISK_FULL_CASCADE",
        "steps": [
            ("SYS",    "ERROR",   "Disk Full",               "Volume D:\\ has 0 bytes free. Write operations blocked."),
            ("LOG",    "ERROR",   "Log Writer Failed",       "Serilog file sink threw IOException: no space left on device."),
            ("DB",     "ERROR",   "SQL Constraint Violation","EF Core: Cannot insert duplicate key in table 'Orders'."),
            ("WORKER", "ERROR",   "Background Job Failed",   "Hangfire job 'NightlyReport' aborted after 3 retries."),
        ],
    },
    {
        "name": "MEMORY_LEAK_CASCADE",
        "steps": [
            ("SYS",    "ERROR",   "Out Of Memory",           "System.OutOfMemoryException raised by GC. Worker process restarting."),
            ("SYS",    "ERROR",   "Stack Overflow",          "General system operation executed successfully."),
            ("API",    "ERROR",   "Null Reference",          "System.NullReferenceException at App.Controller.Process(Guid id)."),
            ("UI",     "ERROR",   "Gateway 502",             "Upstream API returned 502 Bad Gateway."),
        ],
    },
    {
        "name": "REDIS_OUTAGE_CASCADE",
        "steps": [
            ("SYS",    "ERROR",   "Connection Timeout",      "Failed to connect to Redis at 10.0.0.12:6379 after 5000ms."),
            ("API",    "WARNING", "Cache Miss Storm",        "Cache miss ratio reached 98% on /catalog endpoint."),
            ("DB",     "WARNING", "Slow Query Detected",     "SELECT * FROM Transactions took {ms}ms. Optimization required."),
            ("SYS",    "ERROR",   "Deadlock Detected",       "General system operation executed successfully."),
        ],
    },
]


# ---------------------------------------------------------------------------
# 4. UTILITAIRES DE FORMATAGE
# ---------------------------------------------------------------------------

def _fill_template(message: str) -> str:
    """Remplit les placeholders {lat}, {n}, {user}, {rid}, {ms} avec des valeurs aléatoires."""
    return message.format(
        lat=random.randint(50, 8000),
        n=random.randint(10, 500),
        user=random.choice(USERS),
        rid=random.randint(100, 999),
        ms=random.randint(2000, 8000),
    )


def _format_line(ts: datetime, module: str, level: str, event: str, message: str) -> str:
    """Formate une ligne au format : DD-MM-YYYY-HH:MM | MODULE | LEVEL | Event | Message"""
    ts_str = ts.strftime("%d-%m-%Y-%H:%M")
    return f"{ts_str} | {module} | {level} | {event} | {_fill_template(message)}"


def _make_normal_line(ts: datetime) -> str:
    module, level, event, msg = random.choice(NORMAL_LOGS)
    return _format_line(ts, module, level, event, msg)


def _make_cascade_lines(start_ts: datetime, scenario: dict) -> list[str]:
    """Sérialise une cascade : les étapes se suivent dans la minute / quelques minutes."""
    lines = []
    ts = start_ts
    for module, level, event, msg in scenario["steps"]:
        lines.append(_format_line(ts, module, level, event, msg))
        ts += timedelta(seconds=random.randint(5, 90))
    return lines


# ---------------------------------------------------------------------------
# 5. GÉNÉRATION DU TRAINING SET
# ---------------------------------------------------------------------------

def generate_training_block(block_size: int = 15, anomaly_proba: float = 0.5) -> tuple[str, int]:
    """
    Construit UN bloc de logs et calcule root_cause_index.

    Mathématique du root_cause_index :
        - On crée d'abord 'block_size' lignes de bruit.
        - Si une cascade est injectée, on choisit un index 'insert_pos' aléatoire
          dans [0, block_size - len(cascade)] : c'est la position EXACTE de la
          CAUSE RACINE dans le bloc final (la 1ère étape de la cascade).
        - Les étapes suivantes de la cascade REMPLACENT les lignes de bruit
          aux positions insert_pos+1, insert_pos+2, ... (on ne shift pas, sinon
          l'index ne correspondrait plus). On garde donc un bloc de taille fixe.
        - root_cause_index = insert_pos  (l'index 0-based de la cause racine).
        - Si pas de cascade injectée -> root_cause_index = -1.
    """
    base_ts = datetime(2026, 5, 11, 8, 0) + timedelta(minutes=random.randint(0, 600))

    # 1) Bloc de bruit (taille fixe = block_size)
    lines = []
    for i in range(block_size):
        lines.append(_make_normal_line(base_ts + timedelta(seconds=i * random.randint(20, 80))))

    # 2) Décide d'injecter ou non une cascade
    if random.random() < anomaly_proba:
        scenario = random.choice(ANOMALY_SCENARIOS)
        cascade = _make_cascade_lines(base_ts + timedelta(minutes=random.randint(1, 5)),
                                      scenario)

        # On s'assure que la cascade entière tient dans le bloc.
        max_start = block_size - len(cascade)
        if max_start < 0:
            # Cascade plus longue que le bloc : on tronque la cascade.
            cascade = cascade[:block_size]
            max_start = 0

        insert_pos = random.randint(0, max_start)

        # Remplacement en place -> l'index de la CAUSE RACINE = insert_pos
        for offset, cline in enumerate(cascade):
            lines[insert_pos + offset] = cline

        root_cause_index = insert_pos
    else:
        root_cause_index = -1

    return "\n".join(lines), root_cause_index


def generate_training_csv(path: str, n_blocks: int = 5000) -> None:
    rows = [generate_training_block() for _ in range(n_blocks)]
    df = pd.DataFrame(rows, columns=["log_block", "root_cause_index"])
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
    print(f"[OK] training_data..csv -> {len(df)} blocs ({(df['root_cause_index']>=0).sum()} avec anomalie)")


# ---------------------------------------------------------------------------
# 6. GÉNÉRATION DU FLUX DE DÉMO
# ---------------------------------------------------------------------------

def generate_demo_log(path: str, n_lines: int = 500, n_cascades: int = 4) -> None:
    """Produit une chronologie continue avec quelques cascades cachées."""
    start = datetime(2026, 5, 11, 8, 0)
    lines: list[str] = []
    ts = start

    # 1) Choisir les positions des cascades (réparties dans le flux)
    cascade_positions = sorted(random.sample(range(20, n_lines - 20), n_cascades))
    cascade_iter = iter(cascade_positions)
    next_cascade_at = next(cascade_iter, None)

    i = 0
    while i < n_lines:
        if next_cascade_at is not None and i >= next_cascade_at:
            scenario = random.choice(ANOMALY_SCENARIOS)
            for module, level, event, msg in scenario["steps"]:
                lines.append(_format_line(ts, module, level, event, msg))
                ts += timedelta(seconds=random.randint(15, 120))
                i += 1
            next_cascade_at = next(cascade_iter, None)
        else:
            lines.append(_make_normal_line(ts))
            ts += timedelta(seconds=random.randint(30, 180))
            i += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:n_lines]))
    print(f"[OK] demo_production.log -> {min(len(lines), n_lines)} lignes, {n_cascades} cascades cachées")


# ---------------------------------------------------------------------------
# 7. ENTRÉE PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    generate_training_csv("training_data..csv", n_blocks=5000)
    generate_demo_log("demo_production.log", n_lines=500, n_cascades=4)
    print("Done.")

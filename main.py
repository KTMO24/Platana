#!/usr/bin/env python3
# =============================================================================
#  COMPLETE MERGED APPLICATION (NO API KEY)
#    - Platana System (Flask + Selenium + Tor/WireGuard + analytics)
#    - Synthetic Embedded File System (FUSE)
#    - 3-to-1 "fold and flip" vector storage
#    - Data interpolation & polynomial regression (numpy, scikit-learn)
#    - CLI modes: fuse, gui, simulate, api, platana
#    - Minimal "Gemini" references for advanced tasks
#    - No API key checks
#
# (c) 2025, KTMO Travis Michael O’Dell - All Rights Reserved
# =============================================================================

import os
import sys
import math
import stat
import json
import zlib
import random
import secrets
import logging
import argparse
import subprocess
import time
import threading
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# Attempt external libraries
try:
    from fuse import FUSE, Operations, FuseOSError
    FUSE_AVAILABLE = True
except ImportError:
    FUSE_AVAILABLE = False

try:
    from PySide6 import QtWidgets, QtCore
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

try:
    from flask import Flask, jsonify, request, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from requests_html import HTMLSession
    from stem import Signal
    from stem.control import Controller
except ImportError:
    pass

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# For data interpolation & polynomial regression:
try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# SHARED DATA & CONFIGURATION
# ---------------------------------------------------------------------------------
CONFIG = {}
learning_data = []
learning_data_lock = threading.Lock()

vector_storage = {}
vector_storage_lock = threading.Lock()

DATA = []       # For sparse/collected data
MODEL = None    # For polynomial regression
SCALER = None   # For scaling

# =============================================================================
# 1) CONFIG LOADING
# =============================================================================

def load_config(config_path="config.json"):
    global CONFIG
    defaults = {
        "mountpoint": "/mnt/embedded_fs",
        "wireguard_interface": "wg_client",
        "use_wireguard": True,
        "tor_port": 9051,
        "use_tor": False,
        "default_clients": 100,
        "max_workers": 20,
        "neural_data_file": "neural_db.json",
        "vector_data_file": "vector_db.json",
        "sparse_data_file": "sparse_db.json",
        "model_file": "model.json",
        "use_sparse_data": False,
        "api_endpoint": "YOUR_API_ENDPOINT",
        "api_delay": 1,
        "initial_sample_size": 1000,
        "sparse_sample_rate": 0.2,
        "training_epochs": 50,
        "learning_rate": 0.001,
        "retrain_threshold": 0.1
    }
    try:
        with open(config_path, "r") as f:
            fileconf = json.load(f)
        defaults.update(fileconf)
        CONFIG = defaults
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults.")
        CONFIG = defaults
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in config: {e} -> using defaults.")
        CONFIG = defaults

# =============================================================================
# 2) ENTROPY & DELAYS WITH SHANNON ENTROPY
# =============================================================================

def compute_shannon_entropy(data: bytes) -> float:
    """
    Compute the Shannon entropy of the given bytes.
    Entropy = -sum(p * log2(p)) for each unique byte.
    """
    if not data:
        return 0.0
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def generate_entropy_offset():
    """
    Generate an entropy offset using a random 16-bit value.
    The process:
      1. Generate 16-bit random value.
      2. Convert it to bytes.
      3. Compute Shannon entropy.
      4. Invert and offset the entropy value.
      5. Feed the result to logistic_compress.
    """
    random_val = secrets.randbits(16)
    random_bytes = random_val.to_bytes(2, 'big')
    entropy = compute_shannon_entropy(random_bytes)
    # Invert the entropy (avoid division by zero)
    inv_entropy = 1.0 / (entropy + 1e-10)
    # Apply an offset (for example, add 0.5)
    offset_value = inv_entropy + 0.5
    return logistic_compress(offset_value)

def simulate_delay(min_delay=0.1, max_delay=0.5):
    time.sleep(random.uniform(min_delay, max_delay))

def random_human_delay():
    time.sleep(random.uniform(2,8))

# =============================================================================
# 3) WIREGUARD & TOR
# =============================================================================

def setup_wireguard(interface_name="wg0"):
    if not CONFIG.get("use_wireguard", True):
        return
    try:
        subprocess.run(f"sudo wg-quick up {interface_name}", shell=True, check=True)
        logger.info(f"[WireGuard] {interface_name} up.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[WireGuard] Could not bring up {interface_name}. Error: {e}")

def get_new_tor_ip():
    if not CONFIG.get("use_tor", False):
        return
    port = CONFIG.get("tor_port", 9051)
    try:
        with Controller.from_port(port=port) as c:
            c.authenticate()
            c.signal(Signal.NEWNYM)
        logger.info("[Tor] Requested new circuit.")
    except Exception as e:
        logger.error(f"[Tor] Error: {e}")

# =============================================================================
# 4) SELENIUM SIMULATION
# =============================================================================

def record_learning(client_id, data):
    with learning_data_lock:
        entry = {"client_id": client_id, "timestamp": time.time(), **data}
        learning_data.append(entry)
    save_learning_data()

def compress_entropy_for_selenium():
    return generate_entropy_offset()

def setup_wireguard_i2p(client_id):
    base_iface = CONFIG["wireguard_interface"]
    iface = f"{base_iface}{client_id}"
    logger.info(f"[Client {client_id}] Bringing up {iface} ...")
    if CONFIG.get("use_wireguard", True):
        try:
            subprocess.run(f"sudo wg-quick up {iface}", shell=True, check=True)
            logger.info(f"[Client {client_id}] {iface} up.")
        except Exception as e:
            logger.error(f"[Client {client_id}] Could not up {iface}: {e}")
    new_ip = f"10.0.0.{random.randint(2,254)}"
    logger.info(f"[Client {client_id}] IP => {new_ip}")
    return new_ip, iface

def create_selenium_session(proxy_ip):
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not installed. Cannot create session.")
        return None
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"--proxy-server={proxy_ip}")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def random_navigation(driver):
    base_urls = [
        "https://httpbin.org/ip",
        "https://example.com",
        "https://news.ycombinator.com",
        "https://www.reddit.com",
        "https://www.wikipedia.org"
    ]
    nav_count = random.randint(1, 3)
    history = []
    for _ in range(nav_count):
        url = random.choice(base_urls)
        driver.get(url)
        simulate_delay(1, 3)
        history.append({"url": url, "title": driver.title})
    return history

def selenium_client(client_id):
    start_time = time.time()
    try:
        offset = compress_entropy_for_selenium()
        logger.info(f"[Client {client_id}] Waiting {offset:.2f}s offset.")
        time.sleep(offset)

        if CONFIG.get("use_tor", False):
            get_new_tor_ip()
            time.sleep(0.5)

        i2p_ip, iface = setup_wireguard_i2p(client_id)
        proxy_ip = f"http://{i2p_ip}:8080" if not CONFIG.get("use_tor", False) else f"socks5://127.0.0.1:{CONFIG['tor_port']}"

        driver = create_selenium_session(proxy_ip)
        if not driver:
            record_learning(client_id, {"status": "error", "error": "Selenium not available", "elapsed": 0})
            return

        base_url = "https://httpbin.org/ip"
        driver.get(base_url)
        nav_history = [{"url": base_url, "title": driver.title}]
        simulate_delay(1, 3)

        subnav = random_navigation(driver)
        nav_history.extend(subnav)

        driver.get(base_url)
        nav_history.append({"url": base_url, "title": driver.title})
        simulate_delay(1, 2)

        snippet = driver.page_source[:200]
        driver.quit()

        if CONFIG.get("use_wireguard", True):
            try:
                subprocess.run(f"sudo wg-quick down {iface}", shell=True, check=True)
                logger.info(f"[Client {client_id}] Brought down {iface}")
            except Exception as e:
                logger.error(f"[Client {client_id}] Error tearing down {iface}: {e}")

        record_learning(client_id, {
            "status": "success",
            "launch_delay": offset,
            "elapsed": time.time() - start_time,
            "navigation_history": nav_history,
            "page_snippet": snippet
        })
    except Exception as ex:
        logger.error(f"[Client {client_id}] Error: {ex}")
        record_learning(client_id, {"status": "error", "error": str(ex), "elapsed": time.time() - start_time})

def simulate_selenium_clients(client_count, max_workers):
    logger.info(f"Simulating {client_count} clients with concurrency={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(selenium_client, i): i for i in range(client_count)}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"[Client {cid}] Thread error: {e}")

    with learning_data_lock:
        total = len(learning_data)
        successes = len([d for d in learning_data if d.get("status") == "success"])
        errors = total - successes
        avg_elapsed = (sum(d.get("elapsed", 0) for d in learning_data) / total) if total else 0.0
        logger.info("=== Selenium Simulation Done ===")
        logger.info(f"Total={total}, Successes={successes}, Errors={errors}, Avg Elapsed={avg_elapsed:.2f}")
        save_learning_data()

# =============================================================================
# 5) FLASK “MEGA” APP (PLATANA)
# =============================================================================

mega_app = Flask("MegaSystemApp")

@mega_app.route("/status", methods=["GET"])
def get_status():
    with vector_storage_lock:
        active_vectors = len(vector_storage)
    with learning_data_lock:
        entries = len(learning_data)
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "active_vectors": active_vectors,
        "learning_entries": entries
    })

@mega_app.route("/learning/record", methods=["POST"])
def record_learning_data_endpoint():
    if not request.is_json:
        return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    with learning_data_lock:
        entry = {"timestamp": time.time(), "data": data}
        learning_data.append(entry)
        idx = len(learning_data) - 1
    save_learning_data()
    return jsonify({"status": "success", "message": "Learning data recorded", "entry_id": idx})

@mega_app.route("/learning/analyze", methods=["GET"])
def analyze_learning():
    with learning_data_lock:
        if not learning_data:
            return jsonify({"error": "No learning data"}), 404
        total = len(learning_data)
        successes = len([d for d in learning_data if d['data'].get('status') == "success"])
        analysis = {
            "total_entries": total,
            "successful_entries": successes,
            "success_rate": (successes / total) if total > 0 else 0,
            "latest_timestamp": learning_data[-1]["timestamp"]
        }
    return jsonify({"status": "success", "analysis": analysis})

@mega_app.route("/vectors/store", methods=["POST"])
def store_vector_endpoint():
    if not request.is_json:
        return jsonify({"error": "Missing JSON"}), 400
    data = request.get_json()
    if "vector" not in data:
        return jsonify({"error": "No 'vector' field"}), 400
    vector_id = data.get("vector_id", f"vec_{secrets.token_hex(8)}")
    with vector_storage_lock:
        vector_storage[vector_id] = {
            "data": data["vector"],
            "timestamp": time.time(),
            "metadata": data.get("metadata", {})
        }
    save_vector_data()
    return jsonify({"status": "success", "vector_id": vector_id})

@mega_app.route("/vectors/<vector_id>", methods=["GET"])
def get_vector_endpoint(vector_id):
    with vector_storage_lock:
        if vector_id not in vector_storage:
            return jsonify({"error": "not found"}), 404
        vec = vector_storage[vector_id]
    return jsonify({"status": "success", "vector": vec})

@mega_app.route("/vectors/list", methods=["GET"])
def list_vectors_endpoint():
    with vector_storage_lock:
        listing = {vid: {"timestamp": v["timestamp"], "metadata": v["metadata"]}
                   for vid, v in vector_storage.items()}
    return jsonify({"status": "success", "vectors": listing})

@mega_app.route("/simulation/start", methods=["POST"])
def start_simulation_endpoint():
    if not request.is_json:
        return jsonify({"error": "No JSON"}), 400
    data = request.get_json()
    ccount = data.get("client_count", CONFIG.get("default_clients", 100))
    mworkers = data.get("max_workers", CONFIG.get("max_workers", 20))
    def run_sim():
        logger.info(f"Simulation starting with {ccount} clients.")
        simulate_selenium_clients(ccount, mworkers)
        logger.info("Simulation done.")
    thr = threading.Thread(target=run_sim)
    thr.daemon = True
    thr.start()
    return jsonify({"status": "success", "message": f"Simulation started with {ccount} clients"})

def fold_and_flip_3to1(vec):
    folded = []
    for i in range(0, len(vec), 3):
        chunk = vec[i:i+3]
        if len(chunk) < 3:
            chunk = list(chunk) + [0] * (3 - len(chunk))
        val = (chunk[0] - chunk[1] + chunk[2]) / 3.0
        folded.append(val)
    return folded

@mega_app.route("/fold_vector", methods=["POST"])
def fold_vector_api():
    if not request.is_json:
        return jsonify({"error": "No JSON"}), 400
    data = request.get_json()
    vector = data.get("vector", [])
    vector_id = data.get("vector_id", f"vec_{secrets.token_hex(4)}")
    folded = fold_and_flip_3to1(vector)
    with vector_storage_lock:
        vector_storage[vector_id] = {
            "data": folded,
            "timestamp": time.time(),
            "metadata": {"folded": True}
        }
    save_vector_data()
    return jsonify({"status": "success", "vector_id": vector_id, "folded_length": len(folded)})

@mega_app.route("/get_folded_vector/<vid>", methods=["GET"])
def get_folded_vector_endpoint(vid):
    with vector_storage_lock:
        if vid not in vector_storage:
            return jsonify({"error": "not found"}), 404
        return jsonify({"status": "success", "data": vector_storage[vid]})

@mega_app.route("/list_folded_vectors", methods=["GET"])
def list_folded_vectors_endpoint():
    with vector_storage_lock:
        arr = list(vector_storage.keys())
    return jsonify({"status": "success", "vector_ids": arr})

@mega_app.route("/gemini/<platform>/generate_command", methods=["POST"])
def gemini_generate_command_endpoint(platform):
    payload = request.get_json() or {}
    function_name = payload.get("function", "")
    data = payload.get("data", {})
    cmd = f"[Gemini Stub] For {platform}, do {function_name} with {json.dumps(data)}"
    logger.info(f"[Gemini] {cmd}")
    return jsonify({"platform": platform, "command": cmd, "status": "success"})

def run_mega_app():
    setup_wireguard(CONFIG.get("wireguard_interface", "wg0"))
    load_learning_data()
    load_vector_data()
    load_sparse_data()
    load_model()
    logger.info("Starting Mega Flask app on port 5000")
    mega_app.run(host="0.0.0.0", port=5000, debug=True)

# =============================================================================
# 6) SYNTHETIC EMBEDDED FS (FUSE)
# =============================================================================

class SyntheticEmbeddedFS(Operations):
    def __init__(self):
        self.mount_time = time.time()
        self.files = {
            "/": {
                "st_mode": (stat.S_IFDIR | 0o755),
                "st_nlink": 2,
                "st_size": 0,
                "virtual_size": 0,
                "st_ctime": self.mount_time,
                "st_mtime": self.mount_time,
                "st_atime": self.mount_time
            }
        }
        self.data = {}
        self.lock = threading.Lock()
        self.PLACEHOLDER_VIRTUAL_SIZE = 10 * 1024 * 1024

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data)

    def decompress(self, comp: bytes) -> bytes:
        return zlib.decompress(comp)

    def fake_expand(self, raw: bytes, vs: int) -> bytes:
        if len(raw) >= vs:
            return raw[:vs]
        rep = (vs // len(raw)) + 1
        return (raw * rep)[:vs]

    def getattr(self, path, fh=None):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
            m = self.files[path]
            return {
                "st_mode": m["st_mode"],
                "st_nlink": m["st_nlink"],
                "st_size": m["virtual_size"],
                "st_ctime": m["st_ctime"],
                "st_mtime": m["st_mtime"],
                "st_atime": m["st_atime"]
            }

    def readdir(self, path, fh):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
            entries = [".", ".."]
            for k in self.files:
                if k != "/" and k.startswith("/"):
                    entries.append(k[1:])
            for e in entries:
                yield e

    def create(self, path, mode, fi=None):
        with self.lock:
            now = time.time()
            self.files[path] = {
                "st_mode": (stat.S_IFREG | mode),
                "st_nlink": 1,
                "st_size": 11,
                "virtual_size": self.PLACEHOLDER_VIRTUAL_SIZE,
                "st_ctime": now,
                "st_mtime": now,
                "st_atime": now
            }
            placeholder = b"PLACEHOLDER"
            self.data[path] = self.compress(placeholder)
        return 0

    def open(self, path, flags):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
        return 0

    def read(self, path, size, offset, fh):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
            comp = self.data.get(path)
            if not comp:
                return b""
            raw = self.decompress(comp)
            expanded = self.fake_expand(raw, self.files[path]["virtual_size"])
            return expanded[offset:offset+size]

    def write(self, path, data, offset, fh):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
            comp = self.data.get(path)
            curr = self.decompress(comp) if comp else b""
            newd = bytearray(curr)
            if offset + len(data) > len(newd):
                newd.extend(b"X" * ((offset + len(data)) - len(newd)))
            newd[offset:offset+len(data)] = data
            cnew = self.compress(bytes(newd))
            self.data[path] = cnew
            self.files[path]["st_size"] = len(newd)
            if len(newd) > self.files[path]["virtual_size"]:
                self.files[path]["virtual_size"] = len(newd)
            return len(data)

    def truncate(self, path, length, fh=None):
        with self.lock:
            if path not in self.files:
                raise FuseOSError(os.errno.ENOENT)
            comp = self.data.get(path)
            c = self.decompress(comp) if comp else b""
            c = c[:length].ljust(length, b"X")
            self.data[path] = self.compress(c)
            self.files[path]["st_size"] = length
            self.files[path]["virtual_size"] = length

    def unlink(self, path):
        with self.lock:
            if path in self.files:
                del self.files[path]
            if path in self.data:
                del self.data[path]

def run_fuse_mode(mountpoint):
    if not FUSE_AVAILABLE:
        print("fusepy not installed. Exiting FUSE mode.")
        sys.exit(1)
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    fs = SyntheticEmbeddedFS()
    FUSE(fs, mountpoint, foreground=True)

# =============================================================================
# 7) GUI MODE
# =============================================================================

def run_gui_mode():
    if not PYSIDE_AVAILABLE:
        print("PySide6 not installed, no GUI mode.")
        return
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication([])
    w = QtWidgets.QWidget()
    w.setWindowTitle("Embedded FS / Platana Settings")
    lay = QtWidgets.QVBoxLayout()
    lay.addWidget(QtWidgets.QLabel("Mountpoint (for FUSE):"))
    mount_edit = QtWidgets.QLineEdit(CONFIG.get("mountpoint", "/mnt/embedded_fs"))
    lay.addWidget(mount_edit)
    auto_check = QtWidgets.QCheckBox("Auto-mount on start?")
    lay.addWidget(auto_check)
    btn = QtWidgets.QPushButton("Apply")
    lay.addWidget(btn)

    def do_apply():
        mp = mount_edit.text()
        am = auto_check.isChecked()
        logger.info(f"[GUI] mount={mp}, auto={am}")
        w.close()

    btn.clicked.connect(do_apply)
    w.setLayout(lay)
    w.show()
    app.exec()

# =============================================================================
# 8) SIMULATION MODE
# =============================================================================

def run_simulation_mode():
    def waveforms():
        start = time.time()
        freq = 1.0
        while True:
            t = time.time() - start
            sinv = math.sin(2 * math.pi * freq * t)
            saw = 2 * ((t * freq) - math.floor(t * freq)) - 1
            inv = -sinv
            off = (sinv - saw + inv) / 3
            logger.info(f"[Sim] sin={sinv:.2f}, saw={saw:.2f}, off={off:.2f}")
            time.sleep(0.2)
    thr = threading.Thread(target=waveforms, daemon=True)
    thr.start()
    while True:
        time.sleep(1)

# =============================================================================
# 9) LAUNCH SYNTHETIC FS API
# =============================================================================

def launch_synthetic_fs_api():
    if not FLASK_AVAILABLE:
        print("Flask not installed. Exiting FS-API mode.")
        return
    fs_app = Flask("FsAPI")

    @fs_app.route("/fs/files", methods=["GET"])
    def list_files_fs():
        return jsonify({"status": "success", "files": []})

    @fs_app.route("/fs/files", methods=["POST"])
    def create_file_fs():
        data = request.get_json() or {}
        path = data.get("path")
        if not path:
            return jsonify({"error": "Missing path"}), 400
        return jsonify({"status": "success", "path": path})

    fs_app.run(host="0.0.0.0", port=5000)

# =============================================================================
# 10) MACHINE LEARNING / SPARSE DATA & POLYNOMIAL REGRESSION
# =============================================================================

DATA = []
MODEL = None
SCALER = None

def load_learning_data_ml():
    global learning_data
    fname = CONFIG.get("neural_data_file", "neural_db.json")
    try:
        with open(fname, "r") as f:
            learning_data.clear()
            learning_data.extend(json.load(f))
        logger.info(f"Loaded learning data from {fname}")
    except FileNotFoundError:
        logger.info(f"No existing learning data file {fname}. Starting empty.")
    except Exception as e:
        logger.error(f"Error loading {fname}: {e}")

def save_learning_data_ml():
    global learning_data
    fname = CONFIG.get("neural_data_file", "neural_db.json")
    with learning_data_lock:
        try:
            with open(fname, "w") as f:
                json.dump(learning_data, f, indent=2)
            logger.info(f"Saved learning data to {fname}")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")

def load_vector_data():
    global vector_storage
    fname = CONFIG.get("vector_data_file", "vector_db.json")
    try:
        with open(fname, "r") as f:
            vector_storage.clear()
            vector_storage.update(json.load(f))
        logger.info(f"Loaded vector data from {fname}")
    except FileNotFoundError:
        logger.info(f"No vector data file {fname}, starting empty.")
    except Exception as e:
        logger.error(f"Error loading {fname}: {e}")

def save_vector_data():
    global vector_storage
    fname = CONFIG.get("vector_data_file", "vector_db.json")
    with vector_storage_lock:
        try:
            with open(fname, "w") as f:
                json.dump(vector_storage, f, indent=2)
            logger.info(f"Saved vector data to {fname}")
        except Exception as e:
            logger.error(f"Error saving vector data: {e}")

def load_sparse_data():
    global DATA
    fname = CONFIG.get("sparse_data_file", "sparse_db.json")
    try:
        with open(fname, 'r') as f:
            DATA = json.load(f)
        logger.info(f"Sparse data loaded from {fname}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading sparse data from {fname}: {e}")

def save_sparse_data():
    global DATA
    fname = CONFIG.get("sparse_data_file", "sparse_db.json")
    try:
        with open(fname, 'w') as f:
            json.dump(DATA, f, indent=2)
        logger.info(f"Saved sparse data to {fname}")
    except Exception as e:
        logger.error(f"Error saving sparse data: {e}")

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.001, epochs=50):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def _feature_map(self, X):
        X = np.array(X)
        features = []
        for d in range(self.degree+1):
            features.append(X**d)
        return np.concatenate(features, axis=1)

    def _loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def fit(self, X_train, y_train):
        X = self._feature_map(X_train)
        y = y_train.reshape(-1,1)
        self.weights = np.random.randn(X.shape[1],1)
        for e in range(self.epochs):
            y_pred = np.dot(X, self.weights)
            loss = self._loss(y_pred, y)
            grads = np.dot(X.T, (y_pred-y)) / len(X)
            self.weights -= self.learning_rate * grads
            if e % 10 == 0:
                logger.info(f"Epoch {e}/{self.epochs} loss={loss:.4f}")

    def predict(self, X_new):
        if self.weights is None:
            logger.error("No weights found; returning zeros.")
            return np.zeros((X_new.shape[0],1))
        Xf = self._feature_map(X_new)
        return np.dot(Xf, self.weights)

    def get_weights(self):
        return self.weights if self.weights is not None else np.array([])

    def set_weights(self, w):
        self.weights = w

def train_model(X_train, y_train):
    global MODEL
    MODEL = PolynomialRegression(degree=2,
                                 learning_rate=CONFIG.get("learning_rate",0.001),
                                 epochs=CONFIG.get("training_epochs",50))
    MODEL.fit(X_train, y_train)

def preprocess_data():
    if not SKLEARN_AVAILABLE or not DATA:
        logger.warning("Preprocessing not possible; missing scikit-learn or DATA.")
        return None, None
    data_points = [d["value"] for d in DATA if "value" in d]
    X = np.array(data_points).reshape(-1,1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    global SCALER
    SCALER = scaler
    return X_train, X_val

def sparse_sampling():
    global DATA, MODEL, SCALER
    if not MODEL or not DATA or not SKLEARN_AVAILABLE or SCALER is None:
        logger.warning("Sparse sampling not possible; missing model or data.")
        return []
    X = np.array([d["value"] for d in DATA]).reshape(-1,1)
    X_scaled = SCALER.transform(X)
    preds = MODEL.predict(X_scaled)
    y_true = X_scaled
    errors = np.abs(preds - y_true).flatten()
    rate = CONFIG.get("sparse_sample_rate",0.2)
    sample_count = int(rate*len(DATA))
    idxs = np.argsort(errors)[-sample_count:]
    selected_ids = [DATA[i]["point_id"] for i in idxs]
    logger.info(f"Sparse sampling: selected {len(selected_ids)} data points.")
    return selected_ids

def check_retrain():
    global DATA, MODEL, SCALER
    if not MODEL or not DATA or SCALER is None:
        return True
    X = np.array([d["value"] for d in DATA]).reshape(-1,1)
    X_scaled = SCALER.transform(X)
    preds = MODEL.predict(X_scaled)
    loss = np.mean((preds - X_scaled)**2)
    logger.info(f"Model loss: {loss:.4f}, threshold: {CONFIG.get('retrain_threshold',0.1)}")
    return loss > CONFIG.get("retrain_threshold",0.1)

def update_sparse_data():
    sampled_ids = sparse_sampling()
    if not sampled_ids:
        logger.info("No data sampled; skipping update.")
        return
    logger.info(f"Updating {len(sampled_ids)} data points from API.")
    updated = []
    for dp in DATA:
        if dp["point_id"] in sampled_ids:
            dp = fetch_data_from_api(dp["point_id"])
        updated.append(dp)
    DATA[:] = updated
    save_sparse_data()
    if check_retrain():
        X_train, X_val = preprocess_data()
        if X_train is not None and X_val is not None:
            train_model(X_train, X_val)
            save_model()

def fetch_data_from_api(data_point_id):
    time.sleep(CONFIG.get("api_delay",1))
    logger.info(f"Fetching data point {data_point_id} from API...")
    return {"point_id": data_point_id, "value": random.random(), "timestamp": time.time()}

def collect_initial_data():
    global DATA
    size = CONFIG.get("initial_sample_size",1000)
    logger.info(f"Collecting {size} initial data points from API.")
    DATA = [fetch_data_from_api(i) for i in range(size)]
    logger.info(f"Collected {len(DATA)} data points.")
    save_sparse_data()

def load_model():
    global MODEL
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not installed, skipping model load.")
        return
    fname = CONFIG.get("model_file","model.json")
    try:
        with open(fname,"r") as f:
            info = json.load(f)
        MODEL = PolynomialRegression(
            degree=info.get("degree",2),
            learning_rate=info.get("learning_rate",0.001),
            epochs=info.get("epochs",50)
        )
        if "weights" in info:
            w = np.array(info["weights"])
            MODEL.set_weights(w)
        logger.info(f"Loaded model from {fname}")
    except FileNotFoundError:
        logger.info(f"No model file {fname}, skipping.")
    except Exception as e:
        logger.error(f"Error loading model from {fname}: {e}")

def save_model():
    global MODEL
    if MODEL is None:
        logger.info("No model to save.")
        return
    fname = CONFIG.get("model_file","model.json")
    try:
        data = {
            "weights": MODEL.get_weights().tolist(),
            "degree": MODEL.degree,
            "learning_rate": MODEL.learning_rate,
            "epochs": MODEL.epochs
        }
        with open(fname,"w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved model to {fname}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

# =============================================================================
# 11) MAIN CLI & APPLICATION ENTRY
# =============================================================================

def run_fuse_mode(mountpoint):
    if not FUSE_AVAILABLE:
        print("fusepy not installed. Exiting FUSE mode.")
        sys.exit(1)
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    fs = SyntheticEmbeddedFS()
    FUSE(fs, mountpoint, foreground=True)

def run_gui_mode():
    if not PYSIDE_AVAILABLE:
        print("PySide6 not installed, no GUI mode.")
        return
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication([])
    w = QtWidgets.QWidget()
    w.setWindowTitle("Embedded FS / Platana Settings")
    lay = QtWidgets.QVBoxLayout()
    lay.addWidget(QtWidgets.QLabel("Mountpoint (for FUSE):"))
    mount_edit = QtWidgets.QLineEdit(CONFIG.get("mountpoint", "/mnt/embedded_fs"))
    lay.addWidget(mount_edit)
    auto_check = QtWidgets.QCheckBox("Auto-mount on start?")
    lay.addWidget(auto_check)
    btn = QtWidgets.QPushButton("Apply")
    lay.addWidget(btn)
    def do_apply():
        mp = mount_edit.text()
        am = auto_check.isChecked()
        logger.info(f"[GUI] mount={mp}, auto={am}")
        w.close()
    btn.clicked.connect(do_apply)
    w.setLayout(lay)
    w.show()
    app.exec()

def run_simulation_mode():
    def waveforms():
        start = time.time()
        freq = 1.0
        while True:
            t = time.time() - start
            sinv = math.sin(2 * math.pi * freq * t)
            saw = 2 * ((t * freq) - math.floor(t * freq)) - 1
            inv = -sinv
            off = (sinv - saw + inv) / 3
            logger.info(f"[Sim] sin={sinv:.2f}, saw={saw:.2f}, off={off:.2f}")
            time.sleep(0.2)
    thr = threading.Thread(target=waveforms, daemon=True)
    thr.start()
    while True:
        time.sleep(1)

def launch_synthetic_fs_api():
    if not FLASK_AVAILABLE:
        print("Flask not installed. Exiting FS-API mode.")
        return
    fs_app = Flask("FsAPI")
    @fs_app.route("/fs/files", methods=["GET"])
    def list_files_fs():
        return jsonify({"status": "success", "files": []})
    @fs_app.route("/fs/files", methods=["POST"])
    def create_file_fs():
        data = request.get_json() or {}
        path = data.get("path")
        if not path:
            return jsonify({"error": "Missing path"}), 400
        return jsonify({"status": "success", "path": path})
    fs_app.run(host="0.0.0.0", port=5000)

def run_mega_app():
    setup_wireguard(CONFIG.get("wireguard_interface", "wg0"))
    load_learning_data_ml()
    load_vector_data()
    load_sparse_data()
    load_model()
    logger.info("Starting Mega Flask app on port 5000")
    mega_app.run(host="0.0.0.0", port=5000, debug=True)

def main():
    parser = argparse.ArgumentParser(
        description="Complete Merged Application (No API Key) - Combined"
    )
    parser.add_argument("mode", choices=["fuse", "gui", "simulate", "api", "platana"],
                        help="Which mode to run")
    parser.add_argument("--mountpoint", default="/mnt/embedded_fs",
                        help="Mountpoint for FUSE mode")
    parser.add_argument("--config", default="config.json",
                        help="Path to config file (JSON)")
    args = parser.parse_args()
    load_config(args.config)
    if args.mode == "fuse":
        run_fuse_mode(args.mountpoint)
    elif args.mode == "gui":
        run_gui_mode()
    elif args.mode == "simulate":
        run_simulation_mode()
    elif args.mode == "api":
        launch_synthetic_fs_api()
    elif args.mode == "platana":
        run_mega_app()

if __name__ == "__main__":
    main()

"""Unbalanced three-phase LV distribution network with a real power flow.

The first version of this prototype modelled the feeder as a single bus. That
hides the constraint that actually binds on Indian LV networks: voltage. A
controller can "solve" reliability by pushing current down a 400 m service
line until the tail-end node sits at 0.86 pu, which is a worse outcome than
the outage it prevented, and no DISCOM would permit it.

This module builds a radial four-wire LV feeder, assigns single-phase
connections across phases a/b/c, and solves the unbalanced power flow by
backward-forward sweep. It also exposes the linearised voltage sensitivity
matrix dV/dP used as a constraint inside the dispatch optimiser, so the
optimiser respects voltage without needing a nonlinear solve in the loop.

Conventions
-----------
Per-unit on a 230 V line-to-neutral base. Loads are constant-power (PQ).
Statutory band taken as 0.94-1.06 pu, consistent with IS 12360 / CEA
distribution standards for LV supply.
"""
from __future__ import annotations

import numpy as np

V_BASE = 230.0            # line-to-neutral, volts
V_MIN_PU = 0.94
V_MAX_PU = 1.06

# Typical Indian LV aerial bundled conductor, ohms per km
LINE_R_OHM_KM = {"main": 0.32, "lateral": 0.64, "service": 1.38}
LINE_X_OHM_KM = {"main": 0.075, "lateral": 0.082, "service": 0.090}

# Distribution transformer: 315 kVA, 4.5% impedance, X/R = 3.
# Including this matters. With the DT secondary modelled as an ideal slack
# bus, battery injection at the transformer shows ZERO voltage benefit -
# because the LV line current is unchanged by where it is sourced from. The
# real benefit comes from relieving the drop across the transformer itself,
# which only appears once that impedance is represented.
DT_RATING_KVA = 315.0
DT_IMPEDANCE_PU = 0.045
DT_XR_RATIO = 3.0


class LVFeeder:
    """Radial, unbalanced, four-wire LV feeder fed from one DT."""

    def __init__(self, n_connections: int, rng: np.random.Generator,
                 n_laterals: int = 4, poles_per_lateral: int = 8):
        self.rng = rng
        self.n_conn = n_connections
        self.n_lat = n_laterals
        self.n_pole = poles_per_lateral

        # ---- topology -------------------------------------------------
        # node 0 is the DT secondary (slack). Then laterals, then poles.
        self.n_nodes = 1 + n_laterals * poles_per_lateral
        self.parent = np.zeros(self.n_nodes, dtype=int)
        self.seg_len_km = np.zeros(self.n_nodes)
        self.seg_type = ["main"] * self.n_nodes

        node = 1
        for l in range(n_laterals):
            prev = 0
            for p in range(poles_per_lateral):
                self.parent[node] = prev
                # first span off the transformer is heavier main cable
                self.seg_type[node] = "main" if p < 2 else "lateral"
                self.seg_len_km[node] = rng.uniform(0.030, 0.055)
                prev = node
                node += 1

        self.R = np.array([LINE_R_OHM_KM[t] for t in self.seg_type]) * self.seg_len_km
        self.X = np.array([LINE_X_OHM_KM[t] for t in self.seg_type]) * self.seg_len_km

        # ---- connection placement and phase assignment ----------------
        self.conn_node = rng.integers(1, self.n_nodes, size=n_connections)
        self.conn_phase = rng.integers(0, 3, size=n_connections)
        # deliberate, realistic phase imbalance: phase a is over-subscribed
        swap = rng.random(n_connections) < 0.12
        self.conn_phase[swap] = 0

        # service drop from pole to premises
        self.service_len_km = rng.uniform(0.012, 0.045, size=n_connections)
        self.service_R = LINE_R_OHM_KM["service"] * self.service_len_km
        self.service_X = LINE_X_OHM_KM["service"] * self.service_len_km

        self.phase_counts = np.bincount(self.conn_phase, minlength=3)

        z_base = (V_BASE ** 2) / (DT_RATING_KVA * 1000.0 / 3.0)
        z_tx = DT_IMPEDANCE_PU * z_base
        self.tx_R = z_tx / np.sqrt(1 + DT_XR_RATIO ** 2)
        self.tx_X = self.tx_R * DT_XR_RATIO

        # ordering for the sweep: children before parents
        self.order = np.argsort(-self._depth())

    # ------------------------------------------------------------------
    def _depth(self) -> np.ndarray:
        d = np.zeros(self.n_nodes, dtype=int)
        for i in range(1, self.n_nodes):
            d[i] = d[self.parent[i]] + 1
        return d

    # ------------------------------------------------------------------
    def node_phase_injections(self, conn_kw: np.ndarray,
                              pf: float = 0.92) -> np.ndarray:
        """Aggregate per-connection demand onto (node, phase) in kW and kvar."""
        P = np.zeros((self.n_nodes, 3))
        np.add.at(P, (self.conn_node, self.conn_phase), conn_kw)
        Q = P * np.tan(np.arccos(pf))
        return P, Q

    # ------------------------------------------------------------------
    def solve(self, conn_kw: np.ndarray, dt_injection_kw: float = 0.0,
              pf: float = 0.92, tol: float = 1e-6, max_iter: int = 40) -> dict:
        """Backward-forward sweep unbalanced power flow.

        dt_injection_kw is net generation at the DT bus (battery + PV), split
        equally across phases, which is how a three-phase inverter behaves.
        """
        P, Q = self.node_phase_injections(conn_kw, pf)

        V = np.ones((self.n_nodes, 3), dtype=complex)
        S = (P + 1j * Q) * 1000.0 / 3.0 / (V_BASE ** 2 / 3.0)  # not used directly

        Z = (self.R + 1j * self.X)[:, None]        # per node, same all phases
        Vb = V_BASE
        Sn = (P + 1j * Q) * 1000.0                 # VA per node per phase

        converged = False
        for it in range(max_iter):
            V_prev = V.copy()

            # --- backward sweep: accumulate currents toward the source ---
            I_inj = np.conj(Sn / (V * Vb + 1e-9))          # amps
            I_br = I_inj.copy()
            for i in self.order:
                if i == 0:
                    continue
                I_br[self.parent[i]] += I_br[i]

            # --- forward sweep: propagate voltage drops outward ----------
            # The DT secondary is NOT an ideal source. Net import through the
            # transformer (load minus local injection) drops its terminal
            # voltage, and relieving that drop is the battery's real
            # contribution to LV voltage support.
            I_inj = np.conj((dt_injection_kw / 3.0) * 1000.0 / (V[0] * Vb + 1e-9))
            I_tx = I_br[0] - I_inj
            V[0, :] = 1.0 - ((self.tx_R + 1j * self.tx_X) * I_tx) / Vb
            for i in range(1, self.n_nodes):
                V[i] = V[self.parent[i]] - (Z[i] * I_br[i]) / Vb

            if np.max(np.abs(V - V_prev)) < tol:
                converged = True
                break

        Vmag = np.abs(V)

        # voltage at the premises, including the service drop
        I_conn = np.conj((conn_kw * 1000.0 * (1 + 1j * np.tan(np.arccos(pf))))
                         / (Vmag[self.conn_node, self.conn_phase] * Vb + 1e-9))
        dV_service = (self.service_R + 1j * self.service_X) * I_conn / Vb
        V_conn = Vmag[self.conn_node, self.conn_phase] - np.abs(dV_service)

        losses_kw = float(np.sum(np.abs(I_br) ** 2 * self.R[:, None]) / 1000.0)

        return {
            "v_dt_pu": float(Vmag[0].mean()),
            "v_node_pu": Vmag,
            "v_conn_pu": V_conn,
            "v_min_pu": float(V_conn.min()),
            "v_min_node_pu": float(Vmag[1:].min()),
            "losses_kw": losses_kw,
            "iterations": it + 1,
            "converged": converged,
            "violations": int((V_conn < V_MIN_PU).sum()),
            "phase_imbalance_pct": float(
                100 * (Vmag[1:].max(axis=0) - Vmag[1:].min(axis=0)).max()),
        }

    # ------------------------------------------------------------------
    def voltage_sensitivity(self, conn_kw: np.ndarray,
                            pf: float = 0.92) -> dict:
        """Linearise dV_min/dP around an operating point.

        Returns the sensitivity of the worst premises voltage to (a) a
        kilowatt of aggregate load curtailment and (b) a kilowatt injected at
        the DT bus by the battery. The dispatch optimiser uses these two
        scalars to keep voltage inside the statutory band without solving a
        power flow inside the loop.
        """
        base = self.solve(conn_kw, 0.0, pf)
        v0 = base["v_min_pu"]

        d = max(conn_kw.sum() * 0.05, 1.0)
        cut = self.solve(conn_kw * (1.0 - d / max(conn_kw.sum(), 1e-6)), 0.0, pf)
        dv_dcurtail = (cut["v_min_pu"] - v0) / d

        inj = self.solve(conn_kw, d, pf)
        dv_dinject = (inj["v_min_pu"] - v0) / d

        return {
            "v_min_pu": v0,
            "dv_per_kw_curtailed": float(dv_dcurtail),
            "dv_per_kw_injected": float(dv_dinject),
            "losses_kw": base["losses_kw"],
        }

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (f"LV feeder: {self.n_nodes} nodes, {self.n_lat} laterals x "
                f"{self.n_pole} poles, {self.n_conn} connections, "
                f"phase split a/b/c = {self.phase_counts.tolist()}")

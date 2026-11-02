"""Enclave flow encoder — visualization & EDA (marimo).

Run:  marimo run visualize_enclave_mo.py    (read-only app)
Edit: marimo edit visualize_enclave_mo.py

Reads artifacts produced by the pipeline (make dbt / train / ml / lean):
  artifacts/enclave.duckdb, models_out/, lean/out/*.json.
Sections degrade gracefully when an artifact is missing.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Enclave Flow Encoder — Nitroba capture

        **Pipeline**: Cisco Joy `flow.jsonl` → dbt + DuckDB (features, stats,
        motifs) → PyTorch minimodels (MPS/CUDA) → **Lean 4 + SampCert**
        (formal motif definitions, proved ε-DP release, verified sampling).

        This notebook covers: exploratory data analysis, feature description,
        motif evidence, minimodel results, the Lean formulation with its proof
        ledger and DP/sampling outputs, and IP→country enrichment on a
        MapLibre map.
        """
    )
    return


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import duckdb
    import numpy as np
    import pandas as pd

    # bounded guard: a chart embedding >20k rows is a bug (pre-aggregate in
    # SQL/numpy instead) — fail loudly rather than emit a multi-hundred-MB cell
    alt.data_transformers.enable("default", max_rows=20_000)

    ROOT = Path(__file__).resolve().parent

    # Validated reference palette (dataviz method): categorical slots in fixed
    # order, blue sequential ramp, muted chart chrome.
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    INK_MUTED = "#898781"
    GRID = "#e1e0d9"

    def styled(chart, height=260):
        return (
            chart.properties(height=height)
            .configure_axis(gridColor=GRID, domainColor="#c3c2b7",
                            labelColor=INK_MUTED, titleColor="#52514e")
            .configure_view(stroke=None)
        )
    return ROOT, SEQ, PAL, alt, duckdb, json, math, np, pd, styled


@app.cell
def _(ROOT, duckdb, mo):
    _db = ROOT / "artifacts" / "enclave.duckdb"
    mo.stop(
        not _db.exists(),
        mo.md("⚠️ **`artifacts/enclave.duckdb` not found** — run `make dbt` "
              "(then `make train ml lean`) and reload."),
    )
    con = duckdb.connect(str(_db), read_only=True)
    return (con,)


@app.cell
def _(con, mo, pd):
    _row = con.execute(
        """
        select count(*), count(distinct sa), count(distinct da),
               min(time_start), max(time_start),
               sum(bytes_total), sum(case when is_zero_byte then 1 else 0 end)
        from feat_flows
        """
    ).fetchone()
    _n_flows, _n_src, _n_dst, _t0, _t1, _bytes_total, _n_zero = _row
    _start = pd.to_datetime(_t0, unit="s")
    _hours = (_t1 - _t0) / 3600
    mo.hstack(
        [
            mo.stat(f"{_n_flows:,}", label="flows"),
            mo.stat(f"{_n_src} → {_n_dst}", label="src → dst IPs"),
            mo.stat(f"{_hours:.1f} h", label=f"capture ({_start:%Y-%m-%d} UTC)"),
            mo.stat(f"{_bytes_total / 1e6:.1f} MB", label="total bytes"),
            mo.stat(f"{100 * _n_zero / _n_flows:.0f}%", label="zero-byte flows"),
        ],
        justify="space-around",
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Features

        Each Joy record is one bidirectional flow. The dbt layer derives:

        | feature | description |
        |---|---|
        | `duration_s` | `time_end − time_start`, clamped at 0 (Joy emits small negative durations on RST pairs — flagged `qf_negative_duration`) |
        | `bytes_out/in`, `pkts_out/in`, `bytes_per_s`, `byte_ratio_out_in` | volumes and rates; 34% of flows are unanswered (`is_unanswered`) |
        | `entropy` | payload entropy, capped at 8 bits/byte (14 invalid Joy values flagged) |
        | `service_class` | **public taxonomy** from protocol+port (17 classes — doubles as the DP cell grid) |
        | `src/dst_internal` | RFC1918 / link-local / multicast classification |
        | `byte_dist[256]` | payload byte histogram → classifier input |
        | `cv_gap` (per channel) | coefficient of variation of inter-flow gaps per (sa, da, dp) → beacon signal |
        """
    )
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _df = con.execute(
        """
        select service_class, n_flows, bytes_total, n_src, n_dst
        from mart_flow_stats where service_class <> 'TOTAL'
        order by n_flows desc
        """
    ).fetch_df()
    _chart = styled(
        alt.Chart(_df).mark_bar(color=PAL[0], cornerRadiusEnd=4)
        .encode(
            x=alt.X("n_flows:Q", title="flows"),
            y=alt.Y("service_class:N", sort="-x", title=None),
            tooltip=["service_class", "n_flows", "bytes_total", "n_src", "n_dst"],
        )
    )
    mo.vstack([mo.md("## Exploratory data analysis\n### Flows by service class"), _chart])
    return


@app.cell
def _(PAL, alt, con, mo, pd, styled):
    _df = con.execute(
        "select time_start, service_class from feat_flows"
    ).fetch_df()
    _df["t5"] = pd.to_datetime(_df.time_start, unit="s").dt.floor("5min")
    _top = _df.service_class.value_counts().head(3).index.tolist()
    _df["cls"] = _df.service_class.where(_df.service_class.isin(_top), "other")
    _agg = _df.groupby(["t5", "cls"], as_index=False).size()
    _order = _top + ["other"]
    _colors = PAL[:3] + ["#898781"]
    _chart = styled(
        alt.Chart(_agg).mark_line(interpolate="monotone", strokeWidth=2)
        .encode(
            x=alt.X("t5:T", title="time (UTC)"),
            y=alt.Y("size:Q", title="flows / 5 min"),
            color=alt.Color("cls:N", scale=alt.Scale(domain=_order, range=_colors),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("t5:T"), "cls:N", "size:Q"],
        )
    ).interactive()
    mo.vstack([mo.md("### Activity over the capture window"), _chart])
    return


@app.cell
def _(SEQ, alt, con, mo, np, pd, styled):
    # histograms are binned HERE (numpy), never by embedding per-flow rows in
    # the chart spec — 400k-row captures made that a 100s-of-MB cell output
    _df = con.execute(
        """
        select greatest(duration_s, 0.001) as duration_s,
               greatest(bytes_total, 1) as bytes_total, entropy
        from feat_flows
        """
    ).fetch_df()

    def _loghist(values, lo, bins=40):
        _v = values.dropna()
        _edges = np.logspace(np.log10(lo), np.log10(max(float(_v.max()), lo * 10)),
                             bins + 1)
        _n, _ = np.histogram(_v, bins=_edges)
        return pd.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "n": _n})

    def _binbar(hist_df, title, log=True):
        return styled(
            alt.Chart(hist_df).mark_bar(color=SEQ[4], cornerRadiusEnd=2)
            .encode(
                x=alt.X("lo:Q", bin="binned",
                        scale=alt.Scale(type="log") if log else alt.Scale(),
                        title=title),
                x2="hi:Q",
                y=alt.Y("n:Q", title="flows"),
                tooltip=[alt.Tooltip("n:Q", title="flows")],
            ), height=200,
        )

    _dur = _binbar(_loghist(_df.duration_s, 0.001), "duration (s, log)")
    _byt = _binbar(_loghist(_df.bytes_total, 1), "bytes (log)")
    _ec, _ee = np.histogram(_df.entropy.dropna(), bins=32, range=(0, 8))
    _ent = _binbar(pd.DataFrame({"lo": _ee[:-1], "hi": _ee[1:], "n": _ec}),
                   "payload entropy (bits/byte)", log=False)
    mo.vstack([
        mo.md("### Distributions (durations, volumes, entropy)"),
        mo.hstack([_dur, _byt, _ent], widths="equal"),
        mo.md("Negative durations are clamped; entropy is capped at 8 "
              "(see `wiki/data-quality.md`). Low-entropy mass = plaintext HTTP; "
              "the ~6–8 band = compressed/encrypted payloads."),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _talk = con.execute(
        """
        select sa, count(*) as flows, round(sum(bytes_total)/1e6, 2) as mb,
               count(distinct da) as dsts
        from feat_flows group by sa order by flows desc limit 8
        """
    ).fetch_df()
    _dirs = con.execute(
        """
        select case
                 when src_internal and not dst_internal then 'internal → external'
                 when src_internal and dst_internal then 'internal → internal'
                 when not src_internal and dst_internal then 'external → internal'
                 else 'external → external' end as direction,
               count(*) as flows
        from feat_flows group by 1 order by flows desc
        """
    ).fetch_df()
    _dchart = styled(
        alt.Chart(_dirs).mark_bar(color=PAL[0], cornerRadiusEnd=4)
        .encode(
            x=alt.X("flows:Q"), y=alt.Y("direction:N", sort="-x", title=None),
            tooltip=["direction", "flows"],
        ), height=160,
    )
    mo.vstack([
        mo.md("### Top talkers and flow directions"),
        mo.hstack([mo.ui.table(_talk, selection=None), _dchart], widths="equal"),
        mo.md("`192.168.15.4` dominates (the Nitroba suspect host, 66% of "
              "flows); `192.168.1.254` is the gateway whose RST reflection "
              "mimics a port scan — see the motif section."),
    ])
    return


@app.cell
def _(con, mo):
    _dns = con.execute(
        "select query_name, n_queries, n_clients from mart_dns order by n_queries desc limit 12"
    ).fetch_df()
    mo.vstack([
        mo.md("### Top DNS queries (Joy `dns` sub-records, 15% of flows)"),
        mo.ui.table(_dns, selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _bc = con.execute(
        """
        select sa || ' → ' || da || ':' || dp as channel, service_class,
               n_flows, period_s, cv_gap, beacon_score, is_beacon
        from mart_motif_beacon
        where cv_gap is not null and period_s > 0
        order by n_flows desc limit 60
        """
    ).fetch_df()
    _chart = styled(
        alt.Chart(_bc).mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("period_s:Q", scale=alt.Scale(type="log"), title="period (s, log)"),
            y=alt.Y("cv_gap:Q", scale=alt.Scale(type="log"),
                    title="gap CV (log; lower = more periodic)"),
            size=alt.Size("n_flows:Q", legend=None),
            color=alt.Color(
                "is_beacon:N",
                scale=alt.Scale(domain=[True, False], range=[PAL[0], "#898781"]),
                legend=alt.Legend(title="is_beacon (CV ≤ 0.6, n ≥ 8)", orient="top"),
            ),
            tooltip=["channel", "service_class", "n_flows",
                     alt.Tooltip("period_s:Q", format=".1f"),
                     alt.Tooltip("cv_gap:Q", format=".3f")],
        ), height=300,
    ).interactive()
    _tbl = con.execute(
        """
        select sa, da, dp, n_flows, round(period_s, 1) as period_s,
               round(cv_gap, 3) as cv
        from mart_motif_beacon where is_beacon order by cv limit 10
        """
    ).fetch_df()
    mo.vstack([
        mo.md(
            """
            ## Motifs

            ### Beacons — periodic re-contact of one (sa, da, dp) channel

            The cleanest beacon is the Google keepalive at **27.2 s, CV 0.029**;
            AIM and IPsec NAT-T keepalives follow; ~60 s webmail polling forms
            the HTTP-beacon cluster. High-CV channels (SSDP bursts, CDN
            fetches) are correctly rejected.
            """
        ),
        _chart,
        mo.ui.table(_tbl, selection=None),
    ])
    return


@app.cell
def _(con, mo):
    _fan = con.execute(
        """
        select sa, n_distinct_dst, n_distinct_dport, n_flows,
               round(zero_byte_frac, 2) as zero_byte_frac,
               is_fanout, is_portscan_shape
        from mart_motif_fanout order by n_distinct_dst desc limit 8
        """
    ).fetch_df()
    _ret = con.execute(
        """
        select sa, dst_a, dst_b, n_occurrences, round(mean_roundtrip_s, 2) as mean_roundtrip_s
        from mart_motif_return order by n_occurrences desc limit 8
        """
    ).fetch_df()
    mo.vstack([
        mo.md(
            """
            ### Fan-out / scan — and its false positive

            The gateway `192.168.1.254` tops the *port*-count ranking (373
            distinct ports), but `zero_byte_frac ≈ 1` exposes it as NAT/RST
            reflection, not probing — a deliberate negative exemplar
            (`is_portscan_shape`). Genuine fan-out is browser behavior.
            """
        ),
        mo.ui.table(_fan, selection=None),
        mo.md(
            """
            ### Return trips A → B → A

            The GPS return-trip motif, over destination sequences (gaps > 600 s
            break a sequence): hub-and-spoke browsing and gateway
            request/response chatter.
            """
        ),
        mo.ui.table(_ret, selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _fp = con.execute(
        """
        select fingerprint, n_flows, n_src_hosts, n_dst_hosts, c_version, n_cs,
               n_ext, top_scs, round(share, 3) as share, is_rare_stack,
               n_canonical_hs
        from mart_tls_fingerprints order by n_flows desc
        """
    ).fetch_df()
    _n_tls, _n_canon = con.execute(
        "select count(*), sum(is_canonical_hs::int) from feat_tls"
    ).fetchone()
    _fp["stack"] = _fp.fingerprint.where(~_fp.is_rare_stack, _fp.fingerprint + " ⚑")
    _chart = styled(
        alt.Chart(_fp).mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("n_flows:Q", title="flows"),
            y=alt.Y("stack:N", sort="-x", title=None),
            color=alt.Color(
                "is_rare_stack:N",
                scale=alt.Scale(domain=[False, True], range=[PAL[0], PAL[1]]),
                legend=alt.Legend(title="rare stack (≤ 3 flows)", orient="top"),
            ),
            tooltip=["fingerprint", "n_flows", "n_src_hosts", "n_dst_hosts",
                     "top_scs", "share", "n_canonical_hs"],
        ), height=230,
    )
    mo.vstack([
        mo.md(
            f"""
            ## TLS packet structure & fingerprinting

            {_n_tls} flows carry Joy's `tls` block (2008 capture; Joy version
            coding 3 = TLS 1.0, 2 = SSLv3). The **JA3-style client-stack
            fingerprint** hashes hello version + offered cipher suites +
            extension names (`feat_tls.fingerprint`). Stacks carried by ≤ 3
            flows are the **rare-stack motif**
            (`mart_tls_fingerprints.is_rare_stack`, spec `tls_rare_max_flows`)
            — an uncommon TLS implementation inside the enclave is a classic
            tooling lead. The dominant stack is the household's main browser.
            """
        ),
        mo.hstack([
            mo.stat(f"{_n_tls}", label="TLS flows"),
            mo.stat(f"{len(_fp)}", label="client stacks"),
            mo.stat(f"{int(_fp.is_rare_stack.sum())}", label="rare stacks"),
            mo.stat(f"{_n_canon}/{_n_tls}", label="canonical handshakes",
                    caption="client-hello first · CCS before appdata"),
        ], justify="space-around"),
        _chart,
        mo.ui.table(_fp.drop(columns="stack"), selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, pd, styled):
    _shapes = con.execute(
        """
        select shape_sig, n_flows, n_fingerprints, all_canonical,
               round(avg_records, 1) as avg_records,
               round(avg_appdata, 1) as avg_appdata,
               round(avg_tls_bytes_out) as avg_tls_bytes_out,
               round(avg_hs_setup_ms) as avg_hs_setup_ms
        from mart_tls_records order by n_flows desc
        """
    ).fetch_df()
    _examples = con.execute(
        """
        with ranked as (
            select shape_sig, flow_id, recs,
                   row_number() over (partition by shape_sig order by flow_id) as rn,
                   count(*) over (partition by shape_sig) as n_shape
            from feat_tls
        )
        select shape_sig, flow_id, recs, n_shape
        from ranked where rn = 1 order by n_shape desc limit 3
        """
    ).fetchall()
    _tp_name = {20: "change_cipher_spec", 21: "alert", 22: "handshake", 23: "appdata"}
    _rows = []
    for _sig, _fid, _recs, _n in _examples:
        for _i, _r in enumerate(list(_recs)[:14]):
            _rows.append({
                "example": f"flow {_fid} (shape shared by {_n} flows)",
                "record": _i + 1,
                "signed_b": int(_r["b"]) if _r["dir"] == ">" else -int(_r["b"]),
                "type": _tp_name.get(int(_r["tp"]), str(_r["tp"])),
                "direction": "client → server" if _r["dir"] == ">" else "server → client",
                "bytes": int(_r["b"]),
                "gap_ms": int(_r["ipt"]),
            })
    _walk = pd.DataFrame(_rows)
    _dom = ["handshake", "change_cipher_spec", "appdata", "alert"]
    _chart = styled(
        alt.Chart(_walk).mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("record:O", title="TLS record #"),
            y=alt.Y("signed_b:Q", title="record bytes (out ↑ / in ↓)"),
            color=alt.Color("type:N",
                            scale=alt.Scale(domain=_dom,
                                            range=[PAL[0], PAL[2], PAL[1], PAL[7]]),
                            legend=alt.Legend(title=None, orient="top")),
            row=alt.Row("example:N", title=None),
            tooltip=["record", "type", "direction", "bytes", "gap_ms"],
        ), height=100,
    )
    mo.vstack([
        mo.md(
            """
            ### Record shapes — the packet structure that survives encryption

            Joy's `srlt` records type, direction, size and inter-arrival time
            of every TLS record. Flows group into handshake **shapes** (the
            first-8-record signature, `mart_tls_records`); the walk below
            renders one example flow per top shape — ClientHello out,
            ServerHello/certificate in, ChangeCipherSpec both ways, then
            application data. The canonical predicate (client-hello first, no
            appdata before the key change in its direction) is formalized as
            `IsCanonicalHandshake` in `EnclaveFlows/Tls.lean` and re-decided
            per flow by `lake exe checks`; flow 852 is the single violation —
            its capture starts mid-stream with application data.
            """
        ),
        _chart,
        mo.ui.table(_shapes, selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _dom = con.execute(
        """
        select registered_domain, n_distinct_names, n_queries, n_flows,
               n_src_hosts, max_labels, max_qname_len, avg_digit_frac,
               n_service_discovery, n_reverse_lookup, is_subdomain_fan
        from mart_dns_domains order by n_distinct_names desc
        """
    ).fetch_df()
    _nq, _nn, _wf = con.execute(
        "select count(*), count(distinct qname), sum(is_wellformed::int) from feat_dns"
    ).fetchone()
    _chart = styled(
        alt.Chart(_dom.head(12)).mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("n_distinct_names:Q", title="distinct names queried"),
            y=alt.Y("registered_domain:N", sort="-x", title=None),
            color=alt.Color(
                "is_subdomain_fan:N",
                scale=alt.Scale(domain=[False, True], range=[PAL[0], PAL[1]]),
                legend=alt.Legend(title="subdomain fan (≥ 20 names)", orient="top"),
            ),
            tooltip=["registered_domain", "n_distinct_names", "n_queries",
                     "n_src_hosts", "max_labels", "n_service_discovery",
                     "n_reverse_lookup"],
        ), height=280,
    )
    mo.vstack([
        mo.md(
            f"""
            ## DNS name structure & domain population

            {_nq:,} query records over {_nn} distinct names (`feat_dns`); all
            {_wf:,} are RFC 1035 well-formed (labels 1–63 bytes, name ≤ 253 —
            the shape predicate proved and re-decided in Lean). Domains
            queried under many distinct names form the **subdomain-fan
            motif** (`mart_dns_domains.is_subdomain_fan`) — the shape of DNS
            tunneling, DGAs and CDNs. Here the top fan, `dlr.com`, is a
            **search-domain artifact**: DNS-SD queries like
            `_pop3._tcp…` re-suffixed with the local domain — exactly the
            kind of lead the motif is built to surface (and a labeled false
            positive, like the gateway in fan-out).
            """
        ),
        mo.hstack([
            mo.stat(f"{_nq:,}", label="query records"),
            mo.stat(f"{_nn}", label="distinct names"),
            mo.stat(f"{len(_dom)}", label="registered domains"),
            mo.stat(f"{int(_dom.is_subdomain_fan.sum())}", label="subdomain fans"),
            mo.stat(f"{_wf:,}/{_nq:,}", label="RFC-wellformed"),
        ], justify="space-around"),
        _chart,
        mo.ui.table(_dom, selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    # one row per (labels, kind), never per distinct qname — large captures
    # have tens of thousands of names and the chart embeds its dataframe
    _names = con.execute(
        """
        with per_name as (
            select qname, any_value(n_labels) as n_labels,
                   case when any_value(is_reverse_lookup) then 'reverse lookup'
                        when any_value(is_service_discovery) then 'service discovery'
                        else 'ordinary' end as kind
            from feat_dns group by qname
        )
        select n_labels, kind, count(*) as n
        from per_name group by 1, 2 order by 1
        """
    ).fetch_df()
    _kinds = ["ordinary", "reverse lookup", "service discovery"]
    _lab = styled(
        alt.Chart(_names).mark_bar()
        .encode(
            x=alt.X("n_labels:O", title="labels per name"),
            y=alt.Y("n:Q", title="distinct names"),
            color=alt.Color("kind:N",
                            scale=alt.Scale(domain=_kinds,
                                            range=[PAL[0], PAL[2], PAL[3]]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["n_labels:O", "kind:N", alt.Tooltip("n:Q", title="names")],
        ), height=200,
    )
    _scored = con.execute(
        """
        select distinct dns_top_qname as qname, round(dns_name_score, 2) as bits_per_char
        from ml_flow_scores where dns_name_score is not null
        order by bits_per_char desc limit 12
        """
    ).fetch_df()
    _parts = [
        mo.md(
            """
            ### Name shape — labels and the char-LM anomaly score

            Label-count distribution colored by name kind: the 9-label and
            34-label spikes are reverse lookups (`in-addr.arpa`, 34-label
            `ip6.arpa`), the underscore names are DNS-SD service discovery.
            The **DnsCharLM** minimodel (character-level language model over
            query names) assigns each name a bits/char score — high scores =
            names unlike the capture's vocabulary, the DGA-detector shape.
            Most unusual names by the model:
            """
        ),
        _lab,
    ]
    if len(_scored):
        _parts.append(mo.ui.table(_scored, selection=None))
    else:
        _parts.append(mo.md("⚠️ No DNS scores — run `make train ml`."))
    mo.vstack(_parts)
    return


@app.cell
def _(PAL, alt, con, mo, styled):
    _assoc = con.execute(
        """
        select feat_a || ' × ' || feat_b as pair,
               round(chi2, 1) as chi2, dof, n, round(cramers_v, 3) as cramers_v
        from mart_feature_assoc order by cramers_v desc
        """
    ).fetch_df()
    _chart = styled(
        alt.Chart(_assoc.head(12)).mark_bar(color=PAL[0], cornerRadiusEnd=4)
        .encode(
            x=alt.X("cramers_v:Q", title="Cramér's V (effect size)",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("pair:N", sort="-x", title=None),
            tooltip=["pair", "chi2", "dof", "n", "cramers_v"],
        ), height=280,
    )
    mo.vstack([
        mo.md(
            """
            ## Statistical association analysis (χ²)

            Pairwise chi-square over all categorical features **including the
            motif flags** (beacon channel, fan-out source, return source), on
            full contingency grids (zero cells included). Cramér's V
            normalizes χ² to [0, 1] for comparison across pairs. The χ²
            statistic itself is recomputed **exactly over ℚ in Lean**
            (`EnclaveFlows/Assoc.lean`) from the same cells — agreement below.
            """
        ),
        _chart,
        mo.ui.table(_assoc, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo, styled):
    _cells = con.execute(
        """
        select val_a as beacon, val_b as service, observed,
               round(expected, 1) as expected, round(std_residual, 2) as std_residual
        from mart_assoc_cells
        where feat_a = 'beacon_channel' and feat_b = 'service_class'
        """
    ).fetch_df()
    _lim = float(_cells.std_residual.abs().max())
    _chart = styled(
        alt.Chart(_cells).mark_rect(stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("service:N", title=None),
            y=alt.Y("beacon:N", title=None),
            color=alt.Color(
                "std_residual:Q",
                scale=alt.Scale(domain=[-_lim, 0, _lim],
                                range=["#2a78d6", "#f0efec", "#e34948"]),
                legend=alt.Legend(title="std. residual (O−E)/√E"),
            ),
            tooltip=["beacon", "service", "observed", "expected", "std_residual"],
        ), height=140,
    )
    mo.vstack([
        mo.md(
            """
            ### Motif × feature co-occurrence: beacon channels by service

            Standardized residuals (diverging: red = more than independence
            predicts, blue = fewer). Beacon-flagged flows concentrate in the
            keepalive services (chat, ipsec_natt, http polling) and are
            depleted in one-shot services — the motif is not independent of
            the service mix.
            """
        ),
        _chart,
    ])
    return


@app.cell
def _(SEQ, alt, con, mo, styled):
    _tr = con.execute(
        """
        select cur, nxt, n, round(p_next, 3) as p_next, round(lift, 2) as lift
        from mart_seq_transitions where n >= 5
        """
    ).fetch_df()
    _chart = styled(
        alt.Chart(_tr).mark_rect(stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("nxt:N", title="next service"),
            y=alt.Y("cur:N", title="current service"),
            color=alt.Color("p_next:Q",
                            scale=alt.Scale(range=SEQ, type="sqrt"),
                            legend=alt.Legend(title="P(next | current)")),
            tooltip=["cur", "nxt", "n", "p_next", "lift"],
        ), height=260,
    )
    _chains = con.execute(
        """
        select sa, n_connections, n_dns_preceded,
               round(dns_preceded_frac, 3) as dns_frac,
               round(median_latency_s, 3) as median_latency_s
        from mart_seq_chains order by n_connections desc limit 6
        """
    ).fetch_df()
    _sess = con.execute(
        "select count(*) as n_sessions, max(n_flows) as max_flows, "
        "round(avg(n_flows),1) as mean_flows, round(max(duration_s),1) as max_duration_s "
        "from mart_seq_sessions"
    ).fetch_df()
    mo.vstack([
        mo.md(
            """
            ## Temporal sequence analysis

            ### Service transition matrix P(next | current)

            Traffic is strongly **self-transitioning** (bursts): SSDP→SSDP has
            lift 22.8, DNS→DNS 6.2 — hosts repeat the same service in runs.
            Cross-service structure (ntp↔http at the top of the hour) is the
            "timely flow logic" the NextServiceGRU learns.
            """
        ),
        _chart,
        mo.md(
            """
            ### Resolve → connect chains (DNS followed by HTTP/HTTPS ≤ 5 s)

            A forensically interesting *negative*: the suspect host
            `192.168.15.4` has **zero** DNS-preceded connections — its
            resolver sits inside the dorm NAT, invisible to this sensor. The
            Mac `192.168.1.64` shows the expected chain (median 0.75 s).
            """
        ),
        mo.ui.table(_chains, selection=None),
        mo.md("### Timely sessions (max 5 s step)"),
        mo.ui.table(_sess, selection=None),
    ])
    return


@app.cell
def _(ROOT, json, mo):
    _reg = ROOT / "models_out" / "registry.json"
    _active = (json.loads(_reg.read_text()).get("minimodels", {}).get("active")
               if _reg.exists() else None)
    if _active:
        meta = json.loads(
            (ROOT / "models_out" / "minimodels" / _active / "meta.json").read_text())
        _clf = meta["classifier"]["metrics"]
        _ae = meta["autoencoder"]["metrics"]
        _ns = meta.get("next_service", {}).get("metrics")
        _tiles = [
            mo.stat(f"{_clf['val_accuracy']:.3f}",
                    label="classifier val accuracy",
                    caption=f"majority baseline {_clf['majority_baseline']:.3f}"),
            mo.stat(f"{_ae['val_mse_mean']:.3f}",
                    label="autoencoder val MSE",
                    caption=f"p95 {_ae['val_mse_p95']:.3f}"),
        ]
        if _ns:
            _tiles.append(mo.stat(
                f"{_ns['val_accuracy']:.3f}",
                label="next-service GRU val accuracy",
                caption=(f"repeat-last {_ns['repeat_last_baseline']:.3f} · "
                         f"marginal {_ns['marginal_baseline']:.3f}")))
        _tls = meta.get("tls_fingerprint", {}).get("metrics")
        if _tls:
            _tiles.append(mo.stat(
                f"{_tls['val_accuracy']:.3f}",
                label="TLS stack GRU val accuracy",
                caption=(f"majority {_tls['majority_baseline']:.3f} · "
                         + ("adopted" if _tls["adopted"]
                            else "below baseline — not adopted"))))
        _dlm = meta.get("dns_name_lm", {}).get("metrics")
        if _dlm:
            _tiles.append(mo.stat(
                f"{_dlm['val_bits_per_char']:.2f}",
                label="DNS char-LM val bits/char",
                caption=(f"unigram {_dlm['unigram_baseline_bits']:.2f} · "
                         + ("adopted" if _dlm["adopted"] else "not adopted"))))
        _tiles.append(mo.stat(f"{_ae['n_train']:,} / {_ae['n_val']:,}",
                              label="train / val flows",
                              caption="time-ordered 80/20 split"))
        _new_specs = [
            ("bytedist_ae", "ByteDistAE", "val_mse_mean", None),
            ("volume_service", "VolumeServiceMLP", "val_accuracy", "majority_baseline"),
            ("bytes_in", "BytesInMLP", "val_mae_log", "baseline_mae_log"),
            ("duration_band", "DurationBandMLP", "val_accuracy", "majority_baseline"),
            ("pair_gap", "PairGapGRU", "val_mae_log_gap", "repeat_last_mae"),
            ("session_len", "SessionLenMLP", "val_accuracy", "majority_baseline"),
            ("unanswered", "UnansweredMLP", "val_accuracy", "majority_baseline"),
            ("tls_setup", "TlsSetupBandMLP", "val_accuracy", "majority_baseline"),
            ("dns_kind", "DnsKindClassifier", "val_accuracy", "majority_baseline"),
            ("beacon_channel", "BeaconChannelMLP", "val_accuracy", "majority_baseline"),
        ]
        _new_rows = [
            {"model": _nm,
             "val metric": f"{_m[_mk]:.3f}",
             "baseline": "—" if _bk is None else f"{_m[_bk]:.3f}",
             "adopted": "— (ungated)" if _bk is None
                        else ("yes" if _m["adopted"] else "no")}
            for _key, _nm, _mk, _bk in _new_specs
            if (_m := meta.get(_key, {}).get("metrics"))
        ]
        _body = mo.vstack([
            mo.md(f"## Minimodels — `{meta['version']}` (device: {meta['device']})\n"
                  "Fifteen models. The original five: anomaly autoencoder, "
                  "byte-distribution service classifier, the **NextServiceGRU** "
                  "sequence model, the **TlsRecordGRU** (client stack from TLS "
                  "record structure) and the **DnsCharLM** (name-shape "
                  "perplexity, the DGA-detector shape) — tiles below. The ten "
                  "2026-10-30 additions follow in the table, each gated "
                  "against its declared non-ML baseline (majority / "
                  "per-group median / repeat-last); honest negatives keep "
                  "`adopted: no`."),
            mo.hstack(_tiles, justify="space-around"),
            mo.md("### Ten additional minimodels — adoption verdicts"),
            mo.ui.table(_new_rows, selection=None),
        ])
    else:
        meta = None
        _body = mo.md("⚠️ No trained models — run `make train ml`.")
    _body
    return (meta,)


@app.cell
def _(SEQ, alt, con, meta, mo, np, pd, styled):
    mo.stop(meta is None)
    _sc = con.execute(
        """
        select m.anomaly_score, f.sa, f.da, f.dp, f.service_class, f.bytes_total
        from ml_flow_scores m join feat_flows f using (flow_id)
        where m.anomaly_score is not null
        """
    ).fetch_df()
    _hc, _he = np.histogram(_sc.anomaly_score, bins=50)
    _hdf = pd.DataFrame({"lo": _he[:-1], "hi": _he[1:], "n": _hc})
    _hist = styled(
        alt.Chart(_hdf).mark_bar(color=SEQ[4], cornerRadiusEnd=2)
        .encode(
            x=alt.X("lo:Q", bin="binned",
                    title="reconstruction error (anomaly score)"),
            x2="hi:Q",
            y=alt.Y("n:Q", scale=alt.Scale(type="symlog"), title="flows (symlog)"),
            tooltip=[alt.Tooltip("n:Q", title="flows")],
        ), height=220,
    )
    _top = (_sc.sort_values("anomaly_score", ascending=False)
            .head(8)[["anomaly_score", "sa", "da", "dp", "service_class", "bytes_total"]]
            .round({"anomaly_score": 3}))
    mo.vstack([
        mo.md(
            """
            ### Anomaly scores (FlowAutoencoder)

            The top anomaly — `65.212.118.21 → 192.168.1.64:33410`, a 22.5 kB
            one-way external push — was independently flagged by the EDA's
            exfil heuristic; model and rule agree.
            """
        ),
        _hist,
        mo.ui.table(_top, selection=None),
    ])
    return


@app.cell
def _(PAL, alt, con, meta, mo, styled):
    mo.stop(meta is None)
    _emb = con.execute(
        """
        select emb_x, emb_y, weak_label_service, anomaly_score
        from ml_flow_scores where emb_x is not null
        """
    ).fetch_df()
    _top3 = _emb.weak_label_service.value_counts().head(3).index.tolist()
    _emb["cls"] = _emb.weak_label_service.where(_emb.weak_label_service.isin(_top3), "other")
    _order = _top3 + ["other"]
    _chart = styled(
        alt.Chart(_emb.sample(min(len(_emb), 3000), random_state=7))
        .mark_circle(size=24, opacity=0.6)
        .encode(
            x=alt.X("emb_x:Q", title="embedding dim 1"),
            y=alt.Y("emb_y:Q", title="embedding dim 2"),
            color=alt.Color("cls:N", scale=alt.Scale(domain=_order,
                                                     range=PAL[:3] + ["#898781"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["cls", alt.Tooltip("anomaly_score:Q", format=".3f")],
        ), height=300,
    ).interactive()
    mo.vstack([
        mo.md("### Autoencoder bottleneck embedding (2 of 3 dims)\n"
              "Service classes separate with no service labels in the input — "
              "volume/timing/entropy features alone carry the structure. "
              "(Scatter limited to 3 colored classes + other, per palette "
              "all-pairs validation.)"),
        _chart,
    ])
    return


@app.cell
def _(PAL, alt, meta, mo, pd, styled):
    mo.stop(meta is None)
    _m = meta["classifier"]["metrics"]
    _df = pd.DataFrame({
        "model": ["majority baseline", "ByteDistClassifier"],
        "accuracy": [_m["majority_baseline"], _m["val_accuracy"]],
    })
    _chart = styled(
        alt.Chart(_df).mark_bar(color=PAL[0], cornerRadiusEnd=4)
        .encode(
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", title=None),
            tooltip=["model", alt.Tooltip("accuracy:Q", format=".3f")],
        ), height=120,
    )
    mo.vstack([
        mo.md(f"### Service from payload byte histogram alone\n"
              f"256-bin `byte_dist` → {len(_m['classes'])} classes "
              f"({', '.join(_m['classes'])}); weak labels from ports."),
        _chart,
    ])
    return


@app.cell
def _(ROOT, json, mo, pd):
    _p = ROOT / "lean" / "out" / "lean_results.json"
    if _p.exists():
        lean_results = json.loads(_p.read_text())
        _claims = pd.DataFrame(lean_results["claims"])
        _claims["status"] = _claims.status.map(
            {"proved": "✅ proved", "assumed": "⚠️ assumed"})
        _body = mo.vstack([
            mo.md(
                """
                ## Lean 4 + SampCert formal layer

                Lean v4.34.1 + mathlib, SampCert pinned @ `a2c53bb` — builds
                with **zero `sorry`**. The ledger separates machine-checked
                theorems from declared assumptions (goal.md guideline 10):
                """
            ),
            mo.ui.table(_claims, selection=None),
        ])
    else:
        lean_results = None
        _body = mo.md("⚠️ No Lean results — run `make lean`.")
    _body
    return (lean_results,)


@app.cell
def _(lean_results, mo, pd):
    mo.stop(lean_results is None)
    _b = pd.DataFrame(lean_results["beacons"])[
        ["sa", "da", "dp", "n_gaps", "sql_is_beacon", "lean_is_beacon", "agree"]]
    _f = pd.DataFrame(lean_results["fanout"])[
        ["sa", "n_distinct_dst", "sql_is_fanout", "lean_is_fanout", "agree"]]
    _r = pd.DataFrame(lean_results["returns"])
    _a = pd.DataFrame(lean_results.get("assoc", []))
    _s = pd.DataFrame(lean_results.get("sessions", []))
    _tc = pd.DataFrame(lean_results.get("tls_canonical", []))
    _tr = pd.DataFrame(lean_results.get("tls_rare", []))
    _dn = pd.DataFrame(lean_results.get("dns_names", []))
    _df_ = pd.DataFrame(lean_results.get("dns_fans", []))
    _acc = {
        "beacon checks": mo.ui.table(_b, selection=None),
        "fan-out checks": mo.ui.table(_f, selection=None),
        "return-trip checks": mo.ui.table(_r, selection=None),
    }
    _extra = ""
    if len(_a):
        _acc["chi-square recomputation (exact ℚ)"] = mo.ui.table(_a, selection=None)
        _extra += (f" χ² recomputed exactly over ℚ agrees with SQL on "
                   f"**{int(_a.agree.sum())}/{len(_a)}** exported tables.")
    if len(_s):
        _acc["timely-session checks"] = mo.ui.table(_s, selection=None)
        _extra += (f" **{int(_s.lean_is_timely_session.sum())}/{len(_s)}** "
                   f"SQL sessions verify as Lean `IsTimelySession` chains.")
    if len(_tc):
        _acc["TLS canonical-handshake checks"] = mo.ui.table(_tc, selection=None)
        _extra += (f" TLS: **{int(_tc.agree.sum())}/{len(_tc)}** canonical-handshake "
                   f"verdicts agree")
    if len(_tr):
        _acc["TLS rare-stack checks"] = mo.ui.table(_tr, selection=None)
        _extra += (f", **{int(_tr.agree.sum())}/{len(_tr)}** stack counts + rare "
                   f"flags agree.")
    if len(_dn):
        _acc["DNS name well-formedness checks"] = mo.ui.table(_dn, selection=None)
        _extra += (f" DNS: **{int(_dn.agree.sum())}/{len(_dn)}** RFC-shape "
                   f"verdicts agree")
    if len(_df_):
        _acc["DNS subdomain-fan checks"] = mo.ui.table(_df_, selection=None)
        _extra += (f", **{int(_df_.agree.sum())}/{len(_df_)}** subdomain-fan "
                   f"verdicts agree.")
    mo.vstack([
        mo.md(
            f"""
            ### Executable checks: SQL detectors vs Lean definitions

            `lake exe checks` evaluates the *decidable* Lean predicates
            (`IsBeacon`, `IsFanout`, `HasReturnTrip`, `IsTimelySession`,
            `IsCanonicalHandshake`, `IsRareStack`, `IsWellFormedQName`,
            `IsSubdomainFan`) and the exact ℚ chi-square on data exported
            from the enclave store.
            Agreement: **beacons {int(_b.agree.sum())}/{len(_b)}**,
            **fan-out {int(_f.agree.sum())}/{len(_f)}**,
            return-trip motif found in
            **{int(_r.lean_has_return_trip.sum())}/{len(_r)}** sequences.{_extra}
            """
        ),
        mo.accordion(_acc),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The formulation (excerpts)

        **Beacon** — decidable over ℚ (squared CV, no square roots):

        ```lean
        def IsBeacon (p : BeaconParams) (gapsMs : List ℚ) : Prop :=
          p.minFlows ≤ gapsMs.length + 1 ∧
          0 < qMean gapsMs ∧
          p.minPeriodMs ≤ qMean gapsMs ∧
          qSsd gapsMs ≤ p.maxCvSq * (qMean gapsMs) ^ 2 * ((gapsMs.length - 1 : ℕ) : ℚ)
        ```

        **Sensitivity of the clipped histogram cell** (the DP hypothesis):

        ```lean
        theorem clippedSum_sensitivity (c : ℕ) : sensitivity (clippedSum c) c
        ```

        **The proved privacy theorem** — instantiating SampCert's verified
        discrete Laplace mechanism:

        ```lean
        theorem privClippedCell_PureDP (c ε₁ ε₂ : ℕ+) :
            PureDP (privClippedCell c ε₁ ε₂) ((ε₁ : NNReal) / ε₂)
        ```

        **The verified sampler law** (entry point for synthetic generation):

        ```lean
        theorem sampleCellPMF_law (ws : List ℕ) (h : 0 < ws.sum) (i : ℕ) (hi : i < ws.length) :
            (sampleCellPMF ws h : SLang ℕ) i = (ws[i]'hi : ENNReal) / (ws.sum : ENNReal)
        ```

        **Chi-square association, exact over ℚ** (recomputed from exported cells):

        ```lean
        def chiSq (t : ContTable) : ℚ := ...  -- expected from margins, (O−E)²/E

        theorem chiSq_nonneg (t : ContTable) : 0 ≤ chiSq t
        theorem chiSq_eq_zero_of_indep : IsIndependent t → chiSq t = 0
        ```

        **Timely sessions** (the temporal flow logic, decidable):

        ```lean
        def IsTimelySession (maxGap : ℚ) (times : List ℚ) : Prop :=
          List.IsChain (fun a b => a ≤ b ∧ b - a ≤ maxGap) times

        theorem isTimelySession_mono    -- relaxing the gap bound preserves it
        theorem isTimelySession_ordered -- a timely session is chronological
        ```
        """
    )
    return


@app.cell
def _(PAL, ROOT, alt, json, mo, pd, styled):
    _rel_p = ROOT / "lean" / "out" / "dp_release.json"
    _cnt_p = ROOT / "lean" / "data" / "dp_counts.json"
    if _rel_p.exists() and _cnt_p.exists():
        dp_release = json.loads(_rel_p.read_text())
        _raw = {c["cell"]: c["clipped_count"] for c in json.loads(_cnt_p.read_text())["cells"]}
        _df = pd.DataFrame([
            {"cell": c["cell"], "value": _raw.get(c["cell"], 0), "kind": "clipped (in-enclave)"}
            for c in dp_release["cells"]
        ] + [
            {"cell": c["cell"], "value": c["released"], "kind": "DP released (ε-noised)"}
            for c in dp_release["cells"]
        ])
        _led = dp_release["ledger"]
        _chart = styled(
            alt.Chart(_df).mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("cell:N", sort="-y", title=None),
                y=alt.Y("value:Q", title="count"),
                color=alt.Color("kind:N",
                                scale=alt.Scale(domain=[
                                    "clipped (in-enclave)", "DP released (ε-noised)"],
                                    range=[PAL[0], PAL[1]]),
                                legend=alt.Legend(title=None, orient="top")),
                xOffset="kind:N",
                tooltip=["cell", "kind", "value"],
            ), height=280,
        )
        dp_body = mo.vstack([
            mo.md(
                f"""
                ### Differentially private release (SampCert discrete Laplace)

                Privacy unit = one source host; clip = {dp_release['clip']};
                per-cell ε = {dp_release['eps_per_cell']['num']}/{dp_release['eps_per_cell']['den']};
                ledger: **{_led['n_cells']} cells → total ε =
                {_led['total_eps_num']}/{_led['total_eps_den']} ≤ budget
                {_led['budget']}** (the executable refuses over-budget exports).
                Only the orange bars may leave the enclave. Noise of scale 20
                dominates small cells — that is honest DP utility, reported,
                not hidden.
                """
            ),
            _chart,
        ])
    else:
        dp_release = None
        dp_body = mo.md("⚠️ No DP release — run `make lean`.")
    dp_body
    return (dp_release,)


@app.cell
def _(PAL, ROOT, alt, json, math, mo, pd, styled):
    _p = ROOT / "lean" / "out" / "samples.json"
    if not _p.exists():
        _out = mo.md("⚠️ No verified-sampler draws — run `make lean-sample`.")
    else:
        _s = json.loads(_p.read_text())
        _n = _s["n_draws"]
        _tot = _s["total_weight"]
        _cells = [c for c in _s["cells"] if c["weight"] > 0]
        _df = pd.DataFrame([
            {"cell": c["cell"], "frequency": c["weight"] / _tot, "kind": "released weight wᵢ/Σw"}
            for c in _cells
        ] + [
            {"cell": c["cell"], "frequency": c["draws"] / _n, "kind": f"observed draws /{_n}"}
            for c in _cells
        ])
        # χ² goodness-of-fit of draws against the proved law P(i) = wᵢ/Σw
        _chi2 = sum(
            (c["draws"] - _n * c["weight"] / _tot) ** 2 / (_n * c["weight"] / _tot)
            for c in _cells
        )
        _dof = len(_cells) - 1
        # Wilson–Hilferty approximation for the p-value (avoids scipy)
        _z = ((_chi2 / _dof) ** (1 / 3) - (1 - 2 / (9 * _dof))) / math.sqrt(2 / (9 * _dof))
        _pval = 0.5 * math.erfc(_z / math.sqrt(2))
        _chart = styled(
            alt.Chart(_df).mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("cell:N", sort="-y", title=None),
                y=alt.Y("frequency:Q", title="probability"),
                color=alt.Color("kind:N", scale=alt.Scale(range=[PAL[0], PAL[2]]),
                                legend=alt.Legend(title=None, orient="top")),
                xOffset="kind:N",
                tooltip=["cell", "kind", alt.Tooltip("frequency:Q", format=".4f")],
            ), height=260,
        )
        _out = mo.vstack([
            mo.md(
                f"""
                ### Verified sampling — proved law vs observed draws

                `lake exe sample` drew **{_n:,}** samples through SampCert's
                FFI. The law `P(cell i) = wᵢ/Σw` is a *theorem*
                (`sampleCellPMF_law`); the χ² test checks the unverified I/O
                end-to-end: **χ² = {_chi2:.1f}** on {_dof} df →
                p ≈ {_pval:.2f} {'✅ consistent' if _pval > 0.01 else '❌ inconsistent'}.
                """
            ),
            _chart,
        ])
    _out
    return


@app.cell
def _(con, mo, pd):
    try:
        import geoip2fast as _g2f
        _geo = _g2f.GeoIP2Fast()
        _ext = con.execute(
            """
            select 'source' as direction, sa as ip,
                   count(*) as flows, sum(bytes_total) as bytes
            from feat_flows where not src_internal group by 1, 2
            union all
            select 'destination', da, count(*), sum(bytes_total)
            from feat_flows where not dst_internal group by 1, 2
            """
        ).fetch_df()
        _lk = {ip: _geo.lookup(ip) for ip in _ext.ip.unique()}
        _ext["country_code"] = [_lk[ip].country_code or "??" for ip in _ext.ip]
        _ext["country"] = [_lk[ip].country_name or "unknown" for ip in _ext.ip]
        geo_df = (_ext.groupby(["direction", "country_code", "country"],
                               as_index=False)
                  .agg(flows=("flows", "sum"), bytes=("bytes", "sum"),
                       ips=("ip", "nunique"))
                  .sort_values("flows", ascending=False))
        _src = geo_df[geo_df.direction == "source"].drop(columns="direction")
        _dst = geo_df[geo_df.direction == "destination"].drop(columns="direction")
        _body = mo.vstack([
            mo.md(
                """
                ## IP → country enrichment — sources and destinations

                External IPs on both sides of the enclave boundary, resolved
                offline with `geoip2fast` (bundled database): **source
                countries** are external `sa` (inbound traffic), **destination
                countries** external `da` (outbound). **Caveat**: 2026
                geolocation applied to 2017 addresses — CDN/registry moves
                make this approximate; it is enrichment for visualization,
                not forensic evidence.
                """
            ),
            mo.hstack([
                mo.vstack([mo.md("**Source countries (external `sa`)**"),
                           mo.ui.table(_src.head(12), selection=None)]),
                mo.vstack([mo.md("**Destination countries (external `da`)**"),
                           mo.ui.table(_dst.head(12), selection=None)]),
            ], widths="equal"),
        ])
    except Exception as _e:  # geoip2fast missing or lookup failure
        geo_df = pd.DataFrame()
        _body = mo.md(f"⚠️ IP→country enrichment unavailable: `{_e}` — "
                      "run `make setup` to install geoip2fast.")
    _body
    return (geo_df,)


@app.cell
def _(geo_df, json, mo):
    if geo_df.empty:
        _map_out = mo.md("")
    else:
        _centroids = {
            "US": (39.8, -98.6), "CA": (56.1, -106.3), "MX": (23.6, -102.5),
            "GB": (54.0, -2.9), "IE": (53.2, -7.7), "FR": (46.6, 2.2),
            "DE": (51.2, 10.4), "NL": (52.2, 5.3), "BE": (50.6, 4.7),
            "CH": (46.8, 8.2), "AT": (47.6, 14.1), "IT": (42.8, 12.1),
            "ES": (40.2, -3.6), "PT": (39.6, -8.0), "SE": (62.8, 16.7),
            "NO": (64.6, 11.5), "DK": (56.0, 9.9), "FI": (64.5, 26.3),
            "PL": (52.1, 19.4), "CZ": (49.8, 15.5), "RU": (61.5, 105.0),
            "UA": (48.4, 31.2), "TR": (39.0, 35.2), "IL": (31.4, 35.0),
            "SA": (24.0, 45.0), "AE": (24.0, 54.0), "IN": (22.9, 79.6),
            "CN": (35.9, 104.2), "HK": (22.3, 114.2), "TW": (23.7, 121.0),
            "JP": (36.6, 138.0), "KR": (36.4, 127.8), "SG": (1.35, 103.8),
            "TH": (15.1, 101.0), "MY": (4.2, 102.0), "ID": (-2.5, 118.0),
            "PH": (12.9, 121.8), "AU": (-25.3, 133.8), "NZ": (-41.8, 172.8),
            "BR": (-10.8, -53.1), "AR": (-35.4, -65.2), "CL": (-35.7, -71.5),
            "CO": (4.1, -73.1), "PE": (-9.2, -75.0), "ZA": (-29.0, 25.1),
            "EG": (26.6, 30.0), "NG": (9.1, 8.7), "KE": (0.5, 37.9),
        }
        _feats = []
        for _row in geo_df.itertuples():
            _c = _centroids.get(_row.country_code)
            if _c:
                # side-by-side circles per country: sources west, destinations east
                _off = -1.6 if _row.direction == "source" else 1.6
                _feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [_c[1] + _off, _c[0]]},
                    "properties": {"country": _row.country,
                                   "direction": _row.direction,
                                   "flows": int(_row.flows),
                                   "ips": int(_row.ips)},
                })
        _geojson = json.dumps({"type": "FeatureCollection", "features": _feats})
        _html = """<!doctype html><html><head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<style>html,body,#map{margin:0;height:100%;background:#fcfcfb}</style>
</head><body><div id="map"></div><script>
const data = __GEOJSON__;
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-40, 35], zoom: 1.2, attributionControl: true
});
map.addControl(new maplibregl.NavigationControl());
map.on('load', () => {
  map.addSource('flows', {type: 'geojson', data});
  map.addLayer({
    id: 'flow-circles', type: 'circle', source: 'flows',
    paint: {
      'circle-radius': ['max', 5, ['*', 3.2, ['log10', ['+', 1, ['get', 'flows']]]]],
      'circle-color': ['match', ['get', 'direction'],
                       'source', '#eb6834', '#2a78d6'],
      'circle-opacity': 0.75,
      'circle-stroke-color': '#fcfcfb',
      'circle-stroke-width': 2
    }
  });
  map.on('click', 'flow-circles', (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup().setLngLat(e.lngLat)
      .setHTML(`<b>${p.country}</b> — ${p.direction}<br/>${p.flows} flows, ${p.ips} IPs`)
      .addTo(map);
  });
  map.on('mouseenter', 'flow-circles', () => map.getCanvas().style.cursor = 'pointer');
  map.on('mouseleave', 'flow-circles', () => map.getCanvas().style.cursor = '');
});
const legend = document.createElement('div');
legend.style.cssText = 'position:absolute;top:10px;left:10px;background:#fcfcfbcc;' +
  'padding:6px 10px;border-radius:6px;font:12px sans-serif;color:#52514e';
legend.innerHTML = '<span style="color:#eb6834">●</span> source countries (external sa) &nbsp; ' +
  '<span style="color:#2a78d6">●</span> destination countries (external da)';
document.body.appendChild(legend);
</script></body></html>""".replace("__GEOJSON__", _geojson)
        import base64 as _b64
        _src = "data:text/html;base64," + _b64.b64encode(_html.encode()).decode()
        _map_out = mo.vstack([
            mo.md("### Source & destination countries — MapLibre GL JS\n"
                  "Orange = source countries (external `sa`, inbound), blue = "
                  "destination countries (external `da`, outbound); the pair "
                  "sits side by side per country. Circle area ∝ log(flows). "
                  "Click a circle for details. (Requires internet for map "
                  "tiles and the MapLibre CDN.)"),
            mo.Html(f'<iframe src="{_src}" style="width:100%;height:520px;'
                    f'border:1px solid #e1e0d9;border-radius:8px"></iframe>'),
        ])
    _map_out
    return


@app.cell
def _(ROOT, dp_release, meta, mo):
    import hashlib as _hashlib
    _fp = _hashlib.sha256((ROOT / "flow.jsonl").read_bytes()).hexdigest()[:16]
    mo.md(
        f"""
        ---
        ## Provenance

        - data fingerprint (sha256, first 16): `{_fp}`
        - model artifact: `{meta['version'] if meta else '—'}`
          (torch {meta['torch_version'] if meta else '—'}, seed
          {meta['seed'] if meta else '—'})
        - formal layer: Lean v4.34.1, SampCert @ `a2c53bb` (mathlib v4.34.1)
        - DP mechanism: `{dp_release['mechanism'] if dp_release else '—'}`
        - DP outputs are **excluded from bitwise reproducibility** by design
          (non-replayable enclave randomness).
        - Knowledge base: `wiki/` (LLM-Wiki — see `wiki/SCHEMA.md`).
        """
    )
    return


if __name__ == "__main__":
    app.run()

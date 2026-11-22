# (old, new) source replacements for notebook cell 5: re-add ILP division edges lost by the 1:1 motion relink.
REPLACEMENTS = [
    (
        '''            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
''',
        '''            stats["motion_relink_replaced_raw_edges"] = len(edges)
            _raw_edges = edges
            edges = motion_edges
            # CSILP: the ILP (cell-state costs) may emit forks; the 1:1 motion relink drops the second child.
            # Re-add ILP fork edges whose source kept exactly one child and whose target is still parentless.
            if os.environ.get("BIOHUB_CSILP_READD", "1") != "0":
                _cnt = {}
                for _e in _raw_edges:
                    _cnt[int(_e["source_id"])] = _cnt.get(int(_e["source_id"]), 0) + 1
                _raw_forks = {s for s, c in _cnt.items() if c >= 2}
                _kids = {}
                _has_par = set()
                for _e in edges:
                    _kids.setdefault(int(_e["source_id"]), []).append(_e)
                    _has_par.add(int(_e["target_id"]))
                _readded = 0
                for _e in _raw_edges:
                    _s, _t = int(_e["source_id"]), int(_e["target_id"])
                    if (_s in _raw_forks and len(_kids.get(_s, [])) == 1 and _t not in _has_par
                            and _s in nodes_by_id and _t in nodes_by_id
                            and int(nodes_by_id[_t]["t"]) == int(nodes_by_id[_s]["t"]) + 1
                            and any(int(_k["target_id"]) != _t for _k in _kids[_s])):
                        _e2 = dict(_e)
                        _e2["distance_um"] = edge_distance_um(nodes_by_id[_s], nodes_by_id[_t])
                        if OUTPUT_EDGE_MAX_UM > 0 and float(_e2["distance_um"]) > OUTPUT_EDGE_MAX_UM:
                            continue
                        edges.append(_e2)
                        _kids[_s].append(_e2)
                        _has_par.add(_t)
                        _readded += 1
                stats["csilp_forks_readded"] = _readded
                print(f"  [{dataset}] CSILP re-added {_readded} ILP fork edges after motion relink", flush=True)
''',
    ),
]

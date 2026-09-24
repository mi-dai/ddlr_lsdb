# ddlr_lsdb
Host-galaxy matching for transients with the directional light radius (DDLR) method, run at scale with [LSDB](https://docs.lsdb.io) + dask.

- `ddlr.py`: DDLR calculation (placeholder), galaxy-shape converters, and partition functions
- `ddlr_host_match.ipynb`: loads the transient and galaxy (Rubin DP2 object or Legacy Survey DR10.1) catalogs, crossmatches candidates, computes DDLR, and keeps the best host per transient

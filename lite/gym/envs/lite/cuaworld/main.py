"""lite.cuaworld — gym-anything (CMU CUAWorld) software envs on CUA-Lite.

Each line below registers one ``lite.cuaworld.<software>``. Materials (env.json + scripts +
tasks + ``registered.json``) come from the materials repo
(HF dataset ``cua-lite/lite.cuaworld-assets``) via ``scripts/install.sh`` into a
gitignored ``.cache/``; images build on ``cua-lite/lite.cuaworld.base`` — the shared
sandbox Dockerfile built with ``--build-arg USER=ga``, so the in-container
unix user is ``ga``, so fetched assets need no runtime user/path rewrite.
See ``lite/gym/envs/lite/cuaworld/README.md``.

To onboard a software: import it into the materials repo (pristine baseline +
adapt + ``registered.json``), then add one ``register_cuaworld_software(...)`` line.

A per-software kwarg means "this software needs MORE than the shared baseline in
``configs/default.yaml``". Do not restate the baseline: every line here used to
carry a ``memory=`` of 4/6/8GB back when the baseline was 4GB, and once the
baseline moved to 8GB those became silent DOWNGRADES rather than overrides. Pass
``memory=`` again only for a software measured to need more than 8GB.
"""

from __future__ import annotations

from lite.gym.envs.lite.cuaworld.src.software import register_cuaworld_software

register_cuaworld_software("pymol", "pymol_env", cpu="4")

# Active environments migrated to the sandbox + locked materials repo:
register_cuaworld_software("gmat",        "gmat_env",        cpu="4")
register_cuaworld_software("openvsp",     "openvsp_env",     cpu="4")
register_cuaworld_software("pycharm",     "pycharm_env",     cpu="4")
register_cuaworld_software("vscode",      "vscode_env",      cpu="4")
register_cuaworld_software("dbeaver",     "dbeaver_env",     cpu="4")
register_cuaworld_software("eclipse",     "eclipse_env",     cpu="4")
register_cuaworld_software("coppeliasim", "coppeliasim_env", cpu="4")
register_cuaworld_software("sumo",        "sumo_env",        cpu="4")
register_cuaworld_software("imagej",      "imagej_env",      cpu="4")
register_cuaworld_software("freecad",     "freecad_envb",    cpu="4")
# RStudio Desktop is not registered: its embedded Chromium requires
# seccomp=unconfined because the zygote cannot clone a user namespace under the
# default container seccomp profile, and --no-sandbox is not honored. That
# privilege is outside the lite.cuaworld sandbox policy.
# register_cuaworld_software("rstudio",     "rstudio_env",     cpu="4")
register_cuaworld_software("qgis",        "qgis_env",        cpu="4")
register_cuaworld_software("webots",      "webots_env",      cpu="4")
register_cuaworld_software("hec_ras",     "hec_ras_env",     cpu="4")
register_cuaworld_software("knime",       "knime_analytics_platform_env", cpu="4")
register_cuaworld_software("slicer3d",    "slicer3d_env",    cpu="4")
register_cuaworld_software("blender3d",   "blender3d_env",   cpu="4", gpus=1)
register_cuaworld_software("moodle",      "moodle_env",      cpu="4")

# ── Broad desktop software domains ─────────────────────────────────────────────
# Office
register_cuaworld_software("libreoffice_calc", "libreoffice_calc_env", cpu="4")
# Media / creative
register_cuaworld_software("vlc_media_player", "vlc_media_player_env", cpu="4")
register_cuaworld_software("ardour",      "ardour_env",      cpu="4")
# Data / statistics / analysis
register_cuaworld_software("gretl",       "gretl_env",       cpu="4")
register_cuaworld_software("ugene",       "ugene_env",       cpu="4")
register_cuaworld_software("jstock",      "jstock_env",      cpu="4")
# CAD / 3D design
register_cuaworld_software("librecad",    "librecad_env",    cpu="4")
register_cuaworld_software("solvespace",  "solvespace_env",  cpu="4")
register_cuaworld_software("sweet_home_3d", "sweet_home_3d_env", cpu="4")
register_cuaworld_software("openrocket",  "openrocket_env",  cpu="4")
# GIS
register_cuaworld_software("gvsig_desktop", "gvsig_desktop_env", cpu="4")
# Diagramming
register_cuaworld_software("diagrams_net", "diagrams_net_env", cpu="4")
# Astronomy / scientific imaging
register_cuaworld_software("astroimagej", "astroimagej_env", cpu="4")
register_cuaworld_software("kstars_sim",  "kstars_sim_env",  cpu="4")
# Education / knowledge
register_cuaworld_software("geogebra",    "geogebra_env",    cpu="4")
register_cuaworld_software("gcompris",    "gcompris_env",    cpu="4")
# Web platforms (de-dockerized to native LAMP via supervisor — see materials)
register_cuaworld_software("wordpress",   "wordpress_env",   cpu="4")
register_cuaworld_software("openemr",     "openemr_env",     cpu="4")
register_cuaworld_software("odoo",        "odoo_env",        cpu="4")
# Engineering / science (satellite tracking, wind-turbine sim, life-cycle assessment)
register_cuaworld_software("gpredict",    "gpredict_env",    cpu="4")
register_cuaworld_software("qblade",      "qblade_env",      cpu="4")
register_cuaworld_software("openlca",     "openlca_env",     cpu="4")

# Not registered: known upstream/material blockers.
#   libreoffice_writer / libreoffice_impress — soffice aborts with a LibreOffice
#     DeploymentException (Signal 6) on sandbox, breaking both the setup's
#     DOCX/ODT conversion and the launch (calc works; office is covered by it).
#   zotero — XPCOMGlueLoad needed libdbus-glib-1-2 (added); launch still under
#     verification.
# Not registered: excluded software candidates. opentoonz (no rootless install), gimp (0 eval
#   tasks), draw_desktop (duplicate of diagrams_net), jasp/jamovi (Flatpak/bwrap
#   needs seccomp=unconfined).

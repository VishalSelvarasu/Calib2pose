import os
import shutil
import zipfile
import urllib.request

# Pinned. Upstream main can change the UR5e model or the mesh set, which would
# silently change every number in this stage. Update the SHAs deliberately.
MENAGERIE_SHA = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
YCB_SIM_SHA = "57546b87f4724c947eadd4241a7892473febb88d"
MENAGERIE_URL = ("https://codeload.github.com/google-deepmind/"
                 f"mujoco_menagerie/zip/{MENAGERIE_SHA}")
YCB_URL = f"https://codeload.github.com/vikashplus/YCB_sim/zip/{YCB_SIM_SHA}"

UR5E_DIR = "ur5e"
STAGE4_ASSETS = os.path.join("..", "stage4_keypoints", "assets")
DRILL = "035_power_drill"


def _zip_root(z):
    """GitHub names the top folder <repo>-<ref>. Deriving it from the archive
    means the pin above can be changed without touching the extract paths."""
    roots = {m.split("/")[0] for m in z.namelist() if "/" in m}
    if len(roots) != 1:
        raise RuntimeError(f"unexpected archive layout: {sorted(roots)[:3]}")
    return roots.pop()


def fetch_ur5e():
    if os.path.exists(os.path.join(UR5E_DIR, "ur5e.xml")):
        print(f"UR5e already present in {UR5E_DIR}/")
        return
    print("downloading MuJoCo Menagerie (large, extracting UR5e only)...")
    zpath = "_menagerie.zip"
    urllib.request.urlretrieve(MENAGERIE_URL, zpath)

    with zipfile.ZipFile(zpath) as z:
        prefix = f"{_zip_root(z)}/universal_robots_ur5e/"
        members = [m for m in z.namelist() if m.startswith(prefix)]
        for m in members:
            rel = m[len(prefix):]
            if not rel:
                continue
            dst = os.path.join(UR5E_DIR, rel)
            if m.endswith("/"):
                os.makedirs(dst, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(m) as src, open(dst, "wb") as out:
                out.write(src.read())
    os.remove(zpath)
    print(f"  -> {UR5E_DIR}/ ({len(members)} files)")


def fetch_drill():
    assets = os.path.join(UR5E_DIR, "assets")
    os.makedirs(assets, exist_ok=True)
    need = [f"{DRILL}.msh", f"{DRILL}.png"]
    if all(os.path.exists(os.path.join(assets, n)) for n in need):
        print("drill mesh already present")
        return

    if all(os.path.exists(os.path.join(STAGE4_ASSETS, n)) for n in need):
        for n in need:
            shutil.copy(os.path.join(STAGE4_ASSETS, n),
                        os.path.join(assets, n))
        print(f"copied drill mesh from {STAGE4_ASSETS}")
        return

    print("downloading YCB drill mesh...")
    zpath = "_ycb.zip"
    urllib.request.urlretrieve(YCB_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        root = _zip_root(z)
        for member, out in [(f"{root}/meshes/{DRILL}.msh", need[0]),
                            (f"{root}/textures/{DRILL}.png", need[1])]:
            with z.open(member) as src, open(os.path.join(assets, out), "wb") as dst:
                dst.write(src.read())
    os.remove(zpath)
    print(f"  -> {assets}/")


if __name__ == "__main__":
    fetch_ur5e()
    fetch_drill()
    print("\nready. next: python 01_task_error.py --trials 200 --source perfect")

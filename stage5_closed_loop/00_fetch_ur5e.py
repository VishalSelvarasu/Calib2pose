import os
import shutil
import zipfile
import urllib.request

MENAGERIE_URL = ("https://codeload.github.com/google-deepmind/"
                 "mujoco_menagerie/zip/refs/heads/main")
YCB_URL = "https://codeload.github.com/vikashplus/YCB_sim/zip/refs/heads/main"

UR5E_DIR = "ur5e"
STAGE4_ASSETS = os.path.join("..", "stage4_keypoints", "assets")
DRILL = "035_power_drill"


def fetch_ur5e():
    if os.path.exists(os.path.join(UR5E_DIR, "ur5e.xml")):
        print(f"UR5e already present in {UR5E_DIR}/")
        return
    print("downloading MuJoCo Menagerie (large, extracting UR5e only)...")
    zpath = "_menagerie.zip"
    urllib.request.urlretrieve(MENAGERIE_URL, zpath)

    prefix = "mujoco_menagerie-main/universal_robots_ur5e/"
    with zipfile.ZipFile(zpath) as z:
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
        for member, out in [(f"YCB_sim-main/meshes/{DRILL}.msh", need[0]),
                            (f"YCB_sim-main/textures/{DRILL}.png", need[1])]:
            with z.open(member) as src, open(os.path.join(assets, out), "wb") as dst:
                dst.write(src.read())
    os.remove(zpath)
    print(f"  -> {assets}/")


if __name__ == "__main__":
    fetch_ur5e()
    fetch_drill()
    print("\nready. next: python 01_closed_loop.py --trials 200 --source perfect")

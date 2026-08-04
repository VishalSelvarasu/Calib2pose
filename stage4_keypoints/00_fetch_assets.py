import os
import zipfile
import urllib.request

URL = "https://codeload.github.com/vikashplus/YCB_sim/zip/refs/heads/main"
ASSET_DIR = "assets"

OBJECTS = [
    "035_power_drill",
    "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
]


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)

    need = [o for o in OBJECTS
            if not os.path.exists(os.path.join(ASSET_DIR, f"{o}.msh"))]
    if not need:
        print(f"All {len(OBJECTS)} objects already present in {ASSET_DIR}/")
        return

    zpath = os.path.join(ASSET_DIR, "_ycb.zip")
    print(f"downloading YCB assets (~22 MB) for {len(need)} objects...")
    urllib.request.urlretrieve(URL, zpath)

    with zipfile.ZipFile(zpath) as z:
        for obj in need:
            for src, ext in [("meshes", "msh"), ("textures", "png")]:
                member = f"YCB_sim-main/{src}/{obj}.{ext}"
                out = os.path.join(ASSET_DIR, f"{obj}.{ext}")
                try:
                    with z.open(member) as f, open(out, "wb") as g:
                        g.write(f.read())
                except KeyError:
                    print(f"  MISSING in archive: {member}")
            print(f"  {obj}")
    os.remove(zpath)

    n = len([f for f in os.listdir(ASSET_DIR) if f.endswith(".msh")])
    print(f"\n{n} meshes ready in {ASSET_DIR}/")


if __name__ == "__main__":
    main()

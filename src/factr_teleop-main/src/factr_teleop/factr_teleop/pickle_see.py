import pickle
import numpy as np
from pathlib import Path

# ========= 設定 =========
PKL_PATH = Path("/home/watanaberyuto/ws_backup/raw_data/test/ep_00009.pkl")
# ========================


def load_pickle(path: Path):
    print(f"\n[LOAD] {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def print_top_level(data):
    print("\n=== Top-level keys ===")
    for k in data.keys():
        print(" ", k)


def print_topics_summary(data):
    print("\n=== Topics summary ===")
    for topic, values in data["data"].items():
        print(f"{topic:45s} | samples: {len(values)}")


def inspect_topic_values(data, max_print=10):
    print("\n=== Inspect first values per topic ===")
    for topic, values in data["data"].items():
        print(f"\n[Topic] {topic}")
        if len(values) == 0:
            print("  (EMPTY)")
            continue

        for i, v in enumerate(values[:max_print]):
            print(f"  {i}: ", end="")
            if isinstance(v, np.ndarray):
                print(
                    f"type=np.ndarray, shape={v.shape}, "
                    f"dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}"
                )
            else:
                print(f"type={type(v)}, value={v}")


def inspect_timestamps(data):
    print("\n=== Timestamp inspection ===")
    for topic, ts in data["timestamps"].items():
        print(f"\n[Topic] {topic}")
        n = len(ts)
        print(f"  count: {n}")
        if n >= 2:
            diffs = np.diff(ts)
            print(f"  mean dt [ms]: {np.mean(diffs)/1e6:.3f}")
            print(f"  min  dt [ms]: {np.min(diffs)/1e6:.3f}")
            print(f"  max  dt [ms]: {np.max(diffs)/1e6:.3f}")


def inspect_image_sanity(data):
    print("\n=== Image sanity check ===")
    # ZED の topic 名は環境に応じて変えてOK
    image_topics = [t for t in data["data"] if "zed" in t and "im_left" in t]

    if not image_topics:
        print("No image topic found.")
        return

    topic = image_topics[0]
    imgs = data["data"][topic]

    if len(imgs) == 0:
        print(f"{topic} is empty.")
        return

    img = imgs[0]
    print(f"[Topic] {topic}")
    print(" dtype:", img.dtype)
    print(" shape:", img.shape)
    print(" min/max:", img.min(), img.max())


def inspect_episode_overview(data):
    print("\n=== Episode overview ===")
    total_msgs = len(data["all_timestamps"])
    print(f"Total messages (all topics mixed): {total_msgs}")

    if total_msgs >= 2:
        duration = (data["all_timestamps"][-1] - data["all_timestamps"][0]) / 1e9
        print(f"Episode duration [s]: {duration:.2f}")


def main():
    data = load_pickle(PKL_PATH)

    print_top_level(data)
    inspect_episode_overview(data)
    print_topics_summary(data)
    inspect_topic_values(data, max_print=10)
    inspect_timestamps(data)
    inspect_image_sanity(data)

    print("\n=== DONE: pickle looks readable ===")


if __name__ == "__main__":
    main()

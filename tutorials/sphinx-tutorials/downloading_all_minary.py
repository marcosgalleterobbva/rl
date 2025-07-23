import time
from torchrl.data.datasets.minari_data import MinariExperienceReplay

WAIT_SECONDS = 0  # 3 minutes
BATCH_SIZE = 1      # Small batch size to minimize processing
SAVE_ROOT = None    # Set to your desired save location, or None for default


def download_minari_datasets(dataset_id):
    #available_datasets = MinariExperienceReplay.available_datasets
    #print(f"Found {len(available_datasets)} datasets.")

    #for i, dataset_id in enumerate(available_datasets):
    #print(f"[{i + 1}/{len(available_datasets)}] Downloading: {dataset_id}")
    _ = MinariExperienceReplay(
        dataset_id=dataset_id,
        batch_size=BATCH_SIZE,
        root=SAVE_ROOT,
        string_to_tensor_map={"observations/mission": encode_mission_string},
    )
    print(f"✓ Successfully downloaded {dataset_id}")

    #    if i < len(available_datasets) - 1:
    #        print(f"Waiting {WAIT_SECONDS} seconds before next download...")
    #        time.sleep(WAIT_SECONDS)


if __name__ == "__main__":
    # download_minari_datasets()
    # download_minari_datasets("atari/pitfall/expert-v0")
    download_minari_datasets("minigrid/BabyAI-PutNextS7N4/optimal-v0")

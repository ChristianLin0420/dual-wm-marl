import argparse
import json
from m3w.runners.world_model_runner import WorldModelRunner


def main():
    parser = argparse.ArgumentParser(description="M3W training script")
    parser.add_argument(
        "--load-config",
        type=str,
        default="configs/mujoco/installtest/m3w/config.json",
        help="Path to config file"
    )
    parsed = parser.parse_args()

    with open(parsed.load_config, encoding="utf-8") as file:
        all_config = json.load(file)
    args = {
        "algo": all_config["main_args"]["algo"],
        "env": all_config["main_args"]["env"],
        "exp_name": all_config["main_args"]["exp_name"],
    }
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]

    runner = WorldModelRunner(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()

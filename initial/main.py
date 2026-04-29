from argument import args
from synthesize.main import main as synth_main
import json
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Dump args.json next to syn_data so the artifact has a hyperparam record.
    # syn_data_path is "<artifact_dir>/syn_data"; place args.json one level up.
    _args_json_dir = os.path.dirname(os.path.abspath(args.syn_data_path))
    os.makedirs(_args_json_dir, exist_ok=True)
    with open(os.path.join(_args_json_dir, "args.json"), "w") as _f:
        json.dump({k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                   for k, v in vars(args).items()}, _f, indent=2)
    synth_main(args)

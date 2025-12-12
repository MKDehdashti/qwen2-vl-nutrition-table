# to test HF eval and compare with vllm one
import os, json, argparse, yaml
from .dataset.data_utils import get_datasets
from .model_utils import load_model
from .eval_utils import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="outputs/hf_eval_debug")
    parser.add_argument("--name", type=str, default="hf_eval_debug")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model, processor = load_model(
        model_id=cfg["model_id"],
        dtype=None,
        quantized=False,
        use_adapters=True,
        from_dir=args.from_dir,
        cfg=cfg,
    )

    train, val = get_datasets(format_data_flag=True)
    ds = val

    dump_path = os.path.join(args.out_dir, f"per_sample_{args.name}.json")
    stats = evaluate_model(
        model,
        processor,
        n=args.n,
        dataset=ds,
        tag=args.name,
        cfg=cfg,
        training_args=type("T", (), {"output_dir": args.out_dir}),
        plot=False,
        dump_path=dump_path,
    )

    print("HF eval stats:", stats)
    print("per-sample saved to:", dump_path)

if __name__ == "__main__":
    main()

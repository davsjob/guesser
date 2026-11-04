"""
Command line interface.

    guesser predict --pdb 1abc.pdb --mutations A:L45F A:G50A+A:L45F --out preds.csv
    guesser predict --csv mutations.csv --pdb-dir structures/ --out preds.csv
    guesser predict --graphs graphs.pt --out preds.csv
    guesser build-graphs --table Megascale-S-train.xlsx Megascale-D-train.xlsx --pdb-dir pdbs/ --out train.pt
    guesser train --train train.pt --val val.pt --seed 42 --out-dir runs/ --name guesser_S42
    guesser explain ...              (same arguments as python -m guesser.explain.run)
    guesser merge-explanations ...   (same arguments as python -m guesser.explain.merge)
"""
import argparse
import dataclasses
import os
import sys
import time


def _load_graphs(path):
    import torch
    return torch.load(path, weights_only=False)


def cmd_build_graphs(args):
    import torch
    from .graphs import build_graphs, check_graphs, load_table

    df = load_table(args.table)
    if args.num_shards > 1:
        df = df.iloc[args.shard_id::args.num_shards]  # keeps table row labels for merging
    print(f"{len(df)} mutations, {df['pdb'].nunique()} structures from {args.pdb_dir} | radius {args.radius} A | "
          f"edges {args.edge_cutoff} A", flush=True)
    t0 = time.time()

    def progress(done, failed, total):
        if (done + failed) % 2000 == 0 or done + failed == total:
            print(f"  {done + failed}/{total} (failed {failed}) | {(time.time() - t0) / 60:.1f} min", flush=True)

    out = build_graphs(df, args.pdb_dir, radii=[args.radius], edge_cutoff=args.edge_cutoff,
                       device=args.device, progress_callback=progress)
    graphs = out[float(args.radius)]
    check_graphs(graphs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(graphs, args.out)
    print(f"built {len(graphs)}/{len(df)} graph pairs -> {args.out}")


def cmd_train(args):
    from .training import TrainConfig, train

    overrides = {f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)
                 if getattr(args, f.name, None) is not None}
    cfg = TrainConfig(**overrides)
    name = args.name or f"model_S{cfg.seed}"
    best = train(_load_graphs(args.train), _load_graphs(args.val), cfg, out_dir=args.out_dir, name=name,
                 device=args.device)
    print(f"best checkpoint: {best}")


def cmd_predict(args):
    from .inference import released_weights, metrics, predict

    df = None
    if args.graphs:
        graphs = _load_graphs(args.graphs)
    else:
        from .graphs import build_graphs, check_graphs, mutation_table, read_mutation_csv
        if args.csv:
            df = read_mutation_csv(args.csv, default_chain=args.chain)
            pdb_dir = args.pdb_dir or os.path.dirname(os.path.abspath(args.csv))
        elif args.pdb and args.mutations:
            pdb_dir, pdb_file = os.path.split(os.path.abspath(args.pdb))
            df = mutation_table(pdb_file, args.mutations, default_chain=args.chain)
        else:
            sys.exit("give --csv (with --pdb-dir), --pdb with --mutations, or --graphs")
        graphs = build_graphs(df, pdb_dir, radii=[9.0], edge_cutoff=9.0, device=args.device)[9.0]
        check_graphs(graphs)
        built = {it["row"] for it in graphs}
        for row in df.index.difference(sorted(built)):
            print(f"WARNING: could not build {df.loc[row, 'pdb']} {df.loc[row, 'input']} (check file, chain and position)")

    weights = args.weights or released_weights()
    ens, members = predict(graphs, weights, device=args.device, batch_size=args.batch_size, return_members=True)
    if df is not None:   # keep the user's own mutation strings next to the parsed ones
        ens.insert(2, "input", [df.loc[it["row"], "input"] for it in graphs])
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        ens.to_csv(args.out, index=False)
        print(f"ensemble predictions ({len(weights)} members) -> {args.out}")
    if args.member_dir:
        os.makedirs(args.member_dir, exist_ok=True)
        for w, m in zip(weights, members):
            m.to_csv(os.path.join(args.member_dir, os.path.basename(w).replace('.pt', '_predictions.csv')), index=False)
    show = ens[['pdb', 'input' if 'input' in ens else 'mutation', 'pred_ddg', 'intra_epistemic_std', 'calibrated_std']]
    print(show.head(args.show).to_string(index=False))
    if ens['true_ddg'].notna().sum() >= 3 and ens['true_ddg'].abs().sum() > 0:   # measured ddG given
        print(metrics(ens.dropna(subset=['true_ddg']), sigma='intra_epistemic_std'))


def main(argv=None):
    p = argparse.ArgumentParser(prog="guesser", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-graphs", help="structures + mutation table -> graph pairs (.pt); needs MadraX")
    b.add_argument("--table", nargs="+", required=True, help="CSV/XLSX with pdb, mut_list, ddg (concatenated in order)")
    b.add_argument("--pdb-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--radius", type=float, default=9.0)
    b.add_argument("--edge-cutoff", type=float, default=9.0)
    b.add_argument("--shard-id", type=int, default=0)
    b.add_argument("--num-shards", type=int, default=1, help="process rows shard_id::num_shards")
    b.add_argument("--device", default=None)
    b.set_defaults(func=cmd_build_graphs)

    t = sub.add_parser("train", help="train one ensemble member (defaults = released model)")
    t.add_argument("--train", required=True, help="training graphs (.pt)")
    t.add_argument("--val", required=True, help="validation graphs (.pt)")
    t.add_argument("--out-dir", default=".")
    t.add_argument("--name", default=None)
    t.add_argument("--device", default=None)
    from .training import TrainConfig
    for f in dataclasses.fields(TrainConfig):
        kind = {"int": int, "float": float, "str": str, "bool": lambda s: s.lower() in ("1", "true", "yes")}[f.type if isinstance(f.type, str) else f.type.__name__]
        t.add_argument(f"--{f.name}", type=kind, default=None, help=f"default {f.default}")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="ensemble ddG + uncertainty")
    pr.add_argument("--graphs", help="graph pairs (.pt) from build-graphs")
    pr.add_argument("--csv", help="mutation list with columns pdb, mutation[, chain, ddg] (needs MadraX)")
    pr.add_argument("--pdb-dir", help="folder with the structures named in --csv (default: the CSV's folder)")
    pr.add_argument("--pdb", help="a PDB file (needs MadraX to build graphs)")
    pr.add_argument("--mutations", nargs="+", help="e.g. A:L45F  L45F  A:L45F+A:G50A")
    pr.add_argument("--chain", default="A", help="chain for mutations given without one")
    pr.add_argument("--weights", nargs="+", default=None, help="checkpoints (default: the 8 released members)")
    pr.add_argument("--out", default=None, help="ensemble CSV")
    pr.add_argument("--member-dir", default=None, help="also write per-member CSVs (Forward + Reverse rows)")
    pr.add_argument("--batch-size", type=int, default=64)
    pr.add_argument("--device", default=None)
    pr.add_argument("--show", type=int, default=20)
    pr.set_defaults(func=cmd_predict)

    for name, module in (("explain", "run"), ("merge-explanations", "merge")):
        e = sub.add_parser(name, add_help=False, help=f"see python -m guesser.explain.{module} --help")
        e.set_defaults(func=None, module=module)

    args, rest = p.parse_known_args(argv)
    if args.func is None:
        from importlib import import_module
        return import_module(f"guesser.explain.{args.module}").main(rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    args.func(args)


if __name__ == "__main__":
    main()

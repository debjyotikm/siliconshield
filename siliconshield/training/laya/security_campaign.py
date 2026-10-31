"""Multi-seed SiliconShield-RTL training, selection, export eval."""

import argparse
import copy
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from siliconshield.training.laya import security_train as training

# Version of the final evaluation report (report.json) written after test
# evaluation. Distinct from the trainer development report
# (security_train.REPORT_SCHEMA); enforced by tests/test_security_training.py.
FINAL_REPORT_SCHEMA = 3


def run_process(command, log_path, environment, timeout):
    with Path(log_path).open("a") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return child.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124


def temperature_for(config):
    return config["temperature_by_options"].get(
        "choice:3-5", config["temperature"][0]
    )


def fit_temperature(logits, targets):
    from laya.calibrate import fit_temperature_map

    return fit_temperature_map(
        [(0, row, np.eye(3)[target], 3) for row, target in zip(logits, targets)]
    )


def grouped_calibration(rows, logits, targets):
    families = sorted({row["family_id"] for row in rows})
    if len(families) < 4:
        raise ValueError("grouped calibration requires at least four families")
    folds = {family: index % 2 for index, family in enumerate(families)}
    scaled = np.zeros_like(np.asarray(logits, dtype=np.float64))
    fold_reports = []
    for fold in (0, 1):
        held = [
            i for i, row in enumerate(rows) if folds[row["family_id"]] == fold
        ]
        fit_indices = [i for i in range(len(rows)) if i not in set(held)]
        if not held or not fit_indices:
            raise ValueError("calibration fold is empty")
        fitted = fit_temperature(
            [logits[i] for i in fit_indices], [targets[i] for i in fit_indices]
        )
        temperature = fitted["temperature_by_options"].get(
            "choice:3-5", fitted["temperature"][0]
        )
        scaled[held] = np.asarray(logits)[held] / temperature
        fold_metrics = training.metrics(
            scaled[held], [targets[i] for i in held]
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_families": sorted(
                    {rows[i]["family_id"] for i in fit_indices}
                ),
                "validation_families": sorted(
                    {rows[i]["family_id"] for i in held}
                ),
                "fit_n": len(fit_indices),
                "validation_n": len(held),
                "temperature": temperature,
                "temperature_clamped": fitted["temperature_clamped"],
                "validation_metrics": fold_metrics,
            }
        )
    raw = training.metrics(logits, targets)
    cross_validated = training.metrics(scaled, targets)
    fitted = fit_temperature(logits, targets)
    validation = {
        "method": "two_fold_family_grouped_temperature_validation",
        "family_count": len(families),
        "raw": raw,
        "cross_validated": cross_validated,
        "folds": fold_reports,
        "final_fit_n": len(rows),
        "final_temperature_clamped": fitted["temperature_clamped"],
    }
    return fitted, validation


def validate_training_report(output, run_id, seed, dev_hash, config):
    output = Path(output)
    report_path = (
        output / "training_report.json"
        if (output / "training_report.json").exists()
        else output / "report.json"
    )
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != 2
        or report.get("run_id") != run_id
        or report.get("final_evaluation") is not False
        or report.get("test_evaluated")
    ):
        raise ValueError("invalid or stale training report")
    if report["progress"]["step"] < 0 or not report["progress"].get(
        "stop_reason"
    ):
        raise ValueError("training did not reach a terminal state")
    signature = report["signature"]
    if signature["seed"] != seed or signature["data"]["dev"] != dev_hash:
        raise ValueError("seed or development provenance mismatch")
    pointer, best, weights_path = training.load_best(output)
    if (
        report["selection_key"] != best["key"]
        or report["development"] != best["metrics"]
    ):
        raise ValueError("report selection does not match committed best")
    if (
        report["selected_checkpoint"]["epoch"] != best["epoch"]
        or report["selected_checkpoint"]["step"] != best["step"]
    ):
        raise ValueError("report selected epoch or step mismatch")
    if report["progress"]["step"] < best["step"]:
        raise ValueError("selected step exceeds completed training")
    if report["development_acceptance"] != training.acceptance_status(
        best["metrics"]
    ):
        raise ValueError("development acceptance evidence mismatch")
    if (
        training.sha256(Path(config["data"]) / "train.jsonl")
        != signature["data"]["train"]
    ):
        raise ValueError("training data provenance mismatch")
    expected_fields = {
        "world": config.get("world_size", 3),
        "micro_batch": config.get("micro_batch", 4),
        "accumulation": config.get("accumulation", 4),
        "max_len": config.get("max_len", 1024),
        "head_max_len": config.get("head_max_len", 192),
        "encoder_lr": config.get("encoder_lr", 1e-5),
        "head_lr": config.get("head_lr", 1e-4),
        "warmup": config.get("warmup", 60),
        "min_updates": config.get("min_updates", 600),
        "patience": config.get("patience", 4),
        "checkpointing": config.get("checkpointing", False),
        "checkpoint_interval": config.get("checkpoint_interval", 900),
        "sampling": config.get("sampling", "weighted"),
        "eval_every_updates": config.get("eval_every_updates", 0),
        "max_updates": config.get("max_updates", 0),
        "source_mass": training.SOURCE_MASS,
        "acceptance": training.ACCEPTANCE,
    }
    for name, expected in expected_fields.items():
        if signature.get(name) != expected:
            raise ValueError("training signature mismatch: " + name)
    if (
        signature["code_manifest"]["combined_sha256"]
        != training.code_manifest()["combined_sha256"]
    ):
        raise ValueError("training code provenance mismatch")
    if signature["initial_checkpoint"] != training.checkpoint_manifest(
        config["model"]
    ):
        raise ValueError("initial checkpoint provenance mismatch")
    if (
        pointer["weights_sha256"]
        != report["selected_checkpoint"]["weights_sha256"]
    ):
        raise ValueError("selected checkpoint mismatch")
    export = json.loads((output / "export.json").read_text())
    if (
        export["weights_sha256"] != pointer["weights_sha256"]
        or training.sha256(output / "model/model.safetensors")
        != pointer["weights_sha256"]
    ):
        raise ValueError("training export mismatch")
    if report["export"]["export_manifest"] != training.checkpoint_manifest(
        output / "model"
    ):
        raise ValueError("training export file set changed")
    return report


def train_seed(config, state, seed, environment, logs):
    output = Path(config["root"]) / "runs" / ("seed-%03d" % seed)
    output.mkdir(parents=True, exist_ok=True)
    dev_hash = training.sha256(Path(config["data"]) / "dev.jsonl")
    if (output / "report.json").exists():
        return validate_training_report(
            output, config["run_id"], seed, dev_hash, config
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=" + str(config.get("world_size", 3)),
        str(Path(__file__).with_name("security_train.py")),
        "--model",
        config["model"],
        "--data",
        config["data"],
        "--output",
        str(output),
        "--run-id",
        config["run_id"],
        "--max-len",
        str(config.get("max_len", 1024)),
        "--head-max-len",
        str(config.get("head_max_len", 192)),
        "--micro-batch",
        str(config.get("micro_batch", 4)),
        "--accumulation",
        str(config.get("accumulation", 4)),
        "--epochs",
        str(config.get("epochs", 60)),
        "--hours",
        str(config["seed_hours"]),
        "--seed",
        str(seed),
        "--encoder-lr",
        str(config.get("encoder_lr", 1e-5)),
        "--head-lr",
        str(config.get("head_lr", 1e-4)),
        "--warmup",
        str(config.get("warmup", 60)),
        "--min-updates",
        str(config.get("min_updates", 600)),
        "--patience",
        str(config.get("patience", 4)),
        "--sampling",
        config.get("sampling", "weighted"),
        "--eval-every-updates",
        str(config.get("eval_every_updates", 0)),
        "--max-updates",
        str(config.get("max_updates", 0)),
        "--checkpoint-interval",
        str(config.get("checkpoint_interval", 900)),
        "--no-final-evaluation",
    ]
    if not config.get("checkpointing", False):
        command.append("--no-checkpointing")
    if (output / "latest.pt").exists():
        command.append("--resume")
    log_path = logs / ("train-%03d.log" % seed)
    print(
        json.dumps(
            {
                "event": "training_seed",
                "seed": seed,
                "output": str(output),
                "resume": "--resume" in command,
            }
        ),
        flush=True,
    )
    for attempt in range(2):
        code = run_process(
            command,
            log_path,
            environment,
            min(
                state["deadline"] - time.time(),
                config["seed_hours"] * 3600 + 1800,
            ),
        )
        if code == 0:
            return validate_training_report(
                output, config["run_id"], seed, dev_hash, config
            )
        tail = log_path.read_text()[-32000:]
        transient = any(
            value in tail
            for value in (
                "NCCL error",
                "ncclRemoteError",
                "Connection reset by peer",
                "Watchdog caught collective operation timeout",
            )
        )
        if not transient or attempt or not (output / "latest.pt").exists():
            raise RuntimeError("training seed %d exited %d" % (seed, code))
        command = command if "--resume" in command else command + ["--resume"]
    raise RuntimeError("unreachable seed retry state")


def cache_valid(payload, identity):
    return all(payload.get(key) == value for key, value in identity.items())


def sdk_order(agent, rows, output, split, order, identity, batch_size):
    output = Path(output)
    folder = output / "finalization"
    folder.mkdir(exist_ok=True)
    name = "%s-%s.json" % (split, "".join(str(value) for value in order))
    path = folder / name
    expected_identity = dict(
        identity,
        split=split,
        order=list(order),
        sort_by_length=False,
        row_ids=[row["id"] for row in rows],
    )
    if path.exists():
        payload = json.loads(path.read_text())
        if not cache_valid(payload, expected_identity):
            raise ValueError(
                "cached SDK order provenance mismatch: " + str(path)
            )
        return payload
    questions = copy.deepcopy(rows[0]["questions"])
    if any(row["questions"] != rows[0]["questions"] for row in rows):
        raise ValueError("question schema differs within " + split)
    questions["security"]["option_order"] = list(order)
    answers = agent.predict_batch(
        [row["state"] for row in rows],
        questions,
        batch_size=batch_size,
        sort_by_length=False,
    )
    if len(answers) != len(rows):
        raise ValueError("SDK returned misaligned rows")
    probabilities = []
    usage = []
    for index, (row, answer) in enumerate(zip(rows, answers)):
        if answer["usage"].get("truncated"):
            raise ValueError("SDK truncation for " + row["id"])
        values = answer["answers"]["security"]["probabilities"]
        probabilities.append(
            [float(values[label]) for label in training.LABELS]
        )
        usage.append(answer["usage"])
    payload = dict(
        expected_identity,
        probabilities=probabilities,
        usage=usage,
        probability_source="exported_sdk_rounded_4dp",
    )
    training.write_json(path, payload)
    return payload


def sdk_split(agent, rows, output, split, identity, batch_size):
    orders = [(0, 1, 2)] + [
        order
        for order in itertools.permutations(range(3))
        if order != (0, 1, 2)
    ]
    artifacts = [
        sdk_order(agent, rows, output, split, order, identity, batch_size)
        for order in orders
    ]
    targets = [
        training.LABELS.index(row["expected"]["security"]) for row in rows
    ]
    canonical = artifacts[0]["probabilities"]
    result = training.metrics(canonical, targets, from_probabilities=True)
    diagnostics = training.diagnostics(rows, canonical, from_probabilities=True)
    predictions = np.argmax(np.asarray(canonical), axis=1)
    stable = np.ones(len(rows), dtype=bool)
    maximum_delta = np.zeros(len(rows), dtype=np.float64)
    per_order = []
    for order, artifact in zip(orders, artifacts):
        probabilities = np.asarray(artifact["probabilities"])
        stable &= np.argmax(probabilities, axis=1) == predictions
        maximum_delta = np.maximum(
            maximum_delta,
            np.abs(probabilities - np.asarray(canonical)).max(axis=1),
        )
        per_order.append(
            {
                "order": list(order),
                "metrics": training.metrics(
                    probabilities.tolist(), targets, from_probabilities=True
                ),
            }
        )
    audit = {
        "rows": len(rows),
        "orders": len(orders),
        "all_rows_all_orders_evaluated": True,
        "stable_fraction": float(stable.mean()),
        "unstable_row_ids": [rows[i]["id"] for i in np.flatnonzero(~stable)],
        "max_probability_delta": float(maximum_delta.max()),
        "max_probability_delta_row_id": rows[int(maximum_delta.argmax())]["id"],
        "per_order": per_order,
    }
    return {
        "metrics": result,
        "diagnostics": diagnostics,
        "probabilities": canonical,
        "targets": targets,
        "option_order": audit,
    }


def compare_gpu_probabilities(raw_logits, sdk_probabilities, temperature):
    raw = np.asarray(raw_logits, dtype=np.float64) / temperature
    raw -= raw.max(axis=1, keepdims=True)
    expected = np.exp(raw)
    expected /= expected.sum(axis=1, keepdims=True)
    actual = np.asarray(sdk_probabilities, dtype=np.float64)
    delta = float(np.abs(actual - expected).max())
    if not np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)):
        raise ValueError("SDK and internal GPU argmax differ")
    if delta > 0.00025:
        raise ValueError(
            "SDK and internal GPU probabilities exceed BF16/rounding tolerance"
        )
    return {
        "n": len(actual),
        "max_absolute_probability_difference": delta,
        "atol": 0.00025,
        "argmax_match": True,
    }


def prepare_final_model(output, cfg, fitted, validation, expected_weights):
    output = Path(output)
    final_model = output / "final_model"
    expected_cfg = dict(
        cfg,
        training_calibration=fitted,
        grouped_calibration_validation=validation,
        outer_test_evaluated=True,
    )
    if final_model.exists():
        if (
            training.sha256(final_model / "model.safetensors")
            != expected_weights
        ):
            raise ValueError("final model weights changed")
        existing = json.loads(
            (final_model / "rl_agent_config.json").read_text()
        )
        if existing != expected_cfg:
            raise ValueError(
                "final model config or calibration lineage differs"
            )
        return final_model
    temporary = output / "final_model.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(output / "model", temporary)
    training.write_json(temporary / "rl_agent_config.json", expected_cfg)
    if training.sha256(temporary / "model.safetensors") != expected_weights:
        raise ValueError("calibrated export weights mismatch")
    os.replace(temporary, final_model)
    training.fsync_directory(output)
    return final_model


def cpu_parity(model, tok, items, rows, args, final_model, temperature):
    from laya.agent import load

    model.cpu().eval()
    reloaded = load(str(final_model), device="cpu")
    batch = training.batch_for(
        tok, items, list(range(min(3, len(items)))), args
    )
    with torch.inference_mode():
        old = training.forward(model, batch, torch.device("cpu"), False)
        new = training.forward(
            reloaded.model, batch, torch.device("cpu"), False
        )
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    for index, row in enumerate(rows[:3]):
        single = training.batch_for(tok, items, [index], args)
        with torch.inference_mode():
            logits = training.forward(
                reloaded.model, single, torch.device("cpu"), False
            )[0].numpy()
        answer = reloaded.system_one(row["state"], row["questions"])
        actual = [
            answer["answers"]["security"]["probabilities"][label]
            for label in training.LABELS
        ]
        training.assert_sdk_probabilities(actual, logits, temperature)
        if answer["usage"].get("truncated"):
            raise ValueError("CPU SDK truncation")
    del reloaded
    return {
        "n": min(3, len(rows)),
        "exact_fp32_logits": True,
        "exact_four_decimal_probabilities": True,
    }


def finalize_only(args):
    output = Path(args.output)
    final_report = output / "report.json"
    training_report_path = output / "training_report.json"
    if training_report_path.exists():
        saved_training = json.loads(training_report_path.read_text())
    else:
        saved_training = json.loads(final_report.read_text())
        if saved_training.get(
            "final_evaluation"
        ) is not False or saved_training.get("test_evaluated"):
            raise ValueError("no immutable training report for finalization")
        training.write_json(training_report_path, saved_training)
    if (
        saved_training.get("run_id") != args.run_id
        or saved_training.get("final_evaluation") is not False
    ):
        raise ValueError("finalization training lineage mismatch")
    selected = saved_training["selected_checkpoint"]
    args.max_len = saved_training["signature"]["max_len"]
    args.head_max_len = saved_training["signature"]["head_max_len"]
    args.micro_batch = saved_training["signature"]["micro_batch"]
    weights_hash = selected["weights_sha256"]
    if training.sha256(output / "model/model.safetensors") != weights_hash:
        raise ValueError(
            "selected model for finalization does not match training receipt"
        )
    rows = training.load_splits(args.data, final=True)
    saved_hashes = saved_training["signature"]["data"]
    for split in ("train", "dev"):
        if (
            training.sha256(Path(args.data) / (split + ".jsonl"))
            != saved_hashes[split]
        ):
            raise ValueError(
                "training data changed before finalization: " + split
            )
    manifest_path = Path(args.data) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        split_hashes = manifest.get("split_sha256", {})
        for split, expected in split_hashes.items():
            if (
                training.sha256(Path(args.data) / (split + ".jsonl"))
                != expected
            ):
                raise ValueError("final dataset hash mismatch: " + split)
    if final_report.exists() and json.loads(final_report.read_text()).get(
        "final_evaluation"
    ):
        expected_rows = (
            len(rows["calibration"])
            + len(rows["test"])
            + len(rows.get("external", []))
        )
        return validate_final_report(
            output, args.run_id, weights_hash, expected_rows, args.data
        )
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    model, tok, cfg = training.load_checkpoint(str(output / "model"))
    items = {
        split: training.prepare_items(
            tok, split_rows, args.max_len, args.head_max_len
        )
        for split, split_rows in rows.items()
    }
    raw_path = output / "evaluation.json"
    raw_receipt = (
        json.loads(raw_path.read_text()) if raw_path.exists() else None
    )
    raw_identity = {
        "run_id": args.run_id,
        "weights_sha256": weights_hash,
        "code_sha256": training.code_manifest()["combined_sha256"],
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "micro_batch": args.micro_batch,
        "source_export_manifest": training.checkpoint_manifest(
            output / "model"
        ),
        "final_data_manifest_sha256": training.sha256(manifest_path)
        if manifest_path.exists()
        else None,
    }
    if raw_receipt is not None and not cache_valid(raw_receipt, raw_identity):
        raise ValueError("cached raw evaluation provenance mismatch")
    if raw_receipt is not None and not (output / "test_touch.json").exists():
        raise ValueError("raw test cache lacks its test-touch lineage marker")
    if raw_receipt is None:
        model.to(device).eval()
        calibration_baselines = training.trivial_baselines(rows["calibration"])
        calibration_raw, cal_logits, cal_y = training.evaluate(
            model,
            tok,
            items["calibration"],
            args,
            device,
            0,
            1,
            baseline_cost=calibration_baselines["best_selection_cost"],
        )
        fitted, calibration_validation = grouped_calibration(
            rows["calibration"], cal_logits, cal_y
        )
        cfg.update(
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
        )
        temperature = temperature_for(cfg)
        training.write_json(
            output / "test_touch.json",
            {
                "event": "outer_test_accessed",
                "run_id": args.run_id,
                "weights_sha256": weights_hash,
                "time": time.time(),
                "data_manifest_sha256": raw_identity[
                    "final_data_manifest_sha256"
                ],
            },
        )
        test_raw, test_logits, test_y = training.evaluate(
            model,
            tok,
            items["test"],
            args,
            device,
            0,
            1,
            baseline_cost=training.trivial_baselines(rows["test"])[
                "best_selection_cost"
            ],
        )
        external_raw = None
        ext_logits = ext_y = None
        if "external" in rows:
            external_raw, ext_logits, ext_y = training.evaluate(
                model,
                tok,
                items["external"],
                args,
                device,
                0,
                1,
                baseline_cost=training.trivial_baselines(rows["external"])[
                    "best_selection_cost"
                ],
            )
        raw_receipt = dict(
            raw_identity,
            temperature=temperature,
            fitted=fitted,
            calibration_raw=calibration_raw,
            calibration_logits=cal_logits,
            calibration_targets=cal_y,
            grouped_calibration=calibration_validation,
            test_raw=test_raw,
            test_logits=test_logits,
            test_targets=test_y,
            external_raw=external_raw,
            external_logits=ext_logits,
            external_targets=ext_y,
        )
        training.write_json(raw_path, raw_receipt)
        training.write_json(
            output / "finalization_state.json",
            {
                "event": "raw_evaluation_complete",
                "run_id": args.run_id,
                "weights_sha256": weights_hash,
                "temperature": temperature,
            },
        )
    fitted = raw_receipt["fitted"]
    calibration_raw = raw_receipt["calibration_raw"]
    cal_logits, cal_y = (
        raw_receipt["calibration_logits"],
        raw_receipt["calibration_targets"],
    )
    calibration_validation = raw_receipt["grouped_calibration"]
    test_raw = raw_receipt["test_raw"]
    test_logits, test_y = (
        raw_receipt["test_logits"],
        raw_receipt["test_targets"],
    )
    external_raw = raw_receipt["external_raw"]
    ext_logits, ext_y = (
        raw_receipt["external_logits"],
        raw_receipt["external_targets"],
    )
    cfg.update(
        temperature=fitted["temperature"],
        temperature_by_options=fitted["temperature_by_options"],
    )
    temperature = temperature_for(cfg)
    final_model = prepare_final_model(
        output, cfg, fitted, calibration_validation, weights_hash
    )
    parity = cpu_parity(
        model, tok, items["test"], rows["test"], args, final_model, temperature
    )
    del model
    torch.cuda.empty_cache()
    from laya.agent import load

    agent = load(str(final_model), device="cuda:0")
    identity = {
        "run_id": args.run_id,
        "weights_sha256": weights_hash,
        "temperature": temperature,
        "final_model_config_sha256": training.sha256(
            final_model / "rl_agent_config.json"
        ),
        "code_sha256": training.code_manifest()["combined_sha256"],
    }
    calibration_sdk = sdk_order(
        agent,
        rows["calibration"],
        output,
        "calibration",
        (0, 1, 2),
        identity,
        args.micro_batch,
    )
    calibration_parity = compare_gpu_probabilities(
        cal_logits, calibration_sdk["probabilities"], temperature
    )
    test_sdk = sdk_split(
        agent, rows["test"], output, "test", identity, args.micro_batch
    )
    test_parity = compare_gpu_probabilities(
        test_logits, test_sdk["probabilities"], temperature
    )
    external_sdk = None
    external_parity = None
    if "external" in rows:
        external_sdk = sdk_split(
            agent,
            rows["external"],
            output,
            "external",
            identity,
            args.micro_batch,
        )
        external_parity = compare_gpu_probabilities(
            ext_logits, external_sdk["probabilities"], temperature
        )
    test_baselines = training.trivial_baselines(rows["test"])
    external_baselines = (
        training.trivial_baselines(rows["external"])
        if "external" in rows
        else None
    )
    report = {
        "schema": FINAL_REPORT_SCHEMA,
        "run_id": args.run_id,
        "development": saved_training["development"],
        "development_acceptance": saved_training["development_acceptance"],
        "selection_key": saved_training["selection_key"],
        "selected_checkpoint": selected,
        "progress": saved_training["progress"],
        "calibration_raw": calibration_raw,
        "grouped_calibration": calibration_validation,
        "test_raw": test_raw,
        "test_calibrated": test_sdk["metrics"],
        "test_diagnostics_raw": training.diagnostics(rows["test"], test_logits),
        "test_diagnostics_calibrated": test_sdk["diagnostics"],
        "test_logits": test_logits,
        "test_targets": test_y,
        "test_sdk_probabilities": test_sdk["probabilities"],
        "test_baselines": test_baselines,
        "option_order": test_sdk["option_order"],
        "external_evaluation": external_sdk,
        "external_raw": external_raw,
        "external_baselines": external_baselines,
        "cpu_sdk_parity": parity,
        "calibration_gpu_parity": calibration_parity,
        "test_gpu_parity": test_parity,
        "external_gpu_parity": external_parity,
        "sdk_calibrated_probabilities_match": True,
        "sdk_validation_n": len(rows["calibration"])
        + len(rows["test"])
        + len(rows.get("external", [])),
        "final_evaluation": True,
        "test_evaluated": True,
        "export": {
            "path": str(final_model),
            "weights_sha256": weights_hash,
            "manifest": training.checkpoint_manifest(final_model),
        },
        "training_report_sha256": training.sha256(training_report_path),
        "final_data_manifest_sha256": training.sha256(manifest_path)
        if manifest_path.exists()
        else None,
    }
    training.write_json(output / "report.json", report)
    training.write_json(
        output / "finalization_state.json",
        {
            "event": "complete",
            "run_id": args.run_id,
            "weights_sha256": weights_hash,
            "report_sha256": training.sha256(output / "report.json"),
        },
    )
    training.write_json(
        output / "status.json",
        {
            "event": "complete",
            "run_id": args.run_id,
            "test": report["test_calibrated"],
            "final_evaluation": True,
        },
    )
    print(
        json.dumps(
            {
                "event": "final_evaluation_complete",
                "run_id": args.run_id,
                "test": report["test_calibrated"],
                "export": str(final_model),
            }
        ),
        flush=True,
    )


def validate_sdk_artifacts(
    output, rows, split, run_id, weights_hash, final_model
):
    output = Path(output)
    config_hash = training.sha256(Path(final_model) / "rl_agent_config.json")
    canonical = None
    for order in itertools.permutations(range(3)):
        path = (
            output
            / "finalization"
            / ("%s-%s.json" % (split, "".join(str(value) for value in order)))
        )
        payload = json.loads(path.read_text())
        expected = {
            "run_id": run_id,
            "weights_sha256": weights_hash,
            "split": split,
            "order": list(order),
            "row_ids": [row["id"] for row in rows],
            "final_model_config_sha256": config_hash,
        }
        if not cache_valid(payload, expected):
            raise ValueError(
                "final SDK artifact provenance mismatch: " + str(path)
            )
        if len(payload["probabilities"]) != len(rows):
            raise ValueError("final SDK artifact row count mismatch")
        if order == (0, 1, 2):
            canonical = payload["probabilities"]
    return canonical


def validate_final_report(
    output, run_id, expected_weights, expected_rows, data_dir
):
    output = Path(output)
    data_dir = Path(data_dir)
    report = json.loads((output / "report.json").read_text())
    required = {
        "test_sdk_probabilities",
        "test_targets",
        "test_logits",
        "grouped_calibration",
        "cpu_sdk_parity",
        "test_gpu_parity",
        "calibration_gpu_parity",
        "test_diagnostics_calibrated",
        "test_diagnostics_raw",
        "training_report_sha256",
        "final_data_manifest_sha256",
    }
    if report.get("schema") != FINAL_REPORT_SCHEMA or not required <= set(
        report
    ):
        raise ValueError(
            "final report schema or mandatory evidence is incomplete"
        )
    if (
        report.get("run_id") != run_id
        or not report.get("final_evaluation")
        or not report.get("test_evaluated")
    ):
        raise ValueError("final report is incomplete")
    if report["selected_checkpoint"]["weights_sha256"] != expected_weights:
        raise ValueError("final report selected the wrong weights")
    if report["training_report_sha256"] != training.sha256(
        output / "training_report.json"
    ):
        raise ValueError("immutable training report hash mismatch")
    manifest_path = data_dir / "manifest.json"
    if report["final_data_manifest_sha256"] != training.sha256(manifest_path):
        raise ValueError("final dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    for split, expected in manifest.get("split_sha256", {}).items():
        if training.sha256(data_dir / (split + ".jsonl")) != expected:
            raise ValueError("final split hash mismatch: " + split)
    final_model = Path(report["export"]["path"])
    if (
        final_model != output / "final_model"
        or report["export"]["weights_sha256"] != expected_weights
    ):
        raise ValueError("final export path or weights mismatch")
    if report["export"]["manifest"] != training.checkpoint_manifest(
        final_model
    ):
        raise ValueError("final export manifest mismatch")
    rows = training.load_splits(data_dir, final=True)
    test_probabilities = validate_sdk_artifacts(
        output, rows["test"], "test", run_id, expected_weights, final_model
    )
    test_targets = [
        training.LABELS.index(row["expected"]["security"])
        for row in rows["test"]
    ]
    if (
        report["test_targets"] != test_targets
        or report["test_sdk_probabilities"] != test_probabilities
    ):
        raise ValueError("stored test evidence is misaligned")
    recomputed = training.metrics(
        test_probabilities, test_targets, from_probabilities=True
    )
    if report["test_calibrated"] != recomputed:
        raise ValueError("test metrics do not reproduce from SDK probabilities")
    if (
        report["option_order"]["rows"] != len(rows["test"])
        or report["option_order"]["orders"] != 6
    ):
        raise ValueError("test option-order coverage is incomplete")
    if (
        not report["option_order"]["all_rows_all_orders_evaluated"]
        or len(report["option_order"]["per_order"]) != 6
    ):
        raise ValueError("test option-order evidence is incomplete")
    external = report.get("external_evaluation")
    if "external" in rows:
        if external is None:
            raise ValueError("external evidence is missing")
        external_probabilities = validate_sdk_artifacts(
            output,
            rows["external"],
            "external",
            run_id,
            expected_weights,
            final_model,
        )
        external_targets = [
            training.LABELS.index(row["expected"]["security"])
            for row in rows["external"]
        ]
        if (
            external["targets"] != external_targets
            or external["probabilities"] != external_probabilities
        ):
            raise ValueError("stored external evidence is misaligned")
        if external["metrics"] != training.metrics(
            external_probabilities, external_targets, from_probabilities=True
        ):
            raise ValueError(
                "external metrics do not reproduce from SDK probabilities"
            )
        if (
            external["option_order"]["rows"] != len(rows["external"])
            or external["option_order"]["orders"] != 6
            or not external["option_order"]["all_rows_all_orders_evaluated"]
            or len(external["option_order"]["per_order"]) != 6
        ):
            raise ValueError("external option-order evidence is incomplete")
    elif external is not None:
        raise ValueError("unexpected external evidence")
    if report["sdk_validation_n"] != expected_rows:
        raise ValueError("final SDK row coverage is incomplete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Local JSON campaign configuration",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    state_path = root / "campaign_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config"] != config or state["run_id"] != config["run_id"]:
            raise ValueError("campaign configuration or run ID changed")
        if state["event"] == "complete":
            selection = state["selection"]
            splits = training.load_splits(config["data"], final=True)
            expected_rows = (
                len(splits["calibration"])
                + len(splits["test"])
                + len(splits.get("external", []))
            )
            validate_final_report(
                selection["selected_output"],
                config["run_id"],
                selection["weights_sha256"],
                expected_rows,
                config["data"],
            )
            return
    else:
        state = {
            "event": "starting",
            "run_id": config["run_id"],
            "config": config,
            "started": time.time(),
            "deadline": time.time() + config["hours"] * 3600,
            "completed_seeds": [],
        }
    training.write_json(state_path, state)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES=config.get("cuda_devices", "0,1,2"),
    )
    try:
        completed = {entry["seed"]: entry for entry in state["completed_seeds"]}
        for seed in config["seeds"]:
            if (
                time.time()
                >= state["deadline"]
                - config.get("finalization_hours", 1.0) * 3600
            ):
                raise RuntimeError(
                    "insufficient reserved time for finalization"
                )
            if seed in completed:
                report = validate_training_report(
                    completed[seed]["output"],
                    config["run_id"],
                    seed,
                    training.sha256(Path(config["data"]) / "dev.jsonl"),
                    config,
                )
            else:
                report = train_seed(config, state, seed, environment, logs)
            entry = {
                "seed": seed,
                "output": str(
                    Path(config["root"]) / "runs" / ("seed-%03d" % seed)
                ),
                "selection_key": report["selection_key"],
                "development": report["development"],
                "weights_sha256": report["selected_checkpoint"][
                    "weights_sha256"
                ],
                "dev_sha256": report["signature"]["data"]["dev"],
                "train_sha256": report["signature"]["data"]["train"],
                "code_sha256": report["signature"]["code_manifest"][
                    "combined_sha256"
                ],
            }
            completed[seed] = entry
            state["completed_seeds"] = [
                completed[key] for key in sorted(completed)
            ]
            state["event"] = "seeds_training"
            training.write_json(state_path, state)
        candidates = [completed[seed] for seed in config["seeds"]]
        if len({entry["dev_sha256"] for entry in candidates}) != 1:
            raise ValueError(
                "cannot compare seeds with different development data"
            )
        if (
            len({entry["train_sha256"] for entry in candidates}) != 1
            or len({entry["code_sha256"] for entry in candidates}) != 1
        ):
            raise ValueError(
                "cannot compare seeds with different training data or code"
            )
        selected = min(
            candidates, key=lambda entry: tuple(entry["selection_key"])
        )
        selection = {
            "run_id": config["run_id"],
            "selected_seed": selected["seed"],
            "selected_output": selected["output"],
            "weights_sha256": selected["weights_sha256"],
            "selection_key": selected["selection_key"],
            "candidates": candidates,
            "selection_rule": "development_only_lexicographic_selection_key",
        }
        training.write_json(root / "selection.json", selection)
        state.update(event="final_evaluation", selection=selection)
        training.write_json(state_path, state)
        final_output = Path(selected["output"])
        command = [
            sys.executable,
            str(Path(__file__).with_name("security_train.py")),
            "--data",
            config["data"],
            "--output",
            str(final_output),
            "--run-id",
            config["run_id"],
            "--evaluate-only",
            "--no-checkpointing",
            "--micro-batch",
            str(config.get("micro_batch", 4)),
            "--max-len",
            str(config.get("max_len", 1024)),
            "--head-max-len",
            str(config.get("head_max_len", 192)),
        ]
        final_environment = environment
        for attempt in range(2):
            code = run_process(
                command,
                root / "logs/final-evaluation.log",
                final_environment,
                min(
                    state["deadline"] - time.time(),
                    config.get("finalization_hours", 1.0) * 3600,
                ),
            )
            if code == 0:
                break
            if attempt:
                raise RuntimeError("final evaluation exited %d" % code)
        final_splits = training.load_splits(config["data"], final=True)
        expected_rows = (
            len(final_splits["calibration"])
            + len(final_splits["test"])
            + len(final_splits.get("external", []))
        )
        report = validate_final_report(
            final_output,
            config["run_id"],
            selected["weights_sha256"],
            expected_rows,
            config["data"],
        )
        state.update(
            event="complete",
            finished=time.time(),
            report=str(final_output / "report.json"),
            export=report["export"]["path"],
        )
        training.write_json(state_path, state)
        print(
            json.dumps(
                {
                    "event": "campaign_complete",
                    "run_id": config["run_id"],
                    "selected_seed": selected["seed"],
                    "report": state["report"],
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        state.update(event="failed", error=str(exc), failed_at=time.time())
        training.write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import logging
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


today = datetime.today().strftime("%Y_%m_%d")

logging.basicConfig(
    filename=f"powerserve_sd_{today}.log",
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
)

root_folder = Path(".").absolute()
current_platform = platform.machine()

support_type = ["f32", "f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"]


def execute_command(cmd_args: List[str]) -> None:
    print("> " + " ".join(map(str, cmd_args)))
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.stdout:
        print(p.stdout, end="")
    if p.returncode != 0:
        if p.stderr:
            print(p.stderr, end="")
            logging.error(p.stderr)
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(cmd_args)}")


def is_abi_mismatch_error(stderr_or_stdout: str) -> bool:
    msg = stderr_or_stdout or ""
    return (
        "not found" in msg
        and (
            "GLIBCXX_" in msg
            or "GLIBC_" in msg
            or "CXXABI_" in msg
        )
    )


def write_json(path: Path, config: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4)


def make_sd_convert_cmd(sd_cli_path: Path, src_path: Path, dst_path: Path, out_type: str) -> List[str]:
    return [
        str(sd_cli_path),
        "-M",
        "convert",
        "-m",
        str(src_path),
        "-o",
        str(dst_path),
        "--type",
        out_type,
        "-v",
    ]


def build_model_config(model_id: str, out_type: str, converted_paths: Dict[str, str]) -> Dict:
    components: Dict[str, Dict[str, str]] = {}
    for name, rel_path in converted_paths.items():
        components[name] = {
            "format": "gguf",
            "path": rel_path,
        }

    return {
        "version": 1,
        "model_id": model_id,
        "model_arch": "sd3",
        "pipeline": "text_to_image",
        "components": components,
        "quantization": {
            "out_type": out_type,
        },
        "qnn": {
            "enabled": False,
            "workspace": "qnn",
        },
    }


def _find_existing_file(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    return None


def _resolve_model_inputs(args) -> Dict[str, Optional[Path]]:
    model_dir = args.model_dir

    main_model = args.model_path
    clip_l = args.clip_l
    clip_g = args.clip_g
    t5xxl = args.t5xxl
    vae = args.vae

    if model_dir is not None:
        if main_model is None:
            main_model = _find_existing_file(
                model_dir,
                [
                    "sd3.5_medium.safetensors",
                    "sd3_medium_incl_clips_t5xxlfp8.safetensors",
                    "sd3_medium.safetensors",
                    "sd3.5_large.safetensors",
                ],
            )
        if clip_l is None:
            clip_l = _find_existing_file(model_dir, ["clip_l.safetensors"])
        if clip_g is None:
            clip_g = _find_existing_file(model_dir, ["clip_g.safetensors"])
        if t5xxl is None:
            t5xxl = _find_existing_file(model_dir, ["t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors"])
        if vae is None:
            vae = _find_existing_file(model_dir, ["sd3.5vae.safetensors", "vae.safetensors"])

    if main_model is None:
        raise ValueError("main SD model path is required. Pass -m/--model-path or --model-dir with known filenames.")

    # If the user only provides the main model path, infer side-component files from the same directory.
    if model_dir is None:
        inferred_dir = main_model.parent
        if clip_l is None:
            clip_l = _find_existing_file(inferred_dir, ["clip_l.safetensors"])
        if clip_g is None:
            clip_g = _find_existing_file(inferred_dir, ["clip_g.safetensors"])
        if t5xxl is None:
            t5xxl = _find_existing_file(inferred_dir, ["t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors"])
        if vae is None:
            vae = _find_existing_file(inferred_dir, ["sd3.5vae.safetensors", "vae.safetensors"])

    required_components = {
        "weights": main_model,
        "clip_l": clip_l,
        "clip_g": clip_g,
        "t5xxl": t5xxl,
        "vae": vae,
    }
    missing = [name for name, path in required_components.items() if path is None]
    if missing:
        raise ValueError(
            "missing required component files for full conversion: "
            + ", ".join(missing)
            + ". pass explicit paths or use --model-dir with standard SD3.5 filenames."
        )

    for name, path in {
        "weights": main_model,
        "clip_l": clip_l,
        "clip_g": clip_g,
        "t5xxl": t5xxl,
        "vae": vae,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file does not exist: {path}")

    return {
        "weights": main_model,
        "clip_l": clip_l,
        "clip_g": clip_g,
        "t5xxl": t5xxl,
        "vae": vae,
    }


def ensure_sd_cli(sdcpp_root: Path, build_dir: Optional[Path] = None) -> Path:
    candidates = [
        sdcpp_root / "build/bin/sd-cli",
        sdcpp_root / "build-cpu/bin/sd-cli",
        sdcpp_root / "build_x86_64/bin/sd-cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if build_dir is None:
        build_dir = sdcpp_root / "build-cpu"

    execute_command(["cmake", "-S", str(sdcpp_root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    execute_command(["cmake", "--build", str(build_dir), "-j12", "--target", "sd-cli"])

    sd_cli_path = build_dir / "bin/sd-cli"
    if not sd_cli_path.exists():
        raise FileNotFoundError(f"failed to find built sd-cli at: {sd_cli_path}")
    return sd_cli_path


def _probe_sd_cli(sd_cli_path: Path) -> None:
    p = subprocess.run(
        [str(sd_cli_path), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if p.returncode == 0:
        return
    merged = (p.stdout or "") + "\n" + (p.stderr or "")
    if is_abi_mismatch_error(merged):
        raise RuntimeError(f"ABI_MISMATCH::{merged.strip()}")
    raise RuntimeError(f"sd-cli is not runnable: {sd_cli_path}\n{merged.strip()}")


def _auto_detect_sdcpp_root() -> Optional[Path]:
    candidates = [
        root_folder / "sd.cpp",
        root_folder.parent / "sd.cpp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_sd_cli_path(sd_cli_path: Optional[Path], sdcpp_root: Optional[Path], build_dir: Optional[Path]) -> Path:
    if sd_cli_path is not None:
        if not sd_cli_path.exists():
            raise FileNotFoundError(f"sd-cli does not exist: {sd_cli_path}")
        _probe_sd_cli(sd_cli_path)
        return sd_cli_path

    bundled_sd_cli = root_folder / "tools/sd_converter/bin/sd-cli"
    if bundled_sd_cli.exists():
        try:
            _probe_sd_cli(bundled_sd_cli)
            return bundled_sd_cli
        except RuntimeError as err:
            # If bundled binary is not compatible with current runtime, fallback to local build.
            if not str(err).startswith("ABI_MISMATCH::"):
                raise
            logging.warning("bundled sd-cli ABI mismatch, fallback to local build: %s", err)

    if sdcpp_root is None:
        sdcpp_root = _auto_detect_sdcpp_root()
    if sdcpp_root is None:
        raise FileNotFoundError(
            "failed to get runnable sd-cli. Pass --sd-cli-path for a binary built in current environment, "
            "or pass --sdcpp-root (or place sd.cpp at ./sd.cpp or ../sd.cpp) to auto-build."
        )
    if not sdcpp_root.exists():
        raise FileNotFoundError(
            f"sd.cpp root does not exist: {sdcpp_root}. Please pass --sdcpp-root explicitly."
        )
    local_sd_cli = ensure_sd_cli(sdcpp_root, build_dir)
    _probe_sd_cli(local_sd_cli)
    return local_sd_cli


def export_sd_gguf(
    sd_cli_path: Path,
    model_inputs: Dict[str, Optional[Path]],
    model_id: str,
    out_path: Path,
    out_type: str,
    qnn_path: Optional[Path],
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    ggml_dir = out_path / "ggml"
    ggml_dir.mkdir(parents=True, exist_ok=True)

    component_out_names = {
        "weights": "weights.gguf",
        "clip_l": "clip_l.gguf",
        "clip_g": "clip_g.gguf",
        "t5xxl": "t5xxl.gguf",
        "vae": "vae.gguf",
    }

    converted_paths: Dict[str, str] = {}
    for component, src_path in model_inputs.items():
        dst_path = ggml_dir / component_out_names[component]
        print(f">>>>>>>>>> converting {component}: {src_path.name} -> {dst_path.name} <<<<<<<<<<")
        execute_command(make_sd_convert_cmd(sd_cli_path, src_path, dst_path, out_type))
        converted_paths[component] = str(Path("ggml") / component_out_names[component])

    model_config = build_model_config(model_id=model_id, out_type=out_type, converted_paths=converted_paths)
    write_json(out_path / "model.json", model_config)

    qnn_workspace = out_path / "qnn"
    qnn_workspace.mkdir(parents=True, exist_ok=True)
    if qnn_path is not None:
        if not qnn_path.exists():
            raise FileNotFoundError(f"qnn path does not exist: {qnn_path}")
        for entry in qnn_path.iterdir():
            dst = qnn_workspace / entry.name
            if entry.is_dir():
                shutil.copytree(entry, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dst)


def main() -> None:
    parser = argparse.ArgumentParser(prog="PowerServe", description="PowerServe SD GGUF Export Tool")
    parser.add_argument("-m", "--model-path", type=Path, help="Path to SD main model (safetensors/ckpt/diffusers)")
    parser.add_argument("--model-dir", type=Path, help="Model directory to auto-discover SD3.5 component files")
    parser.add_argument("-o", "--out-path", type=Path, default=Path(f"./sd-model-{today}/"), help="Output directory")
    parser.add_argument("-t", "--out-type", type=str, choices=support_type, default="q8_0")
    parser.add_argument("--model-id", type=str, default=f"sd-model-{today}", help="Model ID")

    parser.add_argument("--clip-l", type=Path, help="Path to CLIP-L model")
    parser.add_argument("--clip-g", type=Path, help="Path to CLIP-G model")
    parser.add_argument("--t5xxl", type=Path, help="Path to T5XXL model")
    parser.add_argument("--vae", type=Path, help="Path to VAE model")

    parser.add_argument("--qnn-path", type=Path, help="Optional QNN model path", default=None)

    parser.add_argument(
        "--sd-cli-path",
        type=Path,
        default=None,
        help="Path to sd-cli binary. If omitted, tries tools/sd_converter/bin/sd-cli first.",
    )
    parser.add_argument(
        "--sdcpp-root",
        type=Path,
        default=None,
        help="Path to stable-diffusion.cpp root directory",
    )
    parser.add_argument(
        "--sdcpp-build-dir",
        type=Path,
        default=None,
        help="Optional stable-diffusion.cpp build dir (default: <sdcpp-root>/build-cpu)",
    )

    args = parser.parse_args()

    sd_cli_path = resolve_sd_cli_path(args.sd_cli_path, args.sdcpp_root, args.sdcpp_build_dir)
    model_inputs = _resolve_model_inputs(args)
    export_sd_gguf(
        sd_cli_path=sd_cli_path,
        model_inputs=model_inputs,
        model_id=args.model_id,
        out_path=args.out_path,
        out_type=args.out_type,
        qnn_path=args.qnn_path,
    )
    print(f">>>>>>>>>> done, artifacts at: {args.out_path} <<<<<<<<<<")


if __name__ == "__main__":
    main()

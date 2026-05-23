"""Model export placeholder."""

import argparse


def export_model(checkpoint, output_format, output):
    """
    Export a future model artifact.

    TODO:
    - load checkpoint
    - validate inference latency
    - export to requested format such as .onnx or .pth
    """
    raise NotImplementedError("model export is not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="Export a future Bomberland model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "pth"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_model(args.checkpoint, args.format, args.output)


if __name__ == "__main__":
    main()


import argparse
import json
import math

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.output, "w") as out_fp:
        for path in args.files:
            with open(path, "r") as in_fp:
                data = json.load(in_fp)

                for example in data:
                    if math.isnan(example["embedding"][0]):
                        continue
                    if example["label"] == 1:
                        example["label"] = "salient"
                    else:
                        example["label"] = "not_salient"

                    out_fp.write(json.dumps(example))
                    out_fp.write("\n")


if __name__ == "__main__":
    main()

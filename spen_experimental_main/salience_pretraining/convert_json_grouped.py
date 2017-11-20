import argparse
import json
import math

def write_group(group, fp):
   
    if len(group) > 0:
        fp.write(json.dumps(group))
        fp.write("\n")
    while len(group) > 0:

        ex = group[0]
        print(len(group), ex["docset_id"], ex["doc_id"], ex["label"])

        group.pop(0)
    print("")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    prev_group_id = None
    with open(args.output, "w") as out_fp:
        for path in args.files:
            with open(path, "r") as in_fp:
                data = json.load(in_fp)

                group = []

                for example in data:
                    if math.isnan(example["embedding"][0]):
                        continue
                    if example["label"] == 1:
                        example["label"] = "salient"
                    else:
                        example["label"] = "not_salient"
                    group_id = (example["docset_id"], example["doc_id"])

                    if group_id != prev_group_id:
                        write_group(group, out_fp)
                        prev_group_id = group_id
                    group.append(example)
        
        write_group(group, out_fp)

if __name__ == "__main__":
    main()

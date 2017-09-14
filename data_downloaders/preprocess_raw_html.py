import os
import argparse
import re
from nltk.tokenize import sent_tokenize, word_tokenize


ipa_pat = r'\(<span class="nowrap"><span class="IPA nopopups noexcerpt"><a href="/wiki/Help:IPA/.*?</a></span></span> <a href="/wiki/Help:Pronunciation_respelling_key".*?</a>\)'

def clean_html(html, use_focus=True):

    if use_focus:
        html = re.sub(r"<b>.*?</b>", r"__focus__", html)



    html = re.sub(r'(__focus__"?) \(.*?span>\)', r'\1 ', html)
    #html = re.sub(ipa_pat, r"", html)
    html = re.sub(r'<a href="/wiki/.*?" title=".*?">(.*?)</a>', r'\1', html) 
    html = re.sub(r'<a class="mw-redirect" href="/wiki/.*?" title=".*?">(.*?)</a>', r'\1', html) 
    html = re.sub(r'<a class="new" .*?>(.*?)</a>', r'\1', html) 
    html = re.sub(r'<sup class="reference" .*?</sup>', r'', html)
    html = re.sub(r'</?[pi]>', r'', html)
    return html


def get_content_tokens(items):
    swp_sents = [word_tokenize(s) for s in sent_tokenize(items[3])]
    wp_sents = [word_tokenize(s) for s in sent_tokenize(items[7])]


    gold_wp_sent = []
    while len(wp_sents) > 0 and len(gold_wp_sent) < 10:
        gold_wp_sent.extend(wp_sents.pop(0))            
    wp_tokens = tuple(gold_wp_sent)

    gold_swp_sent = []
    while len(swp_sents) > 0 and len(gold_swp_sent) < 10:
        gold_swp_sent.extend(swp_sents.pop(0))            
    swp_tokens = tuple(gold_swp_sent)

    return wp_tokens, swp_tokens





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--source", required=False, type=str,
        choices=["content", "html"], default="content")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir != "" and not os.path.exists(output_dir):
        os.path.makedirs(output_dir)

    with open(args.input, "r") as in_file, open(args.output, "w") as out_file: 
        in_file.readline()
        out_file.write("source\ttarget\tlen_ratio\n")
        for line in in_file:
            items = line.strip().split("\t")
            assert(len(items) == 9)

            if args.source == "content":
                wp_tokens, swp_tokens = get_content_tokens(items)

            if swp_tokens == wp_tokens:
                continue
            
            len_ratio = len(wp_tokens) / len(swp_tokens)

            out_file.write("{}\t{}\t{}\n".format(
                " ".join(wp_tokens), " ".join(swp_tokens), len_ratio))

                


exit()
path = "/proj/nlp/users/chris/ntg_proj/datasets/swp.wp.par.raw.tsv"

with open(path, "r") as f:
    print(f.readline())




    total = 0
    for line in f:
        total += 1
        items = line.strip().split("\t")
        assert(len(items) == 9)


        #print(items[3])
        swp_sents = [word_tokenize(s) for s in sent_tokenize(items[3])]
        wp_sents = [word_tokenize(s) for s in sent_tokenize(items[7])]


        gold_wp_sent = []
        while len(wp_sents) > 0 and len(gold_wp_sent) < 10:
            gold_wp_sent.extend(wp_sents.pop(0))            
        wp_tokens = tuple(gold_wp_sent)

        gold_swp_sent = []
        while len(swp_sents) > 0 and len(gold_swp_sent) < 10:
            gold_swp_sent.extend(swp_sents.pop(0))            
        swp_tokens = tuple(gold_swp_sent)

        if swp_tokens == wp_tokens:
            continue

        print(" ".join(wp_tokens))
        print(" ".join(swp_tokens))
        print("")




        continue
        exit()





#        print(items[0])
#        print(items[1])
#        print(items[2])
#        print(items[3])
#        print(items[4])
        
        print("")
        swp_html = clean_html(items[4])
        print(swp_html)

        print("")
#        print(items[5])
#        print(items[6])
#        print(items[7])
#        print(items[8])
        print("")
        wp_html = clean_html(items[8])
        print(wp_html)

        print("")

        if total > 15: break
    exit()

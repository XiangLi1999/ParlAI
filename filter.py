import sys
import re
def filter_instance(src, tgt, info='info'):
    # Remove offensive words:
    # do not have the gold list of offensive words
    # if args.bl_words and not args.leaves_only:
    #   bad_words = bl_words.extract_keywords(tgt)
    #   if bad_words:
    #       print("skip\toffensive\t%s\t%s\tbad word(s): %s" % (info, tgt, bad_words), file=sys.stderr)
    #       return True

    # Remove empty targets:
    tgttoks = tgt.split()
    if len(tgttoks) <= 1: # 1 means there is only a weight, and 0 means there's a bug..
        print("skip\temptytarget\t%s\t%s" % (info, tgt), file=sys.stderr)
        return True

    # Skip if word too long:
    toolong = False
    for w in tgttoks:
        if len(w) > 30:
            toolong = True
            break
    if toolong:
        print("skip\tlongword\t%s\t%s\tword too long" % (info, tgt), file=sys.stderr)
        return True

    srctoks = src.split()
    # Remove empty sources: (should probably uncomment, but left for reproducibility)
    #if len(srctoks) <= 1: # 1 means there is only a weight, and 0 means there's a bug..
    #   print("skip\temptysource\t%s\t%s" % (info, src), file=sys.stderr)
    #   return True

    # Remove too long turns:
    nsrctgt = len(srctoks) + len(tgttoks)
    if nsrctgt > 200:
        print("skip\ttoolong\t%s\t%s\tsrc+tgt too long, src=[%s]" % (info, tgt, src), file=sys.stderr)
        return True

    # Skip turns with URLs:
    srctgt = src + " " + tgt
    if "__url__" in srctgt:
        print("skip\turl\t%s\t%s\turl in tgt, or src =[%s]" % (info, tgt, src), file=sys.stderr)
        return True

    # Skip responses with meta data:
    if re.search("[\[\]\(\)]", srctgt) != None:
        print("skip\ttags\t%s\t%s\ttag in tgt (or src: [%s])" % (info, tgt, src), file=sys.stderr)
        return True

    # Skip yelling:
    if re.search("[A-Z]{5,}", srctgt) != None:
        print("skip\tallcaps\t%s\t%s\tall caps in tgt (or src: [%s])" % (info, tgt, src), file=sys.stderr)
        return True

    # Skip word repetitions:
    reps = False
    for i in range(2, len(tgttoks)):
        if tgttoks[i-2] == tgttoks[i] and tgttoks[i-1] == tgttoks[i]:
            reps = True
            break
    if reps:
        print("skip\trepetitions\t%s\t%s\ttoo many repetitions" % (info, tgt), file=sys.stderr)
        return True

    return False


def process_data(path): 

    temp_lst = []
    with open (path, 'r') as f: 
        for line in f:
            temp  = line.strip().split('\t')
            if len(temp) == 2:
                src, tgt = temp
            else:
                continue

            if filter_instance(src, tgt): 
                continue
            else:
                src, tgt = src.split(' '), tgt.split(' ')
                temp_lst.append((src, tgt))
    return temp_lst

def print_data(out_path, temp_lst):
    with open(out_path, 'w') as f:
        for ll in temp_lst:
            print(' '.join(ll), file=f)

    return 


if __name__ == '__main__':
    print('start')
    path = sys.argv[1]
    lst = process_data(path)
    for ll in lst:
        print(ll)

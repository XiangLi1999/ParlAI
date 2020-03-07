import sys
import re
from flashtext import KeywordProcessor

bl_words = KeywordProcessor()
en_words = KeywordProcessor()

with open('offensive.txt', 'r') as of1:
    for line in of1: 
        bl_words.add_keyword(line.strip())

with open('top50.txt', 'r') as tp50:
    for line in tp50: 
        en_words.add_keyword(line.strip())


# print(bl_words)

# bad_words = bl_words.extract_keywords('shut up asshole ?')
# print(bad_words)

def filter_instance(src, tgt, info='info'):
    # Remove offensive words:
    # do not have the gold list of offensive words
    if bl_words:
      bad_words = bl_words.extract_keywords(tgt)
      if bad_words:
          print("skip\toffensive\t%s\t%s\tbad word(s): %s" % (info, tgt, bad_words), file=sys.stderr)
          return True

    # has to be question. 
    if '?' not in src:
        return True

    # has to contain top words in English
    if en_words:
      en_word_exist = en_words.extract_keywords(tgt)
      if not en_word_exist:
          print("skip\tuncommon\t%s\t%s\tbad word(s): %s" % (info, tgt, bad_words), file=sys.stderr)
          return True

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
                src_, tgt = temp
                src_, tgt = src_.split(' '), tgt.split(' ')
                num = int(src_[0])
                src = src_[1:]
                src, tgt = ' '.join(src), ' '.join(tgt)
            else:
                continue

            if filter_instance(src, tgt): 
                continue
            else:
                # src, tgt = src.split(' '), tgt.split(' ')
                temp_lst.append((num, src, tgt))
    return temp_lst

def print_data(out_path, temp_lst):
    with open(out_path, 'w') as f:
        for (nn, ll, rr) in temp_lst:
            print(str(nn) + ' ' + ll + '\t' + rr, file=f)

    return 


if __name__ == '__main__':
    print('start')
    path = sys.argv[1]
    lst = process_data(path)
    print_data(sys.argv[2], lst)

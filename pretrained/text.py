from eng_rules import cmu_g2p, hybrid_g2p, rulebased_g2p
from cleaners import english_cleaners
import re

def pronounce_chars(line, raw_line=None, cmu_only=False, int_timing_punct=True):
    # cleaners strip things...
    puncts = ["!",",",":","?","."]
    #puncts_timing = ["4","1","1","4", "4"]
    puncts_timing = [" "," "," "," ", " "]
    end_punct = [(ni, pi) for ni, pi in enumerate(puncts) if pi in line]
    if len(end_punct) > 0:
        # preserve the end punctuation...
        if end_punct[-1][1] == line[-1]:
            end_punct = end_punct[-1]
        else:
            end_punct = (0, " ")
    else:
        end_punct = (0, " ")
    line = english_cleaners(line)
    if cmu_only:
        r0 = cmu_g2p(line, raw_line)
        return r0

    r = hybrid_g2p(line)

    if any([p in line for p in puncts]):
        new = []
        psym = r.strip().split(" ")
        lsym = line.strip().split(" ")
        for lss, pss in zip(lsym, psym):
            prev = []
            for ssi in pss.strip().split("@")[1:]:
                which_specials = [p for p in puncts if p in lss]
                if any([p in lss for p in puncts]):
                    prev.append(re.sub(re.escape("|".join(puncts)), "", ssi))
                    # ASSUME ONLY 1?
                else:
                    prev.append(ssi)
            if len(which_specials) > 0:
                prev.append(which_specials[0])
            new.append(prev)
            prev = []

        merged = ""
        for ii, chunk in enumerate(new):
            if any([p in chunk for p in puncts]):
                mstr = ""
                for ci in chunk:
                    if any([p in ci for p in puncts]):
                        which_specials = [(n, p) for n, p in enumerate(puncts) if p in ci]
                    else:
                        mstr += "@"
                        mstr += ci
                merged += mstr
                if ii < (len(new) - 1):
                    if not int_timing_punct:
                        merged += which_specials[0][1]
                    else:
                        merged += puncts_timing[which_specials[0][0]]
            else:
                merged += "@"
                merged += "@".join(chunk)
                if ii < (len(new) - 1):
                    merged += " "
        if merged[-1] == " ":
            merged = merged[:-1]
        if not int_timing_punct:
            merged += end_punct[1]
        else:
            merged += puncts_timing[end_punct[0]]
        merged += "~"
        return merged
    else:
        return r

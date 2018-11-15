import numpy as np
from text import pronounce_chars
from cleaning import text_to_sequence
from cleaning import sequence_to_text
from cleaning import get_vocabulary_sizes
import cleaners

clean_names = ["english_cleaners", "english_phone_cleaners"]
lcl_random_state = np.random.RandomState(4142)

def transform_text(char_seq, auto_pronounce=True, phone_seq=None, force_char_spc=True, symbol_processing="blended_pref", random_state=None):
    """
    chars format example: "i am learning english."
    phone_seq format example: "@ay @ae@m @l@er@n@ih@ng @ih@ng@g@l@ih@sh"

    chars_only
    phones_only
    blended_pref

    phone_seq formatting can be gotten from text, using the pronounce_chars function with 'from text import pronounce_chars'
        Uses cmudict to do pronunciation
    """
    if random_state is None:
        random_state = lcl_random_state

    if phone_seq is None and auto_pronounce is False and symbol_processing != "chars_only":
        raise ValueError("phone_seq argument must be provided for iterator with self.symbol_processing != 'chars_only', currently '{}'".format(self.symbol_processing))
    clean_char_seq = cleaners.english_cleaners(char_seq)
    char_seq_chunk = clean_char_seq.split(" ")
    dirty_seq_chunk = char_seq.split(" ")

    if auto_pronounce is True:
        if phone_seq is not None:
            raise ValueError("auto_pronounce set to True, but phone_seq was provided! Pass phone_seq=None for auto_pronounce=True")
        # take out specials then put them back...
        specials = "!?.,;:"
        puncts = "!?."
        tsc = []
        for n, csc in enumerate(char_seq_chunk):
            broke = False
            for s in specials:
                if s in csc:
                    new = csc.replace(s, "")
                    tsc.append(new)
                    broke = True
                    break
            if not broke:
                tsc.append(csc)

        if symbol_processing == "blended_pref":
            chunky_phone_seq_chunk = [pronounce_chars(w, raw_line=dirty_seq_chunk[ii], cmu_only=True) for ii, w in enumerate(tsc)]
            phone_seq_chunk = [cpsc[0] if cpsc != None else None for cpsc in chunky_phone_seq_chunk]
        else:
            phone_seq_chunk = [pronounce_chars(w) for w in tsc]
        for n, psc in enumerate(phone_seq_chunk):
            for s in specials:
                if char_seq_chunk[n][-1] == s and phone_seq_chunk[n] != None:
                    phone_seq_chunk[n] += char_seq_chunk[n][-1]
                    #if char_seq_chunk[n][-1] in puncts and n != (len(phone_seq_chunk) - 1):
                    #    # add eos
                    #    char_seq_chunk[n] += "~"
                    #    phone_seq_chunk[n] += "~"
                    break
    else:
        raise ValueError("Non auto_pronounce setting not yet configured")

    if len(char_seq_chunk) != len(phone_seq_chunk):
        raise ValueError("Char and phone chunking resulted in different lengths {} and {}!\n{}\n{}".format(len(char_seq_chunk), len(phone_seq_chunk), char_seq_chunk, phone_seq_chunk))

    if symbol_processing != "phones_only":
        spc = text_to_sequence(" ", [clean_names[0]])[0]
    else:
        spc = text_to_sequence(" ", [clean_names[1]])[0]

    int_char_chunks = []
    int_phone_chunks = []
    for n in range(len(char_seq_chunk)):
        int_char_chunks.append(text_to_sequence(char_seq_chunk[n], [clean_names[0]])[:-1])
        if phone_seq_chunk[n] == None:
            int_phone_chunks.append([])
        else:
            int_phone_chunks.append(text_to_sequence(phone_seq_chunk[n], [clean_names[1]])[:-2])

    # check inverses
    # w = [sequence_to_text(int_char_chunks[i], [self.clean_names[0]]) for i in range(len(int_char_chunks))]
    # p = [sequence_to_text(int_phone_chunks[i], [self.clean_names[1]]) for i in range(len(int_phone_chunks))]

    # TODO: Unify the two functions?
    char_phone_mask = [0] * len(int_char_chunks) + [1] * len(int_phone_chunks)
    random_state.shuffle(char_phone_mask)
    char_phone_mask = char_phone_mask[:len(int_char_chunks)]
    # setting char_phone_mask to 0 will use chars, 1 will use phones
    # these if statements override the default for blended... (above)
    if symbol_processing == "blended_pref":
        char_phone_mask = [0 if len(int_phone_chunks[i]) == 0 else 1 for i in range(len(int_char_chunks))]
    elif symbol_processing == "phones_only":
        # set the mask to use only phones
        # all files should have phones because of earlier preproc...
        char_phone_mask = [1 for i in range(len(char_phone_mask))]
    elif symbol_processing == "chars_only":
        # only use chars
        char_phone_mask = [0 for i in range(len(char_phone_mask))]

    # if the phones entry is None, the word was OOV or not recognized
    char_phone_int_seq = [int_char_chunks[i] if (len(int_phone_chunks[i]) == 0 or char_phone_mask[i] == 0) else int_phone_chunks[i] for i in range(len(int_char_chunks))]
    # check the inverse is ok
    # char_phone_txt = [sequence_to_text(char_phone_int_seq[i], [self.clean_names[char_phone_mask[i]]]) for i in range(len(char_phone_int_seq))]
    # combine into 1 sequence
    cphi = char_phone_int_seq[0]
    cpm = [char_phone_mask[0]] * len(char_phone_int_seq[0])
    if force_char_spc or self.symbol_processing != "phones_only":
        spc = text_to_sequence(" ", [clean_names[0]])[0]
    else:
        spc = text_to_sequence(" ", [clean_names[1]])[0]
    for i in range(len(char_phone_int_seq[1:])):
        # add space
        cphi += [spc]
        # always treat space as char unless in phones only mode
        if force_char_spc or self.symbol_processing != "phones_only":
            cpm += [0]
        else:
            cpm += [1]
        cphi += char_phone_int_seq[i + 1]
        cpm += [char_phone_mask[i + 1]] * len(char_phone_int_seq[i + 1])
    # trailing space
    #cphi = cphi + [spc]
    # trailing eos
    cphi = cphi + [1]
    # add trailing symbol
    if symbol_processing != "phones_only":
        cpm += [0]
    else:
        cpm += [1]
    # check inverse
    #cpt = "".join([sequence_to_text([cphi[i]], [self.clean_names[cpm[i]]]) for i in range(len(cphi))])
    #if None in phone_seq_chunk:
        #print("NUN")
        #print(cpt)
        #from IPython import embed; embed(); raise ValueError()
    return cphi, cpm

def inverse_transform_text(int_seq, mask):
    """
    mask set to zero will use chars, mask set to 1 will use phones

    should invert the transform_txt function
    """
    cphi = int_seq
    cpm = mask
    cpt = "".join([sequence_to_text([cphi[i]], [clean_names[cpm[i]]]) for i in range(len(cphi))])
    return cpt
    # setting char_phone_mask to 0 will use chars, 1 will use phones

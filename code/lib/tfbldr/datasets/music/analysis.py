# -*- coding: utf-8 -*-
from __future__ import print_function
import subprocess
from collections import OrderedDict
from music21 import converter, roman, key
import os
import math
import numpy as np
import fractions
import itertools


def notes_to_midi(notes):
    """
    notes is list of list
    """
    # r is rest
    # takes in list of list
    # # is sharp
    # b is flat
    # letters should be all caps!
    # C4 = C in 4th octave
    # 0 = rest
    # 1 = hold
    # 13 = C0
    # 25 = C1
    # 37 = C2
    # 49 = C3
    # 61 = C4
    # 73 = C5
    # 85 = C6
    base = {"C": 0,
            "D": 2,
            "E": 4,
            "F": 5,
            "G": 7,
            "A": 9,
            "B": 11}
    pitch_list = []
    for nl in notes:
        pitch_line = []
        for nn in nl:
            if nn == "R":
                base_pitch = 0
                offset = 0
                octave = 0
            elif "#" in nn or "+" in nn:
                base_pitch = base[nn[0]]
                offset = 1
                octave = (int(nn[-1]) + 1) * 12
            elif "b" in nn or "-" in nn:
                base_pitch = base[nn[0]]
                offset = -1
                octave = (int(nn[-1]) + 1) * 12
            else:
                base_pitch = base[nn[0]]
                offset = 0
                octave = (int(nn[-1]) + 1) * 12
            r = base_pitch + octave + offset
            pitch_line.append(r)
        pitch_list.append(pitch_line)
    return pitch_list


def normalize_parts_with_durations(parts, durations):
    value_durations = [[durations_map[dd] for dd in d] for d in durations]
    cumulative_durations = [np.cumsum(vd) for vd in value_durations]
    for n in range(len(parts)):
        cumulative_durations[n] = np.concatenate(([0.], cumulative_durations[n]))

    # everything is the same at the start
    normed_parts = []
    normed_durations = []
    for n in range(len(parts)):
        normed_parts.append([])
        normed_durations.append([])
    step_i = [0 for p in parts]
    held_p_i = [-1 for p in parts]
    finished = False
    # should divide into .5, .33, .25, .125, .0625 (no support smaller than 64th notes...)
    check_min = min([vd for d in value_durations for vd in d])
    cumulative_max = max([cd for d in cumulative_durations for cd in d])

    assert check_min >= .0625
    time_inc = .005
    time = 0.
    prev_event_time = 0.

    n_comb = 3
    exact_timings = [0., 0.0625, 0.125, .25, 0.5, 1., 2., 4.]
    all_exact_timings = list(itertools.product(exact_timings[3:], repeat=n_comb))
    exact_timings = exact_timings[:3] + [sum(et) for et in all_exact_timings]

    while not finished:
        # move in small increments, but only append when an event triggers
        # in any channel
        # check if an event happened
        is_event = False
        which_events = []
        for n in range(len(parts)):
            if time < cumulative_durations[n][step_i[n]]:
                pass
            else:
                is_event = True
                which_events.append(n)

        if is_event:
            for n in range(len(parts)):
                tt = round(time - prev_event_time, 4)
                min_i = np.argmin([np.abs(et - tt) for et in exact_timings])
                tt = exact_timings[min_i]
                if n in which_events:
                    normed_parts[n].append(parts[n][step_i[n]])
                    normed_durations[n].append(tt)
                    held_p_i[n] = parts[n][step_i[n]]
                    step_i[n] += 1
                else:
                    normed_parts[n].append(held_p_i[n])
                    normed_durations[n].append(tt)
            prev_event_time = time
        time += time_inc
        if time >= cumulative_max:
            for n in range(len(parts)):
                # backfill the final timestep...
                tt = round(cumulative_durations[n][-1] - prev_event_time, 4)
                min_i = np.argmin([np.abs(et - tt) for et in exact_timings])
                tt = exact_timings[min_i]
                normed_durations[n].append(tt)
            finished = True
    normed_durations = [nd[1:] for nd in normed_durations]
    normed_durations = [[inverse_durations_map[fracf(ndi)] for ndi in nd] for nd in normed_durations]
    assert len(normed_parts) == len(normed_durations)
    assert all([len(n_p) == len(n_d) for n_p, n_d in zip(normed_parts, normed_durations)])
    return normed_parts, normed_durations


def fixup_parts_durations(parts, durations):
    if len(parts[0]) != len(parts[1]):
        new_parts, new_durations = normalize_parts_with_durations(parts, durations)
        parts = new_parts
        durations = new_durations
    return parts, durations


def intervals_from_midi(parts, durations):
    if len(parts) < 2:
        raise ValueError("Must be at least 2 parts to compare intervals")
    if len(parts) > 3:
        raise ValueError("NYI")

    parts, durations = fixup_parts_durations(parts, durations)

    assert len(parts) == len(durations)
    for p, d in zip(parts, durations):
        assert len(p) == len(d)

    if len(parts) == 2:
        pairs = [(0, 1)]
    elif len(parts) == 3:
        # for 3 voices, follow the style of Fux (assume the 3 are STB)
        # soprano and bass
        # tenor and bass
        # soprano and tenor
        pairs = [(0, 2), (1, 2), (0, 1)]
    else:
        raise ValueError("Shouldn't get here, intervals_from_midi")

    intervals = []
    for pair in pairs:
        this_intervals = []
        proposed = np.array(parts[pair[0]]) - np.array(parts[pair[1]])
        for idx, p in enumerate(proposed):
            try:
                this_intervals.append(intervals_map[p])
            except:
                if len(parts) != 2:
                    from IPython import embed; embed(); raise ValueError()
                    raise ValueError("Intervals from midi, 3 voice - needs fix!")
                if parts[0][idx] == 0:
                    # rest in part 0
                    #print("Possible rest in part0")
                    this_intervals.append("R" + intervals_map[0])

                if parts[1][idx] == 0:
                    # rest in part 1
                    #print("Possible rest in part1")
                    this_intervals.append("R" + intervals_map[0])
        intervals.append(this_intervals)
    return intervals


def motion_from_midi(parts, durations):
    if len(parts) < 2:
        raise ValueError("Need at least 2 voices to get motion")
    if len(parts) > 3:
        raise ValueError("NYI")

    parts, durations = fixup_parts_durations(parts, durations)


    if len(parts) == 2:
        pairs = [(0, 1)]
    elif len(parts) == 3:
        # for 3 voices, follow the style of Fux (assume the 3 are STB)
        # soprano and bass
        # tenor and bass
        # soprano and tenor
        pairs = [(0, 2), (1, 2), (0, 1)]
    else:
        raise ValueError("Shouldn't get here, intervals_from_midi")

    motions = []
    for pair in pairs:
        # similar, oblique, contrary, direct
        p0 = np.array(parts[pair[0]])
        p1 = np.array(parts[pair[1]])
        dp0 = p0[1:] - p0[:-1]
        dp1 = p1[1:] - p1[:-1]
        # first motion is always start...
        this_motions = ["START"]
        for dip0, dip1 in zip(dp0, dp1):
            if dip0 == 0 or dip1 == 0:
                this_motions.append("OBLIQUE")
            elif dip0 == dip1:
                this_motions.append("DIRECT")
            elif dip0 > 0 and dip1 < 0:
                this_motions.append("CONTRARY")
            elif dip0 < 0 and dip1 > 0:
                this_motions.append("CONTRARY")
            elif dip0 < 0 and dip1 < 0:
                this_motions.append("SIMILAR")
            elif dip0 > 0 and dip1 > 0:
                this_motions.append("SIMILAR")
            else:
                raise ValueError("Should never see this case!")
        this_motions.append("END")
        motions.append(this_motions)
    return motions


def two_voice_rules_from_midi(parts, durations, key_signature):
    parts, durations = fixup_parts_durations(parts, durations)
    full_intervals = intervals_from_midi(parts, durations)
    full_motions = motion_from_midi(parts, durations)

    assert len(full_intervals) == len(full_motions)
    all_rulesets = []
    i = 0
    for fi, fm in zip(full_intervals, full_motions):
        fimi = 0
        this_ruleset = []
        while i < len(fi):
            this_interval = fi[i]
            this_motion = fm[i]
            this_notes = tuple([p[i] for p in parts])
            last_interval = None
            last_motion = None
            last_notes = None
            if i > 0:
                last_interval = fi[i - 1]
                last_notes = tuple([p[i - 1] for p in parts])
                last_motion = fm[i - 1]
            this_ruleset.append(make_rule(this_interval, this_motion, this_notes,
                                          key_signature,
                                          last_interval, last_motion, last_notes))
            i += 1
        all_rulesets.append(this_ruleset)
    assert len(all_rulesets[0]) == len(full_intervals[0])
    for ar in all_rulesets:
        assert len(ar) == len(all_rulesets[0])
    return all_rulesets

# previous movement, previous interval, previous notes
rule_template = "{}:{}:{},{}->{}:{}:{},{}"
# key, top note, bottom note
reduced_template = "K{},{},{}->{}:{}:{},{}"

# todo, figure out others...
base_pitch_map = {"C": 0,
                  "C#": 1,
                  "D": 2,
                  "Eb": 3,
                  "E": 4,
                  "F": 5,
                  "F#": 6,
                  "G": 7,
                  "G#": 8,
                  "A": 9,
                  "Bb": 10,
                  "B": 11}
base_note_map = {v: k for k, v in base_pitch_map.items()}

key_signature_map = {}
key_signature_map["C"] = 0
key_signature_inv_map = {v: k for k, v in key_signature_map.items()}

time_signature_map = {}
time_signature_map["4/4"] = (4, 1)

key_check = {"C": ["C", "D", "E", "F", "G", "A", "B"]}
intervals_map = {-28: "-M17",
                -27: "-m17",
                -26: "-M16",
                -25: "-m16",
                -24: "-P15",
                -23: "-M14",
                -22: "-m14",
                -21: "-M13",
                -20: "-m13",
                -19: "-P12",
                -18: "-a11",
                -17: "-P11",
                -16: "-M10",
                -15: "-m10",
                -14: "-M9",
                -13: "-m9",
                -12: "-P8",
                -11: "-M7",
                -10: "-m7",
                -9: "-M6",
                -8: "-m6",
                -7: "-P5",
                -6: "-a4",
                -5: "-P4",
                -4: "-M3",
                -3: "-m3",
                -2: "-M2",
                -1: "-m2",
                0: "P1",
                1: "m2",
                2: "M2",
                3: "m3",
                4: "M3",
                5: "P4",
                6: "a4",
                7: "P5",
                8: "m6",
                9: "M6",
                10: "m7",
                11: "M7",
                12: "P8",
                13: "m9",
                14: "M9",
                15: "m10",
                16: "M10",
                17: "P11",
                18: "a11",
                19: "P12",
                20: "m13",
                21: "M13",
                22: "m14",
                23: "M14",
                24: "P15",
                25: "m16",
                26: "M16",
                27: "m17",
                28: "M17"}

inverse_intervals_map = {v: k for k, v in intervals_map.items()}

def fracf(f):
    return fractions.Fraction(f)

inverse_durations_map = {fracf(8.): "\\breve",
                         fracf(6.): ".4",
                         fracf(4.): "4",
                         fracf(3.): ".2",
                         fracf(2.): "2",
                         fracf(1.5): ".1",
                         fracf(1.): "1",
                         fracf(.75): ".8th",
                         fracf(.5): "8th",
                         fracf(.25): "16th",
                         fracf(.125): "32nd",
                         fracf(.0625): "64th"}

durations_map = {v: k for k, v in inverse_durations_map.items()}

perfect_intervals = {"P1": None,
                     "P8": None,
                     "P5": None,
                     "P4": None,
                     "P11": None,
                     "P12": None,
                     "P15": None,
                     "P18": None,
                     "P19": None,
                     "P22": None}
neg_perfect_intervals = {"-"+str(k): None for k in perfect_intervals.keys() if "R" not in k}
harmonic_intervals = {"RP1": None,
                      "P1": None,
                      "P8": None,
                      "P5": None,
                      "P4": None,
                      "m3": None,
                      "M3": None,
                      "m6": None,
                      "M6": None,
                      "m10": None,
                      "M10": None,
                      "P11": None,
                      "P12": None,
                      "m13": None,
                      "M13": None,
                      "P15": None,
                      "m17": None,
                      "M17": None,
                      "P18": None,
                      "P19": None,
                      "m20": None,
                      "M20": None,
                      "P22": None,
                      "m24": None,
                      "M24": None}
neg_harmonic_intervals = {"-"+str(k): None for k in harmonic_intervals.keys() if "R" not in k}
nonharmonic_intervals = {"m2": None,
                         "M2": None,
                         "a4": None,
                         "m7": None,
                         "M7": None,
                         "m9": None,
                         "M9": None,
                         "a11": None,
                         "m14": None,
                         "M14": None,
                         "m16": None,
                         "M16": None,
                         "a18": None,
                         "m21": None,
                         "M21": None,
                         "m23": None,
                         "M23": None}
neg_nonharmonic_intervals = {"-"+str(k): None for k in nonharmonic_intervals.keys() if "R" not in k}

allowed_perfect_motion = {"CONTRARY": None,
                          "OBLIQUE": None}

def midi_to_notes(parts):
    """
    midi is list of list
    """
    all_parts = []
    for p in parts:
        this_notes = []
        for pi in p:
            if pi == 0:
                this_notes.append("R")
                continue
            octave = int(pi // 12 - 1)
            pos = base_note_map[pi % 12]
            this_notes.append(pos + str(octave))
        all_parts.append(this_notes)
    return all_parts


def make_rule(this_interval, this_motion, this_notes, key_signature,
              last_interval=None, last_motion=None, last_notes=None):
    if last_interval is not None:
        str_last_notes = midi_to_notes([last_notes])[0]
        str_this_notes = midi_to_notes([this_notes])[0]
        nt = rule_template.format(last_motion, last_interval,
                                  str_last_notes[0], str_last_notes[1],
                                  this_motion, this_interval,
                                  str_this_notes[0], str_this_notes[1])
    else:
        key = key_signature_inv_map[key_signature]
        str_notes = midi_to_notes([this_notes])[0]
        nt = reduced_template.format(key, str_notes[0], str_notes[1], this_motion, this_interval, str_notes[0], str_notes[1])
    return nt


def estimate_mode(parts, durations, rules, key_signature):
    parts, durations = fixup_parts_durations(parts, durations)
    first_note = [p[0] for p in parts]
    final_notes = [p[-2:] for p in parts]
    final_notes = np.array(final_notes)
    first_note = np.array(first_note)
    dfinal_notes = final_notes[-1, -1] - final_notes[-1, 0]
    if dfinal_notes == 1.:
        # final cadence indicates the tonic
        # bass almost always ends on I, i, etc except in half cadence...
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
    elif final_notes[-1, -1] == final_notes[0, 0]:
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
    elif rules[0][-1].split("->")[-1].split(":")[1] in ["P8", "P1", "P15"]:
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
    elif rules[0][0].split("->")[-1].split(":")[1] in ["RP1",]:
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
    elif len(rules) > 1 and rules[1][-1].split("->")[-1].split(":")[1] in ["P8", "P1", "P15"]:
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
    else:
        # use the last note, bass as mode estimate
        mode = midi_to_notes([[final_notes[-1, -1]]])[0][0][:-1] # strip octave
        return mode
        #print("Unknown mode estimate...")
        #from IPython import embed; embed(); raise ValueError()
    raise ValueError("This function must return before the end! Bug, rule {}".format(rule))


def rsp(rule):
    return rule.split("->")


def key_start_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices, three_voice_relaxation=False, voice_labels=(0, 1)):
    # ignore voices not used
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    for rule in rules:
        last, this = rsp(rule)
        if "K" in last:
            tm, ti, tn = this.split(":")
            lk, lns, lnb = last.split(",")
            # get rid of the K in the front
            lk = lk[1:]
            # check that note is in key?
            if three_voice_relaxation:
                check = (ti == "P12" or ti == "M10" or ti == "m10" or ti == "P8" or ti == "M6" or ti == "m6" or ti == "P5" or ti == "M3" or ti == "m3" or ti == "P1" or ti == "RP1")
            else:
                check = (ti == "P12" or ti == "P8" or ti == "P5" or ti == "P1" or ti == "RP1")
            if check:
                if lnb[:-1] == mode or lnb == "R":
                    returns.append((True, "key_start_rule: TRUE, start is in mode"))
                else:
                    returns.append((False, "key_start_rule: FALSE, first bass note {} doesn't match estimated mode {}".format(lnb, mode)))
            else:
                returns.append((False, "key_start_rule: FALSE, first interval {} is not in ['P1', 'P5', 'P8', 'P12']".format(ti)))
        else:
            returns.append((None, "key_start_rule: NONE, not applicable"))
    return returns


def next_step_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices, three_voice_relaxation=True, voice_labels=(0, 1)):
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    for rule in rules:
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")
        try:
            lm, li, ln = last.split(":")
        except ValueError:
            returns.append((None, "next_step_rule: NONE, not applicable"))
            continue
        ln0, ln1 = ln.split(",")
        dn0 = np.diff(np.array(notes_to_midi([[tn0, ln0]])[0]))
        dn1 = np.diff(np.array(notes_to_midi([[tn1, ln1]])[0]))
        note_sets = [[ln0, tn0], [ln1, tn1]]
        voice_ok = None
        msg = None
        for n, voice_step in enumerate([dn0, dn1]):
            try:
                this_step = intervals_map[-int(voice_step)]
            except KeyError:
                if note_sets[n][0] == "R":
                    if msg is None:
                        msg = "next_step_rule: NONE, rest in voice"
                    continue
                elif -int(voice_step) < min(intervals_map.keys()) or -int(voice_step) > max(intervals_map.keys()):
                    mink = min(intervals_map.keys())
                    maxk = max(intervals_map.keys())
                    msg = "next_step_rule: FALSE, voice {} stepwise movement {}->{}, jump size {} outside known range {}:{} to {}:{}".format(voice_labels[n], note_sets[n][0], note_sets[n][1], -int(voice_step), mink, intervals_map[mink],
                       maxk, intervals_map[maxk])
                    voice_ok = False
                else:
                    print("error in next step rule")
                    print("this step {}".format(this_step))
                    print("rule {}".format(rule))
                    from IPython import embed; embed(); raise ValueError()
                    raise ValueError("error in next step rule")
                    from IPython import embed; embed(); raise ValueError()

            if ignore_voices is not None and n in ignore_voices:
                if msg is None:
                    msg = "next_step_rule: NONE, skipped voice"
                continue
            if voice_ok is False:
                continue
            if this_step in ["a4", "-a4"]:
                msg = "next_step_rule: FALSE, voice {} stepwise movement {}->{}, {} not allowed".format(voice_labels[n], note_sets[n][0], note_sets[n][1], this_step)
                voice_ok = False
            elif this_step in ["P8", "-P8", "m6", "M6", "-m6", "-M6", "-M3", "-m3"]:
                msg = "next_step_rule: TRUE, voice {} skip {}->{}, {} acceptable".format(voice_labels[n], note_sets[n][0], note_sets[n][1], this_step)
                voice_ok = True
            elif abs(int(voice_step)) > 7:
                msg = "next_step_rule: FALSE, voice {} stepwise skip {}->{}, {} too large".format(voice_labels[n], note_sets[n][0], note_sets[n][1], this_step)
                voice_ok = False
            else:
                msg = "next_step_rule: TRUE, step move valid"
                voice_ok = True
        returns.append((voice_ok, msg))
    return returns


def leap_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    rules = two_voice_rules_from_midi(parts, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    returns.extend([(None, "leap_rule: NONE, not applicable")] * 2)
    for i in range(2, len(parts[0])):
        msg = None
        voice_ok = None
        for n in range(len(parts)):
            if ignore_voices is not None and n in ignore_voices:
                if msg is None:
                    msg = "leap_rule: NONE, skipped voice"
                continue
            prev_jmp = parts[n][i - 1] - parts[n][i - 2]
            cur_step = parts[n][i] - parts[n][i - 1]
            if abs(prev_jmp) > 3:
                is_opposite = math.copysign(1, cur_step) != math.copysign(1, prev_jmp)
                is_step = abs(cur_step) == 1 or abs(cur_step) == 2
                # check if it outlines a triad?
                if is_opposite and is_step:
                    msg = "leap_rule: TRUE, voice {} leap of {} corrected".format(n, prev_jmp)
                    voice_ok = True
                else:
                    msg = "leap_rule: FALSE, voice {} leap of {} not corrected".format(n, prev_jmp)
                    voice_ok = False
            else:
                msg = "leap_rule: NONE, not applicable"
                voice_ok = None
        returns.append((voice_ok, msg))
    assert len(returns) == len(parts[0])
    return returns


def parallel_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices,
                  three_voice_relaxation=False, voice_labels=(0, 1)):
    # ignore voices not used
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    for idx, rule in enumerate(rules):
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")
        try:
            lm, li, ln = last.split(":")
        except ValueError:
            returns.append((None, "parallel_rule: NONE, not applicable"))
            continue
        ln0, ln1 = ln.split(",")
        dn0 = np.diff(np.array(notes_to_midi([[tn0, ln0]])[0]))
        dn1 = np.diff(np.array(notes_to_midi([[tn1, ln1]])[0]))
        note_sets = [[ln0, tn0], [ln1, tn1]]
        if li == "M10" or li == "m10":
            if not three_voice_relaxation and ti == "P8" and timings[0][idx] == 0.:
                # battuta octave
                returns.append((False, "parallel_rule: FALSE, battuta octave {}->{} disallowed on first beat".format(li, ti)))
                continue
        if ti in perfect_intervals or ti in neg_perfect_intervals:
            if three_voice_relaxation:
                allowed = allowed_perfect_motion
            else:
                allowed = allowed_perfect_motion
            if tm in allowed:
                returns.append((True, "parallel_rule: TRUE, movement {} into perfect interval {} allowed".format(tm, ti)))
                continue
            else:
                returns.append((False, "parallel_rule: FALSE, movement {} into perfect interval {} not allowed".format(tm, ti)))
                continue
        elif ti in harmonic_intervals or ti in neg_harmonic_intervals or ti in nonharmonic_intervals or ti in neg_nonharmonic_intervals:
            # allowed note check is elsewhere
            returns.append((True, "parallel_rule: TRUE, all movements including {} allowed into interval {}".format(tm, ti)))
        else:
            print("parallel_rule: shouldn't get here")
            from IPython import embed; embed(); raise ValueError()
            raise ValueError("parallel_rule: shouldn't get here")
    return returns


def beat_parallel_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    # ignore voices not used
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    for idx, rule in enumerate(rules):
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")
        try:
            lm, li, ln = last.split(":")
        except ValueError:
            returns.append((None, "beat_parallel_rule: NONE, not applicable"))
            continue
        ln0, ln1 = ln.split(",")

        dn0 = np.diff(np.array(notes_to_midi([[tn0, ln0]])[0]))
        dn1 = np.diff(np.array(notes_to_midi([[tn1, ln1]])[0]))
        note_sets = [[ln0, tn0], [ln1, tn1]]

        # rP1 is rest
        if ti in ["P8", "P5"]:
            if idx < 2:
                returns.append((True, "beat_parallel_rule: TRUE, no earlier parallel move"))
                continue
            plast, pthis = rsp(rules[idx - 2])
            pm, pi, pn = pthis.split(":")
            if pi in ["P8", "P5"] and pi == ti:
                # check beats - use the 0th voice?
                if 0. == timings[0][idx] and 0. == timings[0][idx - 2] and abs(inverse_intervals_map[li]) < 5:
                    if pi == "P5":
                        common_notes = {}
                        for _n in pn.split(",") + ln.split(",") + tn.split(","):
                            common_notes[_n] = True
                        # 4 common notes over 3 events with 2 voices means it is syncopated
                        if len(common_notes) == 4:
                            returns.append((True, "beat_parallel_rule: TRUE, parallel perfect interval {} allowed in syncopation".format(pi)))
                        else:
                            returns.append((False, "beat_parallel_rule: FALSE, parallel perfect interval {} not allowed in syncopation".format(pi)))
                    else:
                        returns.append((False, "beat_parallel_rule: FALSE, previous downbeat had parallel perfect interval {}".format(pi)))
                    continue
            returns.append((True, "beat_parallel_rule: TRUE, no beat parallel move"))
        else:
            returns.append((True, "beat_parallel_rule: TRUE, no beat parallel move"))
    return returns


def bar_consonance_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices, three_voice_relaxation=True, voice_labels=(0, 1)):
    # ignore voices not used
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    assert all([len(timings[i]) == len(timings[0]) for i in range(len(timings))])
    for idx, rule in enumerate(rules):
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")

        timing_i = timings[0][idx]
        for n in range(len(timings)):
            assert timings[n][idx] == timing_i

        if timing_i != 0.:
            returns.append((None, "bar_consonance_rule: NONE, rule not applicable on beat {}".format(timing_i)))
        elif timing_i == 0.:
            if ti in harmonic_intervals or ti in neg_harmonic_intervals:
                returns.append((True, "bar_consonance_rule: TRUE, harmonic interval {} allowed on downbeat".format(ti)))
            else:
                if idx < len(rules) - 1:
                    nthis, nxt = rsp(rules[idx + 1])
                    nm, ni, nn = nxt.split(":")
                    if ni in harmonic_intervals or ni in neg_harmonic_intervals:
                        if int(ni[-1]) == 0 or int(ti[-1]) == 0:
                            returns.append((False, "bar_consonance_rule: FALSE, suspension outside range"))
                        else:
                            if int(ti[-1]) - int(ni[-1]) == 1:
                                returns.append((True, "bar_consonance_rule: TRUE, non-consonant interval {} resolves downward to {}".format(ti, ni)))
                            elif int(ti[-1]) - int(ni[-1]) == -1:
                                returns.append((True, "bar_consonance_rule: TRUE, non-consonant interval {} resolves upward to {}".format(ti, ni)))
                            else:
                                returns.append((False, "bar_consonance_rule: FALSE, non-consonant interval {} not resolved, goes to {}".format(ti, ni)))
                    else:
                        returns.append((False, "bar_consonance_rule: FALSE, non-consonant interval {} disallowed on downbeat".format(ti)))
                else:
                    returns.append((False, "bar_consonance_rule: FALSE, non-consonant interval {} disallowed on downbeat".format(ti)))
        else:
            raise ValueError("bar_consonance_rule: shouldn't get here")
    return returns


def passing_tone_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    # ignore voices not used
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    assert all([len(timings[i]) == len(timings[0]) for i in range(len(timings))])
    for idx, rule in enumerate(rules):
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")

        timing_i = timings[0][idx]
        for n in range(len(timings)):
            assert timings[n][idx] == timing_i

        if timing_i == 0.:
            returns.append((None, "passing_tone_rule: NONE, rule not applicable on beat {}".format(timing_i)))
        elif timing_i != 0.:
            if ti in harmonic_intervals or ti in neg_harmonic_intervals:
                returns.append((True, "passing_tone_rule: TRUE, harmonic interval {} allowed on downbeat".format(ti)))
            else:
                lm, li, ln = last.split(":")
                ln0, ln1 = ln.split(",")
                # passing tone check
                pitches = np.array(notes_to_midi([[ln0, ln1], [tn0, tn1]]))
                last_diffs = np.diff(pitches, axis=0)

                this, nxt = rsp(rules[idx + 1])
                nm, ni, nn = nxt.split(":")
                nn0, nn1 = nn.split(",")
                pitches = np.array(notes_to_midi([[tn0, tn1], [nn0, nn1]]))
                nxt_diffs = np.diff(pitches, axis=0)

                not_skip = [n for n in range(last_diffs.shape[1]) if n not in ignore_voices]
                last_diffs = last_diffs[:, not_skip]
                nxt_diffs = nxt_diffs[:, not_skip]
                last_ok = np.where(np.abs(last_diffs) >= 3)[0]
                nxt_ok = np.where(np.abs(nxt_diffs) >= 3)[0]
                if len(last_ok) == 0 and len(nxt_ok) == 0:
                    returns.append((True, "passing_tone_rule: TRUE, passing tones allowed on upbeat"))
                else:
                    returns.append((False, "passing_tone_rule: FALSE, non-passing tones not allowed on upbeat"))
        else:
            raise ValueError("passing_tone_rule: shouldn't get here")
    return returns


def sequence_step_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    rules = rules[0]
    key = key_signature_inv_map[key_signature]
    returns = []
    assert all([len(timings[i]) == len(timings[0]) for i in range(len(timings))])
    last_timing_i = 0.
    for idx, rule in enumerate(rules):
        last, this = rsp(rule)
        tm, ti, tn = this.split(":")
        tn0, tn1 = tn.split(",")

        timing_i = timings[0][idx]
        for n in range(len(timings)):
            assert timings[n][idx] == timing_i

        time_num = time_signature[0]
        time_denom = time_signature[1]

        diff_timing_i = timing_i - last_timing_i
        # diff timing is circular
        if timing_i == 0. and last_timing_i == 3.:
            diff_timing_i = 1.
        last_timing_i = timing_i
        # force to match quarters
        if timing_i not in [0., 1., 2., 3.]:
            raise ValueError("sequence_step_rule: timing not recognized!")
        if idx < 1 or abs(diff_timing_i) != 1.:
            returns.append((None, "sequence_step_rule: NONE, not applicable at step {}".format(idx)))
            continue
        elif abs(diff_timing_i) == 1.:
            lm, li, ln = last.split(":")
            ln0, ln1 = ln.split(",")

            pitches = np.array(notes_to_midi([[ln0, ln1], [tn0, tn1]]))
            last_diffs = np.diff(pitches, axis=0)
            not_skip = [n for n in range(last_diffs.shape[1]) if n not in ignore_voices]
            last_diffs = last_diffs[:, not_skip]
            last_ok = np.where(np.abs(last_diffs) >= 3)[0]

            if idx + 1 == len(rules):
                if ti in harmonic_intervals or ti in neg_harmonic_intervals:
                    returns.append((True, "sequence_step_rule: TRUE, interval {} always allowed".format(ti)))
                elif len(last_ok) == 0 and timing_i not in [0., 2.]:
                    returns.append((True, "sequence_step_rule: TRUE, interval {} is a continuation".format(ti)))
                else:
                    returns.append((False, "sequence_step_rule: FALSE, interval {} disallowed in termination".format(ti)))
                continue

            this, nxt = rsp(rules[idx + 1])
            nm, ni, nn = nxt.split(":")
            nn0, nn1 = nn.split(",")
            pitches = np.array(notes_to_midi([[tn0, tn1], [nn0, nn1]]))
            nxt_diffs = np.diff(pitches, axis=0)
            nxt_diffs = nxt_diffs[:, not_skip]
            nxt_ok = np.where(np.abs(nxt_diffs) >= 3)[0]

            if ti in harmonic_intervals or ti in neg_harmonic_intervals:
                returns.append((True, "sequence_step_rule: TRUE, interval {} always allowed".format(ti)))
            else:
                if timing_i == 0.:
                    returns.append((False, "sequence_step_rule: FALSE, cannot have non-harmonic interval {} on bar part 0.".format(ti)))
                elif timing_i == 1.:
                    if len(nxt_ok) == 0 and len(last_ok) == 0:
                        if ni in harmonic_intervals or ni in neg_harmonic_intervals:
                            returns.append((True, "sequence_step_rule: TRUE, interval {} at bar part 1. allowed as part of continuation".format(ti)))
                        else:
                            returns.append((False, "sequence_step_rule: FALSE, interval {} at bar part 1. not allowed, next interval not harmonic".format(ti)))
                    else:
                        nxt, nxtnxt = rsp(rules[idx + 2])
                        nnm, nni, nnn = nxtnxt.split(":")
                        nnn0, nnn1 = nnn.split(",")
                        pitches = np.array(notes_to_midi([[nn0, nn1], [nnn0, nnn1]]))
                        nxtnxt_diffs = np.diff(pitches, axis=0)
                        nxtnxt_diffs = nxtnxt_diffs[:, not_skip]
                        nxtnxt_ok = np.where(np.abs(nxtnxt_diffs) >= 3)[0]
                        nxtnxt_resolves = np.where(np.sign(nxtnxt_diffs) != np.sign(nxt_diffs))[0]

                        # check that it resolves in cambiata...
                        if len(nxt_ok) == 1 and len(nxtnxt_ok) == 0 and nni in harmonic_intervals and sum(nxtnxt_resolves) == 0:
                            if not_skip == [1]:
                                info_tup = (tn1, nn1, nnn1)
                            elif not_skip == [0]:
                                info_tup = (tn0, nn0, nnn0)
                            else:
                                print("sequence_step_rule: other not_skip voices not yet supported...")
                                from IPython import embed; embed(); raise ValueError()

                            returns.append((True, "sequence_step_rule: TRUE, cambiata {}->{}->{} in voice {} detected at bar part 1. to 3.".format(info_tup[0], info_tup[1], info_tup[2], not_skip[0])))
                        else:
                            returns.append((False, "sequence_step_rule: FALSE, interval {} at bar part 1. not allowed, not a continuation or cambiata".format(ti)))
                elif timing_i == 2.:
                    # last and next must be harmonic, and must be continuation...
                    if len(nxt_ok) == 0 and len(last_ok) == 0:
                        if ni in harmonic_intervals or ni in neg_harmonic_intervals:
                            returns.append((True, "sequence_step_rule: TRUE, interval {} at bar part 2. allowed as part of continuation".format(ti)))
                        else:
                            returns.append((False, "sequence_step_rule: FALSE, interval {} at bar part 2. not allowed, next interval not harmonic or no continuation".format(ti)))
                elif timing_i == 3.:
                    if len(nxt_ok) == 0 and len(last_ok) == 0:
                        if ni in harmonic_intervals or ni in neg_harmonic_intervals:
                            returns.append((True, "sequence_step_rule: TRUE, interval {} at bar part 3. allowed as part of continuation".format(ti)))
                        else:
                            returns.append((False, "sequence_step_rule: FALSE, interval {} at bar part 3. not allowed, next interval not harmonic".format(ti)))
                    else:
                        print("sequence_step_rule, timing 3. edge case")
                        from IPython import embed; embed(); raise ValueError()
                else:
                    print("sequence_step_rule: shouldn't get here")
                    from IPython import embed; embed(); raise ValueError()
        else:
            print("sequence_step_rule, shouldn't get here")
            from IPython import embed; embed(); raise ValueError()
    return returns

two_voice_species1_minimal_rules_map = OrderedDict()
two_voice_species1_minimal_rules_map["next_step_rule"] = next_step_rule
two_voice_species1_minimal_rules_map["parallel_rule"] = parallel_rule
two_voice_species1_minimal_rules_map["bar_consonance_rule"] = bar_consonance_rule

def check_two_voice_species1_minimal_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    res = [two_voice_species1_minimal_rules_map[arm](parts, durations, key_signature, time_signature, mode, timings, ignore_voices) for arm in two_voice_species1_minimal_rules_map.keys()]

    global_check = True
    for r in res:
        rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
        if all(rr):
            pass
        else:
            global_check = False
    return (global_check, res)

two_voice_species1_rules_map = OrderedDict()
two_voice_species1_rules_map["key_start_rule"] = key_start_rule
two_voice_species1_rules_map["bar_consonance_rule"] = bar_consonance_rule
two_voice_species1_rules_map["next_step_rule"] = next_step_rule
two_voice_species1_rules_map["parallel_rule"] = parallel_rule

# leap rule is not a rule :|
#all_rules_map["leap_rule"] = leap_rule

def check_two_voice_species1_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    res = [two_voice_species1_rules_map[arm](parts, durations, key_signature, time_signature, mode, timings, ignore_voices) for arm in two_voice_species1_rules_map.keys()]

    global_check = True
    for r in res:
        rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
        if all(rr):
            pass
        else:
            global_check = False
    return (global_check, res)

two_voice_species2_rules_map = OrderedDict()
two_voice_species2_rules_map["key_start_rule"] = key_start_rule
two_voice_species2_rules_map["bar_consonance_rule"] = bar_consonance_rule
two_voice_species2_rules_map["parallel_rule"] = parallel_rule
two_voice_species2_rules_map["beat_parallel_rule"] = beat_parallel_rule
two_voice_species2_rules_map["next_step_rule"] = next_step_rule
two_voice_species2_rules_map["passing_tone_rule"] = passing_tone_rule
def check_two_voice_species2_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    res = [two_voice_species2_rules_map[arm](parts, durations, key_signature, time_signature, mode, timings, ignore_voices) for arm in two_voice_species2_rules_map.keys()]

    global_check = True
    for r in res:
        rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
        if all(rr):
            pass
        else:
            global_check = False
    return (global_check, res)

two_voice_species3_rules_map = OrderedDict()
two_voice_species3_rules_map["key_start_rule"] = key_start_rule
two_voice_species3_rules_map["bar_consonance_rule"] = bar_consonance_rule
two_voice_species3_rules_map["parallel_rule"] = parallel_rule
two_voice_species3_rules_map["beat_parallel_rule"] = beat_parallel_rule
two_voice_species3_rules_map["next_step_rule"] = next_step_rule
two_voice_species3_rules_map["sequence_step_rule"] = sequence_step_rule
def check_two_voice_species3_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    res = [two_voice_species3_rules_map[arm](parts, durations, key_signature, time_signature, mode, timings, ignore_voices) for arm in two_voice_species3_rules_map.keys()]

    global_check = True
    for r in res:
        rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
        if all(rr):
            pass
        else:
            global_check = False
    return (global_check, res)

two_voice_species4_rules_map = OrderedDict()
two_voice_species4_rules_map["key_start_rule"] = key_start_rule
two_voice_species4_rules_map["bar_consonance_rule"] = bar_consonance_rule
two_voice_species4_rules_map["parallel_rule"] = parallel_rule
two_voice_species4_rules_map["beat_parallel_rule"] = beat_parallel_rule
two_voice_species4_rules_map["next_step_rule"] = next_step_rule
two_voice_species4_rules_map["sequence_step_rule"] = sequence_step_rule
def check_two_voice_species4_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):
    res = [two_voice_species4_rules_map[arm](parts, durations, key_signature, time_signature, mode, timings, ignore_voices) for arm in two_voice_species4_rules_map.keys()]

    global_check = True
    for r in res:
        rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
        if all(rr):
            pass
        else:
            global_check = False
    return (global_check, res)

def make_timings(durations, beats_per_measure, duration_unit):
    # use normalized_durations?
    if beats_per_measure != 4:
        raise ValueError("beats per measure {} needs support in handle_durations".format(beats_per_measure))

    if duration_unit != 1:
        raise ValueError("duration unit {} needs support in handle_durations".format(duration_unit))

    # U for upbeat, D for downbeat?
    all_lines = []
    all_timings = []

    if beats_per_measure == 4 and duration_unit == 1:
        pass
    else:
        raise ValueError("Beats per measure {} and duration unit {} NYI".format(beats_per_measure, duration_unit))

    value_durations = [[float(durations_map[di]) for di in d] for d in durations]
    cumulative_starts = [np.concatenate(([0.], np.cumsum(vd)))[:-1] for vd in value_durations]
    for cline in cumulative_starts:
        this_lines = []
        for cl in cline:
            this_lines.append(cl % beats_per_measure)
            #if cl % beats_per_measure in downbeats:
            #    this_lines.append("D")
            #else:
            #    this_lines.append("U")
        all_lines.append(this_lines)
    return all_lines


def estimate_timing(parts, durations, time_signature):
    # returns U or D for each part if it starts on upbeat or downbeat
    parts, durations = fixup_parts_durations(parts, durations)
    beats_per_measure = time_signature[0]
    duration_unit = time_signature[1]
    ud = make_timings(durations, beats_per_measure, duration_unit)
    return ud


def analyze_two_voices(parts, durations, key_signature_str, time_signature_str, species="species1",
                       cantus_firmus_voices=None):
    # not ideal but keeps stuff consistent
    key_signature = key_signature_map[key_signature_str]
    # just check that it parses here
    time_signature = time_signature_map[time_signature_str]
    beats_per_measure = time_signature[0]
    duration_unit = time_signature[1]

    parts, durations = fixup_parts_durations(parts, durations)

    rules = two_voice_rules_from_midi(parts, durations, key_signature)
    mode = estimate_mode(parts, durations, rules, key_signature)
    timings = estimate_timing(parts, durations, time_signature)

    ignore_voices = cantus_firmus_voices
    if species == "species1_minimal":
        r = check_two_voice_species1_minimal_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    elif species == "species1":
        r = check_two_voice_species1_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    elif species == "species2":
        r = check_two_voice_species2_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    elif species == "species3":
        r = check_two_voice_species3_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    elif species == "species4":
        r = check_two_voice_species4_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    else:
        raise ValueError("Unknown species argument {}".format(species))
    all_ok = r[0]
    this_ok = []
    true_false = OrderedDict()
    true_false["True"] = []
    true_false["False"] = []
    for rr in r[1]:
        for n in range(len(rr)):
            this_ok.append((n, rr[n][0], rr[n][1]))
            if rr[n][0] == True or rr[n][0] == None:
                true_false["True"].append(n)
            else:
                true_false["False"].append(n)
    true_false["True"] = sorted(list(set(true_false["True"])))
    true_false["False"] = sorted(list(set(true_false["False"])))
    return (all_ok, true_false, rules, sorted(this_ok))


def test_two_voice_species1():
    print("Running test for species1...")
    all_ex = fetch_two_voice_species1()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        # TODO: handle strings like "C"
        key_signature = "C"
        # as in sheet music
        time_signature = "4/4"
        # durations can be "64th", "32nd", "16th", "8th", "1", "2", "4", "8"
        # also any of these can be dotted (".") e.g. ".8th" (dotted eighth)
        # or summed for a tie "1+8th"
        # TODO: Triplets?
        aok = analyze_two_voices(parts, durations, key_signature, time_signature,
                                 species="species1", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)
        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
        else:
            print("Test passed for note sequence {}".format(fig_name))


def test_two_voice_species2():
    print("Running test for species2...")
    all_ex = fetch_two_voice_species2()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        key_signature = "C"
        time_signature = "4/4"
        aok = analyze_two_voices(parts, durations, key_signature, time_signature,
                                 species="species2", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)

        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
        else:
            print("Test passed for note sequence {}".format(fig_name))


def test_two_voice_species3():
    print("Running test for species3...")
    all_ex = fetch_two_voice_species3()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        key_signature = "C"
        time_signature = "4/4"
        aok = analyze_two_voices(parts, durations, key_signature, time_signature,
                                 species="species3", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)

        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        all_answers = [True if aa == None else aa for aa in all_answers]
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
        else:
            print("Test passed for note sequence {}".format(fig_name))


def test_two_voice_species4():
    print("Running test for species4...")
    all_ex = fetch_two_voice_species4()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        key_signature = "C"
        time_signature = "4/4"
        aok = analyze_two_voices(parts, durations, key_signature, time_signature,
                                 species="species4", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)

        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        all_answers = [True if aa == None else aa for aa in all_answers]
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
            from IPython import embed; embed(); raise ValueError()
        else:
            print("Test passed for note sequence {}".format(fig_name))


def three_voice_rules_from_midi(parts, durations, key_signature):
    parts, durations = fixup_parts_durations(parts, durations)
    full_intervals = intervals_from_midi(parts, durations)
    full_motions = motion_from_midi(parts, durations)

    assert len(full_intervals) == len(full_motions)
    all_rulesets = []
    for fi, fm in zip(full_intervals, full_motions):
        i = 0
        fimi = 0
        this_ruleset = []
        while i < len(fi):
            this_interval = fi[i]
            this_motion = fm[i]
            this_notes = tuple([p[i] for p in parts])
            last_interval = None
            last_motion = None
            last_notes = None
            if i > 0:
                last_interval = fi[i - 1]
                last_notes = tuple([p[i - 1] for p in parts])
                last_motion = fm[i - 1]
            this_ruleset.append(make_rule(this_interval, this_motion, this_notes,
                                          key_signature,
                                          last_interval, last_motion, last_notes))
            i += 1
        all_rulesets.append(this_ruleset)
    assert len(all_rulesets[0]) == len(full_intervals[0])
    for ar in all_rulesets:
        assert len(ar) == len(all_rulesets[0])
    return all_rulesets

three_voice_species1_minimal_rules_map = OrderedDict()
three_voice_species1_minimal_rules_map["bar_consonance_rule"] = bar_consonance_rule
three_voice_species1_minimal_rules_map["next_step_rule"] = next_step_rule
three_voice_species1_minimal_rules_map["parallel_rule"] = parallel_rule

def check_three_voice_species1_minimal_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):

    pairs = [(0, 2), (1, 2), (0, 1)]
    res = []
    for n, pair in enumerate(pairs):
        if n > 0:
            # skip key start rule on inner voices
            skip_rules = ["key_start_rule"]
        else:
            skip_rules = []
        res_i = [three_voice_species1_rules_map[arm]([parts[pair[0]], parts[pair[1]]],
                    [durations[pair[0]], durations[pair[1]]], key_signature,
                    time_signature, mode, [timings[pair[0]], timings[pair[1]]],
                    ignore_voices=[], three_voice_relaxation=True, voice_labels=pair)
                for arm in three_voice_species1_rules_map.keys() if arm not in skip_rules]
        res.append(res_i)

    global_check = True
    # better check all 3...
    for res_i in res:
    # only check top 2 voices
    #for res_i in res[:-1]:
        for r in res_i:
            rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
            if all(rr):
                pass
            else:
                global_check = False
    return (global_check, res)


three_voice_species1_rules_map = OrderedDict()
three_voice_species1_rules_map["key_start_rule"] = key_start_rule
three_voice_species1_rules_map["bar_consonance_rule"] = bar_consonance_rule
three_voice_species1_rules_map["next_step_rule"] = next_step_rule
three_voice_species1_rules_map["parallel_rule"] = parallel_rule

# leap rule is not a rule :|
#all_rules_map["leap_rule"] = leap_rule

def check_three_voice_species1_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices):

    pairs = [(0, 2), (1, 2), (0, 1)]
    res = []
    for n, pair in enumerate(pairs):
        if n > 0:
            # skip key start rule on inner voices
            skip_rules = ["key_start_rule"]
        else:
            skip_rules = []
        res_i = [three_voice_species1_rules_map[arm]([parts[pair[0]], parts[pair[1]]],
                    [durations[pair[0]], durations[pair[1]]], key_signature,
                    time_signature, mode, [timings[pair[0]], timings[pair[1]]],
                    ignore_voices=[], three_voice_relaxation=True, voice_labels=pair)
                for arm in three_voice_species1_rules_map.keys() if arm not in skip_rules]
        res.append(res_i)

    global_check = True
    # only check top 2 voices
    for res_i in res[:-1]:
        for r in res_i:
            rr = [True if ri[0] is True or ri[0] is None else False for ri in r]
            if all(rr):
                pass
            else:
                global_check = False
    return (global_check, res)


def analyze_three_voices(parts, durations, key_signature_str, time_signature_str, species="species1",
                         cantus_firmus_voices=None):
    # not ideal but keeps stuff consistent
    key_signature = key_signature_map[key_signature_str]
    # just check that it parses here
    time_signature = time_signature_map[time_signature_str]
    beats_per_measure = time_signature[0]
    duration_unit = time_signature[1]

    parts, durations = fixup_parts_durations(parts, durations)

    rules = three_voice_rules_from_midi(parts, durations, key_signature)
    mode = estimate_mode(parts, durations, rules, key_signature)
    timings = estimate_timing(parts, durations, time_signature)

    ignore_voices = cantus_firmus_voices
    if species == "species1_minimal":
        r = check_three_voice_species1_minimal_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    elif species == "species1":
        r = check_three_voice_species1_rule(parts, durations, key_signature, time_signature, mode, timings, ignore_voices)
    else:
        raise ValueError("Unknown species argument {}".format(species))
    all_ok = r[0]
    true_false = OrderedDict()
    true_false["True"] = []
    true_false["False"] = []
    this_ok = []
    # only check top 2 voice pairs
    #for res_i in r[1][:-1]:
    for res_i in r[1]:
        for rr in res_i:
            for n in range(len(rr)):
                this_ok.append((n, rr[n][0], rr[n][1]))
                if rr[n][0] == True or rr[n][0] == None:
                    true_false["True"].append(n)
                else:
                    true_false["False"].append(n)
    true_false["True"] = sorted(list(set(true_false["True"])))
    true_false["False"] = sorted(list(set(true_false["False"])))
    return (all_ok, true_false, rules, sorted(this_ok))


def test_three_voice_species1():
    print("Running test for three voice species1...")
    all_ex = fetch_three_voice_species1()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        key_signature = "C"
        time_signature = "4/4"
        aok = analyze_three_voices(parts, durations, key_signature, time_signature,
                                   species="species1", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)

        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        all_answers = [True if aa == None else aa for aa in all_answers]
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
            from IPython import embed; embed(); raise ValueError()
        else:
            print("Test passed for note sequence {}".format(fig_name))


def test_three_voice_mcts_species1_counterexample():
    print("Running test for three voice species1...")
    all_ex = fetch_three_voice_mcts_species1_counterexample()

    for ex in all_ex:
        nd = ex["notes_and_durations"]
        notes = [[ndii[0] for ndii in ndi] for ndi in nd]
        durations = [[ndii[1] for ndii in ndi] for ndi in nd]
        #notes = ex["notes"]
        #durations = ex["durations"]
        answers = ex["answers"]
        fig_name = ex["name"]
        ig = [ex["cantus_firmus_voice"],]
        parts = notes_to_midi(notes)
        key_signature = "C"
        time_signature = "4/4"
        aok = analyze_three_voices(parts, durations, key_signature, time_signature,
                                   species="species1_minimal", cantus_firmus_voices=ig)
        aok_lu = aok[1]
        aok_rules = aok[2]

        all_answers = [-1] * len(answers)

        for a in aok[-1]:
            if all_answers[a[0]] == -1:
                all_answers[a[0]] = a[1]
            else:
                if a[1] in [None, True]:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = True
                    else:
                        all_answers[a[0]] &= True
                else:
                    if all_answers[a[0]] == None:
                        all_answers[a[0]] = False
                    else:
                        all_answers[a[0]] &= False
        all_answers = [True if aa == None else aa for aa in all_answers]
        assert len(all_answers) == len(answers)
        equal = [aa == a for aa, a in zip(all_answers, answers)]
        if not all(equal):
            print("Test FAIL for note sequence {}".format(fig_name))
            from IPython import embed; embed(); raise ValueError()
        else:
            print("Test passed for note sequence {}".format(fig_name))



if __name__ == "__main__":
    import argparse

    from datasets import fetch_two_voice_species1
    from datasets import fetch_two_voice_species2
    from datasets import fetch_two_voice_species3
    from datasets import fetch_two_voice_species4
    from datasets import fetch_three_voice_species1
    from datasets import fetch_three_voice_mcts_species1_counterexample


    #test_two_voice_species1()
    #test_two_voice_species2()
    #test_two_voice_species3()
    #test_two_voice_species4()
    #test_three_voice_species1()
    test_three_voice_mcts_species1_counterexample()

    """
    # fig 5, gradus ad parnassum
    notes = [["A3", "A3", "G3", "A3", "B3", "C4", "C4", "B3", "D4", "C#4", "D4"],
             ["D3", "F3", "E3", "D3", "G3", "F3", "A3", "G3", "F3", "E3", "D3"]]
    durations = [[4.] * len(notes[0]), [4.] * len(notes[1])]
    # can add harmonic nnotations as well to plot
    #chord_annotations = ["i", "I6", "IV", "V6", "I", "IV6", "I64", "V", "I"]
    """
    ex = fetch_three_voice_species1()
    nd = ex[-1]["notes_and_durations"]
    notes = [[ndii[0] for ndii in ndi] for ndi in nd]
    durations = [[ndii[1] for ndii in ndi] for ndi in nd]
    # can we do all these automatically?
    parts = notes_to_midi(notes)
    interval_figures = intervals_from_midi(parts, durations)
    _, interval_durations = fixup_parts_durations(parts, durations)
    # need to figure out duration convention (maybe support floats and str both?)
    durations = [[int(di) for di in d] for d in durations]

    # treble, bass, treble_8, etc
    clefs = ["treble", "treble", "bass"]
    time_signatures = [(4, 4), (4, 4), (4, 4)]

    from visualization import pitches_and_durations_to_pretty_midi
    from visualization import plot_pitches_and_durations
    pitches_and_durations_to_pretty_midi([parts], [durations],
                                         save_dir="samples",
                                         name_tag="sample_{}.mid",
                                         default_quarter_length=240,
                                         voice_params="piano")

    # figure out plotting of tied notes
    # fix zoom
    plot_pitches_and_durations(parts, durations,
                               interval_figures=interval_figures,
                               interval_durations=interval_durations,
                               use_clefs=clefs)

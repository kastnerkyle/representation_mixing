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


# https://github.com/davidnalesnik/lilypond-roman-numeral-tool
# http://lsr.di.unimi.it/LSR/Snippet?id=710
roman_include = r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A function to create Roman numerals for harmonic analysis.
%%
%% Syntax: \markup \rN { ...list of symbols... }
%%
%% List symbols in this order (as needed): Roman numeral (or note name),
%% quality, inversion figures from top to bottom, "/" (if a secondary
%% function), Roman numeral (or note name).  Usually, you can skip unnecessary
%% items, though a spacer may be needed in some cases.  Use "" instead of the
%% initial symbol to start with the quality or inversion, for example.  Elements
%% must be separated by whitespace.
%%
%% Notenames are represented by their English LilyPond names.  In addition, you
%% may capitalize the name for a capitalized note name.
%%
%% Preceding a string representing a Roman numeral with English alterations
%% (f, flat, s, sharp, ff, flatflat, ss, x, sharpsharp, natural)
%% will attach accidentals, for example, "fVII" -> flat VII; "sharpvi" -> sharp vi.
%% You may precede inversion numbers with alterations, though "+" is not
%% presently supported.
%%
%% Qualities: use "o" for diminished, "h" for half-diminished, "+" for augmented,
%% and "f" for flat.  Other indications are possible such as combinations of "M"
%% and "m" (M, m, MM7, Mm, mm, Mmm9, etc.); add, add6, etc.
%%
%% To scale all numerals: \override  LyricText #'font-size = #2
%% or \override  TextScript #'font-size = #2
%% To scale individual numerals: \markup \override #'(font-size . 2) \rN { ... }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% THE APPROACH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% In our approach, a Roman numeral consists of

%% 1. A "base".  OPTIONAL. This may be a Roman numeral (some combination of I, i, V,
%% and v, unenforced); a note name; or some other string.  Roman numerals may be
%% preceded by an accidental, and a note name may be followed by one.

%% 2. a quality indicator.  OPTIONAL.  Eventually, this will simply be something to
%% set as a superscript following the base, whether or not it is actually a
%% indicator of quality.

%% 3. A single inversion number, or more than one, to be set as a column.  OPTIONAL.
%% An initial accidental is supported.  (This will be extended to "anything you want
%% to appear in a column after the quality indicator.")

%% 4. "/" followed by a "secondary base" for indicating tonicization.  OPTIONAL.
%% As with 1. this may a Roman numeral or note name, and may include an accidental.

%% The input syntax is chosen to be friendly to the user rather than the computer.
%% In particular, the user usually need only type the symbols needed when
%% reading the analytical symbol aloud.  This is not perfect: spacers
%% may be necessary for omissions.  Additionally, we try to interpret symbols
%% without requiring extra semantic indicators: i.e., figure out whether a string
%% represents a Roman numeral or a note name without the user adding an extra sign.
%% In the future, indicators might prove necessary to resolve ambiguity: along with
%% a flag to distinguish Roman numeral from note name, braces to enclose inversion
%% figures may be useful.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT FORMATTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The user's input is available as a list of strings.  Here we convert this
%% list into a nested list which describes the structure of the input.

#(define (split-list symbols splitter-list)
   "Split a list of strings by a splitter which is a member of a list of
potential splitters.  The splitter may be alone or part of a string.
input is split into
@code{(( ...strings up to splitter... ) ( ...strings beginning with splitter... ))}
This function is Used to split notation for secondary chords and to isolate
inversion numbers."
   (let loop ((sym symbols) (result '()))
     (cond
      ((or (null? sym)
           (find (lambda (y) (string-contains (car sym) y)) splitter-list))
       (list (reverse result) sym))
      (else (loop (cdr sym) (cons (car sym) result))))))

#(define numbers '("2" "3" "4" "5" "6" "7" "8" "9" "11" "13"))

#(define qualities
   ;; only to allow omission of base when quality is alone
   ;; TODO--combinations of M and m, add, ADD . . .
   '("o" "+" "h"))

#(define (base-and-quality arg)
   (let ((len (length arg)))
     (cond
      ((= 0 len) '(() ()))
      ((= 1 len)
       (if (find (lambda (y) (string= (car arg) y)) qualities)
           (list '() (list (car arg)))
           (list (list (car arg)) '()))) ;; TODO figure out which is given
      ((= 2 len) (list (list (car arg)) (cdr arg))))))

#(define (base-quality-figures symbols)
   ;; given (vii o 4 3) --> ((vii o) (4 3)) --> ((vii) (o) (4 3))
   ;; (4 3) --> (() (4 3)) --> (() () (4 3))
   ;; () --> (() ()) --> (() () ())
   (let* ((split-by-numbers (split-list symbols numbers))
          (b-and-q (base-and-quality (car split-by-numbers))))
     (append b-and-q (cdr split-by-numbers))))

#(define (parse-input input)
   (let (;; (vii o 4 3 / ii) --> ((vii o 4 3) (/ ii))
          (split (split-list input '("/"))))
     ;; --> ( ((vii) (o) (4 3)) (/ ii) )
     (append
      (list (base-quality-figures (car split)))
      (cdr split))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOTE NAMES / ACCIDENTALS %%%%%%%%%%%%%%%%%%%%%%%%%%

%% Formatting the input into interpretable lists continues here.  We are now
%% concerned with distinguishing Roman numerals from note names, and with representing
%% the presence and position of accidentals.

%% If a string belongs to the list of possible English notenames, we assume that
%% it is a note name.  The note name will be typeset as uppercase or lowercase depending
%% on the capitalization of the input string.

%% If a string is not a note name, we look for an alteration prefix, never a suffix.

%% The procedure parse-string-with-accidental breaks a string into a list representing
%% initial/terminal alterations and what is left.

%% Notenames and names of accidentals are based on English names.  Other
%% languages may be used by adding variables modeled after english-note names and
%% english-alterations, and changing the definitions of note names and alterations to
%% point to these new variables.

#(define english-note-names
   (map (lambda (p) (symbol->string (car p)))
     (assoc-get 'english language-pitch-names)))

#(define note-names english-note-names)

#(define (note-name? str)
   (let ((lowercased (format #f "~(~a~)" str)))
     (list? (member lowercased note-names))))

%% Groupings sharing an initial character are arranged in descending length so there
%% is no need to search for longest match in parse-string-with-accidental.
#(define english-alterations
   '("flatflat" "flat" "ff" "f"
      "sharpsharp" "sharp" "ss" "s" "x"
      "natural" "n"))

#(define alterations english-alterations)

#(define (parse-note-name str)
   "Given a note name, return a list consisting of the general name followed by
the alteration or @code{#f} if none."
   (let* ((first-char (string-take str 1))
          (all-but-first (string-drop str 1))
          (all-but-first (if (string-prefix? "-" all-but-first)
                             (string-drop all-but-first 1)
                             all-but-first))
          (all-but-first (if (string-null? all-but-first) #f all-but-first)))
     (list first-char all-but-first)))

#(define (parse-string-with-accidental str)
   "Given @var{str}, return a list in this format: (initial-accidental?
note-name-or-figure-or-RN terminal-accidental?) If an accidental is found, include
its string, otherwise @code{#t}."
   (if (not (string-null? str))
       (if (note-name? str)
           (cons #f (parse-note-name str))
           ;; Is it a Roman numeral or figure preceded (or followed) by an accidental?
           (let* ((accidental-prefix
                   (find (lambda (s) (string-prefix? s str)) alterations))
                  (accidental-suffix
                   (find (lambda (s) (string-suffix? s str)) alterations))
                  (rest (cond
                         (accidental-prefix
                          (string-drop str (string-length accidental-prefix)))
                         (accidental-suffix
                          (string-drop-right str (string-length accidental-suffix)))
                         (else str))))
             (list accidental-prefix rest accidental-suffix)))))
%{
#(define (inversion? str)
   "Check to see if a string contains a digit.  If so, it is an inversion figure."
   (not (char-set=
         char-set:empty
         (char-set-intersection (string->char-set str) char-set:digit))))
%}

%% We need to add extra space after certain characters in the default LilyPond
%% font to avoid overlaps with characters that follow.  Several of these kernings
%% don't seem to be necessary anymore, and have been commented out.
#(define (get-extra-kerning arg)
   (let ((last-char (string-take-right arg 1)))
     (cond
      ((string= last-char "V") 0.1)
      ((string= last-char "f") 0.2)
      ;((string= last-char "s") 0.2) ; sharp
      ;((string= last-char "x") 0.2) ; double-sharp
      ;((string= last-char "ss") 0.2) ; double-sharp
      (else 0.0))))

%% Create accidentals with appropriate vertical positioning.
#(define make-accidental-markup
   `(("f" . ,(make-general-align-markup Y DOWN (make-flat-markup)))
     ("flat" . ,(make-general-align-markup Y DOWN (make-flat-markup)))
     ("ff" . ,(make-general-align-markup Y DOWN (make-doubleflat-markup)))
     ("flatflat" . ,(make-general-align-markup Y DOWN (make-doubleflat-markup)))
     ("s" . ,(make-general-align-markup Y -0.6 (make-sharp-markup)))
     ("sharp" . ,(make-general-align-markup Y -0.6 (make-sharp-markup)))
     ("ss" . ,(make-general-align-markup Y DOWN (make-doublesharp-markup)))
     ("x" . ,(make-general-align-markup Y DOWN (make-doublesharp-markup)))
     ("sharpsharp" . ,(make-general-align-markup Y DOWN (make-doublesharp-markup)))
     ("n" . ,(make-general-align-markup Y -0.6 (make-natural-markup)))
     ("natural" . ,(make-general-align-markup Y -0.6 (make-natural-markup)))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BASE MARKUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define (make-base-markup base scaling-factor)
   (let* ((base-list (parse-string-with-accidental base))
          (init-acc (first base-list))
          (end-acc (last base-list))
          (extra-space-right (get-extra-kerning (second base-list))))
     (cond
      (init-acc
       (make-concat-markup
        (list
         (make-fontsize-markup -3
           (assoc-ref make-accidental-markup init-acc))
         (make-hspace-markup (* 0.2 scaling-factor))
         (second base-list))))
      (end-acc
       (make-concat-markup
        (list
         (second base-list)
         (make-hspace-markup (* (+ 0.2 extra-space-right) scaling-factor))
         (make-fontsize-markup -3
           (assoc-ref make-accidental-markup end-acc)))))
      (else
       (if (> extra-space-right 0.0)
           (make-concat-markup
            (list
             base
             (make-hspace-markup (* extra-space-right scaling-factor))))
           base)))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% QUALITY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Symbols representing diminished, half-diminished, and augmented qualities are
%% drawn to rest atop of baseline (alignment direction = DOWN), and moved by
%% make-quality-markup to their final vertical position.  They are tailored to
%% the font-size (-3) of the ultimate caller (\rN -- default font-size = 1).

%% These symbols are drawn from scratch to allow for customization.  should we
%% simply use symbols from a font?

#(define (make-diminished-markup font-size)
   "Create circle markup for diminished quality."
   (let* ((scaling-factor (magstep font-size))
          (r (* 0.48 scaling-factor))
          (th (* 0.1 scaling-factor)))
     (make-translate-markup
      (cons r r)
      (make-draw-circle-markup r th #f))))

#(define (make-half-diminished-markup font-size)
   "Create slashed circle markup for half-diminished quality."
   (let* ((scaling-factor (magstep font-size))
          (x (* 0.56 scaling-factor))
          (y (* 0.56 scaling-factor))
          (r (* 0.48 scaling-factor))
          (th (* 0.1 scaling-factor)))
     (make-translate-markup
      (cons x y)
      (make-combine-markup
       (make-draw-circle-markup r th #f)
       (make-override-markup `(thickness . ,scaling-factor)
         (make-combine-markup
          (make-draw-line-markup (cons (- x) (- y)))
          (make-draw-line-markup (cons x y))))))))

% Noticeably thinner than "+" from font -- change?
#(define (make-augmented-markup font-size)
   "Create cross markup for augmented quality."
   (let* ((scaling-factor (magstep font-size))
          (x (* 0.56 scaling-factor))
          (y (* 0.56 scaling-factor)))
     (make-override-markup `(thickness . ,scaling-factor)
       (make-translate-markup (cons x y)
         (make-combine-markup
          (make-combine-markup
           (make-draw-line-markup (cons (- x) 0))
           (make-draw-line-markup (cons 0 (- y))))
          (make-combine-markup
           (make-draw-line-markup (cons x 0))
           (make-draw-line-markup (cons 0 y))))))))

%% TODO: more "science" in the vertical position of quality markers.
#(define (make-quality-markup quality font-size offset)
   (cond
    ;; The quantity 'offset' by itself will cause symbol to rest on the midline.  We
    ;; enlarge offset so that the symbol will be more centered alongside a possible
    ;; figure.  (Topmost figure rests on midline.)
    ((string= quality "o") (make-raise-markup (* offset 1.25) (make-diminished-markup font-size)))
    ((string= quality "h") (make-raise-markup (* offset 1.25) (make-half-diminished-markup font-size)))
    ((string= quality "+") (make-raise-markup (* offset 1.25) (make-augmented-markup font-size)))
    (else (make-raise-markup offset (make-fontsize-markup font-size quality)))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIGURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define (make-figure-markup font-size)
   `(("f" . ,(make-general-align-markup Y DOWN
               (make-fontsize-markup font-size (make-flat-markup))))
     ("ff" . ,(make-general-align-markup Y DOWN
               (make-fontsize-markup font-size (make-doubleflat-markup))))
     ("flat" . ,(make-general-align-markup Y DOWN
                  (make-fontsize-markup font-size (make-flat-markup))))
     ("flatflat" . ,(make-general-align-markup Y DOWN
               (make-fontsize-markup font-size (make-doubleflat-markup))))
     ("s" . ,(make-general-align-markup Y -0.6
               (make-fontsize-markup font-size (make-sharp-markup))))
     ("x" . ,(make-general-align-markup Y -1.9
               (make-fontsize-markup font-size (make-doublesharp-markup))))
     ("ss" . ,(make-general-align-markup Y -1.9
               (make-fontsize-markup font-size (make-doublesharp-markup))))
     ("sharp" . ,(make-general-align-markup Y -0.6
                   (make-fontsize-markup font-size (make-sharp-markup))))
     ("sharpsharp" . ,(make-general-align-markup Y -1.9
               (make-fontsize-markup font-size (make-doublesharp-markup))))
     ("+" . ,(make-general-align-markup Y -1.5 (make-augmented-markup (+ font-size 2))))
     ("n" . ,(make-general-align-markup Y -0.6
               (make-fontsize-markup font-size (make-natural-markup))))
     ("natural" . ,(make-general-align-markup Y -0.6
                     (make-fontsize-markup font-size (make-natural-markup))))
     ))

#(use-modules (ice-9 regex))

#(define (hyphen-to-en-dash str)
   (string-regexp-substitute "-" "â" str))

%% Regular expression for splitting figure strings into words, digits, and connector characters.
#(define figure-regexp (make-regexp "[[:alpha:]]+|[[:digit:]]+|[^[:alnum:]]+"))

#(define (format-figures figures font-size)
   (let ((scaling-factor (magstep font-size)))
     (map (lambda (fig)
            (let* ((parsed-fig (map match:substring (list-matches figure-regexp fig)))
                   ;; Conversion causes character encoding problem with Frescobaldi
                   ;; if done before applying regexp
                   (parsed-fig (map hyphen-to-en-dash parsed-fig)))
              (reduce
               (lambda (elem prev) (make-concat-markup (list prev elem)))
               empty-markup
               (map (lambda (f)
                      (let ((alteration
                             (assoc-ref (make-figure-markup (- font-size 2)) f)))
                        (make-concat-markup
                         (list
                          (if alteration alteration (make-fontsize-markup font-size f))
                          ;; TODO: don't add space at the end
                          (make-hspace-markup (* 0.2 scaling-factor))))))
                 parsed-fig))))
       figures)))

#(define (make-figures-markup figures font-size offset)
   ;; Without offset the column of figures would be positioned such that the
   ;; topmost figure rests on the baseline. Adding offset causes the upper figure
   ;; to rest on the midline of base.
   (let ((formatted-figures (format-figures figures -3)))
     (make-override-markup `(baseline-skip . ,(* 1.4 (magstep font-size)))
       (make-raise-markup offset
         (make-right-column-markup formatted-figures)))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SECONDARY RN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define (make-secondary-markup second-part scaling-factor)
   (make-concat-markup
    (list
     (car second-part)
     (if (string-null? (cadr second-part))
         empty-markup
         (make-concat-markup
          (list
           (make-hspace-markup (* 0.2 scaling-factor))
           (if (car (parse-string-with-accidental (cadr second-part)))
               (make-hspace-markup (* 0.2 scaling-factor))
               empty-markup)
           (make-base-markup (cadr second-part) scaling-factor)))))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SYNTHESIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define-markup-command (rN layout props symbols) (markup-list?)
   #:properties ((font-size 1))
   "Create a symbol for Roman numeral analysis from a @var{symbols}, a list
of strings."
   (let* ((parsed-input (parse-input symbols))
          (first-part (car parsed-input))
          (second-part (cadr parsed-input)) ; slash and what follows
          (base (car first-part))
          (quality (cadr first-part))
          (figures (caddr first-part))
          ;; A multiplier for scaling quantities measured in staff-spaces to
          ;; reflect font-size delta.  Spacing between elements is currently
          ;; controlled by the magstep of the rN font-size.
          (scaling-factor (magstep font-size))
          (base-markup
           (if (or (null? base) (string-null? (car base))) ; "" used as spacer
               #f
               (make-base-markup (car base) scaling-factor)))
          ;; The height of figures and quality determined by midline of base.  If
          ;; there is no base, use forward slash as a representative character.
          (dy (* 0.5
                (interval-length
                 (ly:stencil-extent
                  (interpret-markup
                   layout props (if (markup? base-markup)
                                    base-markup "/"))
                  Y))))
          (quality-markup
           (if (null? quality)
               #f
               (make-concat-markup
                (list
                 (make-hspace-markup (* 0.1 scaling-factor))
                 (make-quality-markup (car quality) -3 dy)))))
          (figures-markup
           (if (null? figures)
               #f
               (make-concat-markup
                (list (make-hspace-markup (* 0.1 scaling-factor))
                  (make-figures-markup figures font-size dy)))))
          (secondary-markup
           (if (null? second-part)
               #f
               (make-concat-markup
                (list
                 (if (= (length figures) 1)
                     ;; allows slash to tuck under if single figure
                     (make-hspace-markup (* -0.2 scaling-factor))
                     ;; slightly more space given to slash
                     (make-hspace-markup (* 0.2 scaling-factor)))
                 (make-secondary-markup second-part scaling-factor)))))
          (visible-markups
           (filter markup?
                   (list base-markup quality-markup figures-markup secondary-markup))))
     (interpret-markup layout props
       (make-concat-markup visible-markups))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KEY INDICATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define-markup-command (keyIndication layout props arg) (markup?)
   #:properties ((font-size 1))
   "Create a key indicator consisting of a English note name followed by a
colon.  Whitespace after the note name will be included in the returned markup."
   (let* ((scaling-factor (magstep font-size))
          (divide-at-spaces (string-match "([^[:space:]]+)([[:space:]]+)$" arg))
          (base (if divide-at-spaces
                    (match:substring divide-at-spaces 1)
                    arg))
          (trailing-spaces (if divide-at-spaces
                               (match:substring divide-at-spaces 2)
                               empty-markup)))
     (interpret-markup layout props
       (make-concat-markup
        (list
         (make-base-markup base scaling-factor)
         (make-hspace-markup (* 0.2 scaling-factor))
         ":"
         trailing-spaces)))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCALE DEGREES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#(define (parse-scale-degree str alteration-list)
   "Given @var{str}, return a list in this format: (name-of-alteration-or-#f degree)."
   (if (not (string-null? str))
       (let* ((alteration
               (find (lambda (s) (string-prefix? s str)) alteration-list))
              (rest (if alteration
                        (string-drop str (string-length alteration))
                        str)))
         (list alteration rest))))

#(define (hat font-size)
   "Draw a caret for use with scale degrees."
   (let* ((scaling-factor (magstep font-size))
          (x (* 0.25 scaling-factor))
          (y x)
          (th scaling-factor))
     (make-override-markup `(thickness . ,th)
       (make-combine-markup
        (make-draw-line-markup (cons x y))
        (make-translate-markup (cons x y)
          (make-draw-line-markup (cons x (- y))))))))

#(define-markup-command (scaleDegree layout props degree) (markup?)
   #:properties ((font-size 1))
   "Return a digit topped by a caret to represent a scale degree.  Alterations may
be added by prefacing @var{degree} with an English alteration."
   (let* ((scale-factor (magstep font-size))
          (caret (hat font-size))
          (degree-list (parse-scale-degree degree english-alterations))
          (alteration (car degree-list))
          (number (cadr degree-list))
          (alteration-markup (assoc-ref make-accidental-markup alteration))
          (alteration-markup
           (if alteration-markup
               (make-fontsize-markup -3 alteration-markup)
               alteration-markup))
          (number-and-caret
           (make-general-align-markup Y DOWN
             (make-override-markup `(baseline-skip . ,(* 1.7 scale-factor))
               (make-center-column-markup
                (list
                 caret
                 number))))))
     (interpret-markup layout props
       (if alteration-markup
           (make-concat-markup (list
                                alteration-markup
                                number-and-caret))
           number-and-caret))))
"""

# Convenience function to reuse the defined env
def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=True):
    """
    Print and execute command on system
    """
    ret = []
    for line in execute(cmd, shell=shell):
        ret.append(line)
        print(line, end="")
    return ret


def chord_and_chord_duration_to_quantized(list_of_chords, list_of_chord_durations, min_dur,
                                          list_of_chord_metas=None, verbose=False):
    def expand(chords, durs, min_dur, chord_metas):
        assert len(chords) == len(durs)
        expanded = [int(d // min_dur) for d in durs]
        check = [d / min_dur for d in durs]
        if not all([e == c for e, c in zip(expanded, check)]):
            if verbose:
                errs = sum([e != c for e, c in zip(expanded, check)])

                print("WARNING: {} of {} notes did not evenly quantize".format(errs, len(check)))
        # shift up by 1 to accomodate the fact that 0 is rest
        stretch = [[c] * e for c, e in zip(chords, expanded)]
        stretch_metas = []
        for m in chord_metas:
           assert len(m) == len(expanded)
           stretch_m = [[mi] * e for mi, e in zip(m, expanded)]
           stretch_metas.append(stretch_m)
        # flatten out to 1 voice
        return ([ci for c in stretch for ci in c],) + tuple([[smii for smi in sm for smii in smi] for sm in stretch_metas])

    assert len(list_of_chords) == len(list_of_chord_durations)
    if list_of_chord_metas is not None:
        assert len(list_of_chords) == len(list_of_chord_metas[0])
        for lcm in list_of_chord_metas:
            assert len(lcm) == len(list_of_chord_metas[0])

    ci = expand(list_of_chords, list_of_chord_durations, min_dur, [list_of_chord_metas[i] for i in range(len(list_of_chord_metas))])

    if list_of_chord_metas is not None:
        return (ci[0],) + tuple([ci[i + 1] for i in range(len(ci[1:]))])
    else:
        return (ci[0],)


def pitch_and_duration_to_quantized(list_of_pitch_voices, list_of_duration_voices, min_dur,
                                    list_of_metas_voices=None, verbose=False, hold_symbol=True):
    def expand(pitch, dur, min_dur, metas):
        assert len(pitch) == len(dur)
        expanded = [int(d // min_dur) for d in dur]
        check = [d / min_dur for d in dur]
        if not all([e == c for e, c in zip(expanded, check)]):
            if verbose:
                errs = sum([e != c for e, c in zip(expanded, check)])

                print("WARNING: {} of {} notes did not evenly quantize".format(errs, len(check)))
        # shift up by 1 to accomodate the fact that 0 is rest
        if hold_symbol:
            stretch = [[p if p == 0 else p + 1] + [1] * (e - 1) if e > 1 else [p if p == 0 else p + 1] for p, e in zip(pitch, expanded)]
        else:
            stretch = [[p] * e for p, e in zip(pitch, expanded)]
        stretch_metas = []
        for m in metas:
           assert len(m) == len(expanded)
           stretch_m = [[mi] * e for mi, e in zip(m, expanded)]
           stretch_metas.append(stretch_m)
        # flatten out to 1 voice
        return ([pi for p in stretch for pi in p],) + tuple([[smii for smi in sm for smii in smi] for sm in stretch_metas])

    assert len(list_of_pitch_voices) == len(list_of_duration_voices)
    if list_of_metas_voices is not None:
        assert len(list_of_duration_voices) == len(list_of_metas_voices[0])
        for lmv in list_of_metas_voices:
            assert len(lmv) == len(list_of_metas_voices[0])
    res = []
    for n, (lpv, ldv) in enumerate(zip(list_of_pitch_voices, list_of_duration_voices)):
        qi = expand(lpv, ldv, min_dur, [list_of_metas_voices[i][n] for i in range(len(list_of_metas_voices))])
        res.append(qi)

    min_len = min([len(ri[0]) for ri in res])
    max_len = max([len(ri[0]) for ri in res])
    if min_len != max_len:
        if verbose:
           print("min_len != max_len, truncating")
    res0 = [ri[0][:min_len] for ri in res]
    quantized = np.array(res0).transpose()
    if list_of_metas_voices is not None:
        return (quantized,) + tuple([[ri[i + 1] for ri in res] for i in range(len(res[0][1:]))])
    else:
        return (quantized,)


def pitches_and_durations_to_pretty_midi(pitches, durations,
                                         save_dir="samples",
                                         name_tag="sample_{}.mid",
                                         add_to_name=0,
                                         lower_pitch_limit=12,
                                         list_of_quarter_length=None,
                                         default_quarter_length=120,
                                         default_resolution=220,
                                         voice_params="woodwinds"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # allow list of list of list
    """
    takes in list of list of list, or list of array with axis 0 time, axis 1 voice_number (S,A,T,B)
    outer list is over samples, middle list is over voice, inner list is over time
    durations assumed to be scaled to quarter lengths e.g. 1 is 1 quarter note
    2 is a half note, etc
    """
    is_seq_of_seq = False
    try:
        pitches[0][0]
        durations[0][0]
        if not hasattr(pitches, "flatten") and not hasattr(durations, "flatten"):
            is_seq_of_seq = True
    except:
        raise ValueError("pitches and durations must be a list of array, or list of list of list (time, voice, pitch/duration)")

    if is_seq_of_seq:
        if hasattr(pitches[0], "flatten"):
            # it's a list of array, convert to list of list of list
            pitches = [[[pitches[i][j, k] for j in range(pitches[i].shape[0])] for k in range(pitches[i].shape[1])] for i in range(len(pitches))]
            durations = [[[durations[i][j, k] for j in range(durations[i].shape[0])] for k in range(durations[i].shape[1])] for i in range(len(durations))]

    import pretty_midi
    # BTAS mapping
    def weird():
        voice_mappings = ["Sitar", "Orchestral Harp", "Acoustic Guitar (nylon)",
                          "Pan Flute"]
        voice_velocity = [20, 80, 80, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., .95]
        return voice_mappings, voice_velocity, voice_offset, voice_decay

    if voice_params == "weird":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
    elif voice_params == "weird_r":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
        voice_mappings = voice_mappings[::-1]
        voice_velocity = voice_velocity[::-1]
        voice_offset = voice_offset[::-1]
    elif voice_params == "nylon":
        voice_mappings = ["Acoustic Guitar (nylon)"] * 4
        voice_velocity = [20, 16, 25, 10]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
        voice_decay = voice_decay[::-1]
    elif voice_params == "legend":
        # LoZ
        voice_mappings = ["Acoustic Guitar (nylon)"] * 3 + ["Pan Flute"]
        voice_velocity = [20, 16, 25, 5]
        voice_offset = [0, 0, 0, 24]
        voice_decay = [1., 1., 1., .95]
    elif voice_params == "organ":
        voice_mappings = ["Church Organ"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [.98, .98, .98, .98]
    elif voice_params == "piano":
        voice_mappings = ["Acoustic Grand Piano"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "electric_piano":
        voice_mappings = ["Electric Piano 1"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "harpsichord":
        voice_mappings = ["Harpsichord"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "woodwinds":
        voice_mappings = ["Bassoon", "Clarinet", "English Horn", "Oboe"]
        voice_velocity = [50, 30, 30, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    else:
        # eventually add and define dictionary support here
        raise ValueError("Unknown voice mapping specified")

    # normalize
    mm = float(max(voice_velocity))
    mi = float(min(voice_velocity))
    dynamic_range = min(80, (mm - mi))
    # keep same scale just make it louder?
    voice_velocity = [int((80 - dynamic_range) + int(v - mi)) for v in voice_velocity]

    if not is_seq_of_seq:
        order = durations.shape[-1]
    else:
        try:
            # TODO: reorganize so list of array and list of list of list work
            order = durations[0].shape[-1]
        except:
            order = len(durations[0])
    voice_mappings = voice_mappings[-order:]
    voice_velocity = voice_velocity[-order:]
    voice_offset = voice_offset[-order:]
    voice_decay = voice_decay[-order:]
    if not is_seq_of_seq:
        pitches = [pitches[:, i, :] for i in range(pitches.shape[1])]
        durations = [durations[:, i, :] for i in range(durations.shape[1])]

    n_samples = len(durations)
    for ss in range(n_samples):
        durations_ss = durations[ss]
        pitches_ss = pitches[ss]
        # same number of voices
        assert len(durations_ss) == len(pitches_ss)
        # time length match
        assert all([len(durations_ss[i]) == len(pitches_ss[i]) for i in range(len(pitches_ss))])
        pm_obj = pretty_midi.PrettyMIDI(initial_tempo=default_quarter_length, resolution=default_resolution)
        # Create an Instrument instance
        def mkpm(name):
            return pretty_midi.instrument_name_to_program(name)

        def mki(p):
            return pretty_midi.Instrument(program=p, name=str(p))

        pm_programs = [mkpm(n) for n in voice_mappings]
        pm_instruments = [mki(p) for p in pm_programs]

        if list_of_quarter_length is None:
            # qpm to s per quarter = 60 s per min / quarters per min
            time_scale = 60. / default_quarter_length
        else:
            time_scale = 60. / list_of_quarter_length[ss]

        time_offset = np.zeros((order,))

        # swap so that SATB order becomes BTAS for voice matching
        pitches_ss = pitches_ss[::-1]
        durations_ss = durations_ss[::-1]

        # voice
        for jj in range(order):
            # time / steps
            for ii in range(len(durations_ss[jj])):
                try:
                    pitches_isj = pitches_ss[jj][ii]
                    durations_isj = durations_ss[jj][ii]
                except IndexError:
                    # voices may stop short
                    continue
                p = int(pitches_isj)
                d = durations_isj
                if d < 0:
                    continue
                if p < 0:
                    continue
                # hack out the whole last octave?
                s = time_scale * time_offset[jj]
                e = time_scale * (time_offset[jj] + voice_decay[jj] * d)
                time_offset[jj] += d
                if p < lower_pitch_limit:
                    continue
                note = pretty_midi.Note(velocity=voice_velocity[jj],
                                        pitch=p + voice_offset[jj],
                                        start=s, end=e)
                # Add it to our instrument
                pm_instruments[jj].notes.append(note)
        # Add the instrument to the PrettyMIDI object
        for pm_instrument in pm_instruments:
            pm_obj.instruments.append(pm_instrument)
        # Write out the MIDI data

        sv = save_dir + os.sep + name_tag.format(ss + add_to_name)
        try:
            pm_obj.write(sv)
        except ValueError:
            print("Unable to write file {} due to mido error".format(sv))


def quantized_to_pitch_duration(quantized,
                                quantized_bin_size,
                                hold_symbol,
                                max_hold_bars=1):
    """
    takes in list of list of list, or list of array with axis 0 time, axis 1 voice_number (S,A,T,B)
    outer list is over samples, middle list is over voice, inner list is over time
    """
    if hold_symbol != False:
        raise ValueError("quantized_to_pitch_duration NEEDS FIX FOR HELD NOTES")

    is_seq_of_seq = False
    try:
        quantized[0][0]
        if not hasattr(quantized[0], "flatten"):
            is_seq_of_seq = True
    except:
        try:
            quantized[0].shape
        except AttributeError:
            raise ValueError("quantized must be a sequence of sequence (such as list of array, or list of list) or numpy array")

    # list of list or mb?
    n_samples = len(quantized)
    all_pitches = []
    all_durations = []

    max_hold = int(max_hold_bars / quantized_bin_size)
    if max_hold < max_hold_bars:
        max_hold = max_hold_bars

    for ss in range(n_samples):
        pitches = []
        durations = []
        if is_seq_of_seq:
            voices = len(quantized[ss])
            qq = quantized[ss]
        else:
            voices = quantized[ss].shape[1]
            qq = quantized[ss].T

        for i in range(voices):
            q = qq[i]
            pitch_i = [0]
            dur_i = []
            cur = 0
            count = 0
            for qi in q:
                if qi != cur or count > max_hold:
                    pitch_i.append(qi)
                    quarter_count = quantized_bin_size * (count + 1)
                    dur_i.append(quarter_count)
                    cur = qi
                    count = 0
                else:
                    count += 1
            quarter_count = quantized_bin_size * (count + 1)
            dur_i.append(quarter_count)
            pitches.append(pitch_i)
            durations.append(dur_i)
        all_pitches.append(pitches)
        all_durations.append(durations)
    return all_pitches, all_durations


def quantized_to_pretty_midi(quantized,
                             quantized_bin_size,
                             save_dir="samples",
                             name_tag="sample_{}.mid",
                             hold_symbol=False,
                             max_hold_bars=1,
                             add_to_name=0,
                             lower_pitch_limit=12,
                             default_quarter_length=120,
                             default_resolution=220,
                             list_of_quarter_length=None,
                             voice_params="woodwinds"):
    """
    takes in list of list of list, or list of array with axis 0 time, axis 1 voice_number (S,A,T,B)
    outer list is over samples, middle list is over voice, inner list is over time
    """
    all_pitches, all_durations = quantized_to_pitch_duration(quantized, quantized_bin_size, hold_symbol, max_hold_bars=max_hold_bars)
    pitches_and_durations_to_pretty_midi(all_pitches, all_durations,
                                         save_dir=save_dir,
                                         name_tag=name_tag,
                                         add_to_name=add_to_name,
                                         lower_pitch_limit=lower_pitch_limit,
                                         list_of_quarter_length=list_of_quarter_length,
                                         default_quarter_length=default_quarter_length,
                                         default_resolution=default_resolution,
                                         voice_params=voice_params)


# rough guide https://www.python-course.eu/python_scores.php
def plot_lilypond(upper_voices, lower_voices=None, own_staves=False,
                  key_signatures=None,
                  time_signatures=None,
                  chord_annotations=None,
                  interval_figures=None,
                  interval_durations=None,
                  use_clefs=None,
                  make_pdf=False,
                  fpath="tmp.ly",
                  title="Tmp", composer="Tmperstein", tagline="Copyright:?",
                  show=False,
                  x_zoom_bounds=(90, 780), y_zoom_bounds=(50, 220)):
    """
    Expects upper_voices and lower_voices to be list of list

    Needs lilypond, and pdf2svg installed (sudo apt-get install pdf2svg)
    """
    if len(upper_voices) > 1:
        if lower_voices == None and own_staves==False:
            raise ValueError("Multiple voices in upper staff with own_staves=False")
        if interval_durations is not None and len(interval_durations) > (len(upper_voices) - 1):
            print("WARNING: Truncating multi-part interval information to first {} sequences to match stave gaps, this is normal.".format(len(upper_voices) - 1))
            interval_durations = interval_durations[:len(upper_voices) - 1]
            interval_figures = interval_figures[:len(upper_voices) - 1]

    if use_clefs is None:
        use_clefs = ["treble" for i in range(len(upper_voices))]
    # need to align them for chord write T_T
    # for now assume 4/4
    pre = '\\version "2.12.3"'
    pre += roman_include
    minus_keys_flats = ["b", "e", "a", "d", "g", "c", "f"]
    minus_keys_names = ["\key f \major", "\key g \minor",  "\key c \minor",
                        "\key f \minor", "\key bes \minor", "\key ees \minor",
                        "\key aes \minor"]
    minus_keys_flats = minus_keys_flats[::-1]
    minus_keys_names = minus_keys_names[::-1]
    plus_keys_sharps = ["f", "c", "g", "d", "a", "e", "b"]
    plus_keys_names = ["\key g \major", "\key d \major",  "\key a \major",
                       "\key e \major", "\key b \major", "\key fis \major",
                       "\key cis \major"]
    trange = len(upper_voices)
    if lower_voices is not None:
        trange += len(lower_voices)
    if key_signatures is None:
        key_signatures = [[0] for i in range(trange)]
    if time_signatures is None:
        time_signatures = [(4, 1) for i in range(trange)]
    assert len(key_signatures) == trange
    assert len(time_signatures) == trange
    chord_str_pre = """\nanalysis = \lyricmode {
    % \set stanza  = #"G:"
  % For bare Roman numerals, \\rN simply outputs the string."""
    chord_str_post = "\n}\n"
    # this is fake test data
    '''
    chords = """
      \markup \\rN { I }
      I
      \markup \\rN { V 6 5 }
      \markup \\rN { vii o 4 3 / IV }
      \markup \\rN { IV 6 }
      \markup \\rN { ii h 4 3 }
      \markup \\rN { Fr +6 }
      \markup \\rN { I 6 4 }
      \markup \\rN { vii o 7 / vi }
      vi
    """
    '''
    # parse chord annotations
    if chord_annotations is None:
        chord_str = chord_str_pre + chord_str_post
    else:
        chords = ""
        lily_chords = map_music21_romans_to_lilypond(chord_annotations)
        chord_template_pre = "\markup \\rN { "
        chord_template_post = " }\n"
        for n, lc in enumerate(lily_chords):
            if len(lc.strip()) == 1:
                chord_template = lc + "\n"
            else:
                chord_template = chord_template_pre + lc + chord_template_post
            chords += chord_template
            # need to double the first element due to rendering issues in lilypond
            if n == 0:
                chords += chord_template
        chord_str = chord_str_pre + chords + chord_str_post
        chord_str += ""

    pre += chord_str

    if own_staves == False:
        raise ValueError("FIX")
        upper_staff = ""
        lower_staff = ""

        for n, uv in enumerate(upper_voices):
            ksi = key_signatures[n][0]
            tsi = time_signatures[n]
            if ksi != 0:
                if ksi < 0:
                    key_name = minus_keys_names[ksi]
                else:
                    assert ksi - 1 >= 0
                    key_name = plus_keys_names[ksi - 1]
                upper_staff += key_name + " "
                upper_staff += "\\time {}/{}".format(tsi[0], tsi[1]) + " "
            for u in uv:
                upper_staff += u + " "

        if lower_voices is not None:
            for n, lv in lower_voices:
                n_offset = n + len(upper_voices)
                ksi = key_signatures[n_offset][0]
                tsi = time_signatures[n_offset]
                if ksi != 0:
                    if ksi < 0:
                        key_name = minus_keys_names[ksi]
                    else:
                        assert ksi - 1 >= 0
                        key_name = plus_keys_names[ksi - 1]
                    lower_staff += key_name + " "
                    lower_staff += "\\time {}/{}".format(tsi[0], tsi[1]) + " "
                for l in lv:
                    lower_staff += l + " "

        staff = "{\n\\new PianoStaff << \n"
        staff += "  \\new Staff {" + upper_staff + "}\n"
        if lower_staff != "":
            staff += "  \\new Staff { \clef bass " + lower_staff + "}\n"
        staff += ">>\n}\n"
        raise ValueError("upper/lower voice not handled yet")
    else:
        if lower_voices is not None:
            raise ValueError("Put all voices into list of list upper_voices!")
        staff = "{\n\\new StaffGroup << \n"
        for n, v in enumerate(upper_voices):
            this_staff = ""
            ksi = key_signatures[n][0]
            tsi = time_signatures[n]
            if ksi != 0:
                if ksi < 0:
                    key_name = minus_keys_names[ksi]
                else:
                    assert ksi - 1 >= 0
                    key_name = plus_keys_names[ksi - 1]
                this_staff += key_name + " "
                this_staff += "\\time {}/{}".format(tsi[0], tsi[1]) + " "
            for vi in v:
                this_staff += vi + " "

            this_voice = "{}".format("voice{}".format(n))
            staff += '  \\new Voice = "{}"'.format(this_voice) + " {" + '\clef "' + use_clefs[n] + '" ' + this_staff + "}\n"
            if interval_figures is not None and len(interval_figures) > n:
                assert interval_durations is not None
                this_intervals = interval_figures[n]
                this_durations = interval_durations[n]
                # need to fix this
                duration_map = {"4": "1",
                                "2": "2",
                                "1": "4"}
                intervals_str = ""
                for di, ti in zip(this_durations, this_intervals):
                    intervals_str += "<" + str(ti) + ">{} ".format(duration_map[di])
                intervals_str = intervals_str.strip()
                staff += "  \\new FiguredBass \\figuremode { " + intervals_str + " }\n"
            # only the bottom staff...
            if n == trange - 1:
                staff += '  \\new Lyrics \\lyricsto "{}"'.format(this_voice) + " { \\analysis }\n"
        staff += ">>\n}\n"
    title = """\header {{
title = "{}"
composer = "{}"
tagline = "{}"
}}""".format(title, composer, tagline)

    final_ly = pre + staff + title
    with open(fpath, "w") as f:
        f.write(final_ly)

    # also make the pdf?
    if make_pdf:
        pe("lilypond {}".format(fpath))
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        pe("lilypond -fpng {}".format(fpath))

        if len(fpath.split(os.sep)) == 1:
            flist = os.listdir(os.getcwd())
        else:
            flist = os.listdir(str(os.sep).join(fpath.split(os.sep)[:-1]))

        valid_files_name = ".".join(fpath.split(os.sep)[-1].split(".")[:-1])
        flist = [fl for fl in flist if valid_files_name in fl]
        # hardcode to only show 1 page for now...
        flist = [fl for fl in flist if "page1" in fl or "page" not in fl]
        latest_file = max(flist, key=os.path.getctime)
        img = mpimg.imread(latest_file)
        f = plt.figure()
        ax = plt.gca()
        if None in x_zoom_bounds:
            if x_zoom_bounds[-1] is not None:
                raise ValueError("None for x_zoom_bounds only supported on last entry")
            x_zoom_bounds = (x_zoom_bounds[0], img.shape[1])

        if None in y_zoom_bounds:
            if y_zoom_bounds[-1] is not None:
                raise ValueError("None for y_zoom_bounds only supported on last entry")
            y_zoom_bounds = (y_zoom_bounds[0], img.shape[0])
        ax.set_xlim(x_zoom_bounds[0], x_zoom_bounds[1])
        ax.set_ylim(y_zoom_bounds[1], y_zoom_bounds[0])
        ax.imshow(img)
        plt.show()


def map_midi_pitches_to_lilypond(pitches, key_signatures=None):
    # takes in list of list
    # 0 = rest
    # 12 = C0
    # 24 = C1
    # 36 = C2
    # 48 = C3
    # 60 = C4
    # 72 = C5
    # 84 = C6

    # accidentals are key dependent! oy vey
    sharp_notes = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
    flat_notes =  ["c", "des", "d", "ees", "e", "f", "ges", "g", "aes", "a", "bes", "ces"]
    octave_map = [",,,", ",,", ",", "", "'", "''", "'''"]
    minus_keys_flats = ["b", "e", "a", "d", "g", "c", "f"]
    minus_keys_flats = minus_keys_flats[::-1]
    plus_keys_sharps = ["f", "c", "g", "d", "a", "e", "b"]
    rest = "r"
    lily_str_lists = []
    if key_signatures is None:
        key_signatures = [[0] for i in range(len(pitches))]
    use_voice_notes = [sharp_notes if key_signatures[i][0] >= 0 else flat_notes
                       for i in range(len(pitches))]
    assert len(key_signatures) == len(pitches)
    for n, (ks, pv) in enumerate(zip(key_signatures, pitches)):
        use_notes = use_voice_notes[n]
        note_str = [use_notes[int(pvi % 12)] if pvi != 0 else rest for pvi in pv]
        octave_str = [octave_map[int(pvi // 12)] if pvi != 0 else "" for pvi in pv]
        str_list = [ns + os for ns, os in zip(note_str, octave_str)]
        lily_str_lists.append(str_list)
    return lily_str_lists


def map_music21_romans_to_lilypond(chord_annotations):
    #ordered long to short so first match is "best"
    #major_functions = ["I", "II", "III", "IV", "V", "VI", "VII"]
    major_functions = ["VII", "III", "VI", "IV", "II", "V", "I"]
    minor_functions = [mf.lower() for mf in major_functions]
    lilypond_chords = []
    for ca in chord_annotations:
        fca = None
        ext = None
        pre = ""
        if "#" == ca[0]:
            ca = ca[1:]
            pre += ca[0]

        # need to parse it :|
        # try major first, then minor
        if len(ca) > 1:
            for n in range(1, len(ca)):
               for maf, mif in zip(major_functions, minor_functions):
                   if ca[:n] in maf:
                       fca = "".join(ca[:n])
                       ext = " ".join([cai for cai in ca[n:]])
                       pre += ""
                       break
                   elif ca[:n] in mif:
                       fca = "".join(ca[:n])
                       ext = " ".join([cai for cai in ca[n:]])
                       pre += ""
                       break
        elif len(ca) == 1:
            # len == 1
            fca = ca
            ext = ""
            pre += ""
        else:
            print("empty chord annotation!")
            from IPython import embed; embed(); raise ValueError()

        # still no matches!
        if fca is None:
            print("wrong...")
            from IPython import embed; embed(); raise ValueError()

        if fca in major_functions:
            matches = [major_functions[n] for n, ma in enumerate(major_functions) if fca == ma]
        elif fca in minor_functions:
            matches = [minor_functions[n] for n, mi in enumerate(minor_functions) if fca == mi]
        else:
            print("???")
            from IPython import embed; embed(); raise ValueError()
        lily_chord_function = matches[0] + " " + ext
        lilypond_chords.append(lily_chord_function)
    return lilypond_chords


def map_midi_durations_to_lilypond(durations, extras=None):
    # assumed to be relative lengths from quarter note?
    # do I need to make Fraction objects?
    # default is quarter note
    def ff(f):
        return fractions.Fraction(f)

    duration_map = {ff(8.): "\\breve",
                    ff(6.): "1.",
                    ff(4.): "1",
                    ff(3.): "2.",
                    ff(2.): "2",
                    ff(1.5): "4.",
                    ff(1.25): "4~{}16",
                    ff(1.): "4",
                    ff(.75): "8.",
                    ff(.5): "8",
                    ff(.25): "16",
                    ff(.125): "32",
                    ff(.0625): "64"}

    if extras is None:
        extras = []
        for du in durations:
            e = []
            for diu in du:
                e.append(0)
            extras.append(e)

    lily_str_lists = []
    assert len(durations) == len(extras)
    for dv, ev in zip(durations, extras):
        str_list = []
        assert len(dv) == len(ev)
        for dvi, evi in zip(dv, ev):
            try:
                frac_dvi = duration_map[ff(dvi)]
                if evi != 0:
                   if evi == 1 or evi == 2:
                       frac_dvi += "~"
                str_list.append(frac_dvi)
            except KeyError:
                raise KeyError("No known mapping for duration {}".format(dvi))
        lily_str_lists.append(str_list)
    return lily_str_lists


def pitches_and_durations_to_lilypond_notation(pitches, durations, extras=None,
                                               key_signatures=None):
    lilypitches = map_midi_pitches_to_lilypond(pitches, key_signatures=key_signatures)
    lilydurs = map_midi_durations_to_lilypond(durations, extras)
    assert len(lilypitches) == len(lilydurs)
    lilycomb = []
    for lp, ld in zip(lilypitches, lilydurs):
        assert len(lp) == len(ld)
        lc = []
        for lpi, ldi in zip(lp, ld):
            if "~" in ldi:
                from IPython import embed; embed(); raise ValueError()
                lc.append(lpi + ldi.format(lpi))
            else:
                lc.append(lpi + ldi)
        lilycomb.append(lc)
    return lilycomb


def plot_pitches_and_durations(pitches, durations,
                               save_dir="plots",
                               name_tag="plot_{}.ly",
                               make_pdf=False,
                               extras=None,
                               time_signatures=None,
                               key_signatures=None,
                               chord_annotations=None,
                               interval_figures=None,
                               interval_durations=None,
                               use_clefs=None):
    # list of list of list inputs
    assert len(pitches) == len(durations)
    try:
        pitches[0][0][0]
        durations[0][0][0]
    except:
        raise ValueError("pitches and durations should be list of list of list")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for n, (pitches, durations) in enumerate(zip(pitches, durations)):
        """
        try:
        """
        # map midi pitches to lilypond ones... oy
        voices = pitches_and_durations_to_lilypond_notation(pitches, durations, extras, key_signatures=key_signatures)
        #plot_lilypond([voices[1]])
        #plot_lilypond([voices[0]], [voices[-1]])
        #plot_lilypond([voices[0]], [voices[-1]], own_staves=True)
        # TODO: fix own_staves=False issues with conflicting time/key signatures
        # raise an error
        # down the line, fix accidentals on case by case basis :|
        # add options for chord notations, and intervals for final analysis
        # add grey notes (all possibles) as well to visualize the decoding?
        this_dir = os.getcwd()
        #full_fpath = save_dir + os.sep + name_tag.format(n)
        local_fpath = name_tag.format(n)
        os.chdir(save_dir)
        # do it this way because lilypond uses the local dir by default...
        plot_lilypond(voices, own_staves=True,
                      fpath=local_fpath,
                      make_pdf=make_pdf,
                      time_signatures=time_signatures,
                      key_signatures=key_signatures,
                      chord_annotations=chord_annotations,
                      interval_figures=interval_figures,
                      interval_durations=interval_durations,
                      use_clefs=use_clefs)
        os.chdir(this_dir)
        """
        except:
            print("Error writing index {}, continuing...".format(n))
        """

def music21_to_chord_duration(p, key):
    """
    Takes in a Music21 score, and outputs three lists
    List for chords (by primeFormString string name)
    List for chord function (by romanNumeralFromChord .romanNumeral)
    List for durations
    """
    p_chords = p.chordify()
    p_chords_o = p_chords.flat.getElementsByClass('Chord')
    chord_list = []
    chord_function_list = []
    duration_list = []
    for ch in p_chords_o:
        duration_list.append(ch.duration.quarterLength)
        ch.closedPosition(forceOctave=4, inPlace=True)
        rn = roman.romanNumeralFromChord(ch, key)
        rp = rn.pitches
        rp_names = ",".join([pi.name + pi.unicodeNameWithOctave[-1] for pi in rp])
        chord_list.append(rp_names)
        chord_function_list.append(rn.figure)
    return chord_list, chord_function_list, duration_list


def music21_to_pitch_duration(p, verbose=False):
    """
    Takes in a Music21 score, and outputs 4 list of list
    One for pitch
    One for duration
    list for part times of each voice
    list of list of fermatas
    """
    parts = []
    parts_times = []
    parts_delta_times = []
    parts_fermatas = []
    for i, pi in enumerate(p.parts):
        part = []
        part_time = []
        part_delta_time = []
        part_fermatas = []
        total_time = 0
        all_chord = True
        for n in pi.stream().flat.notesAndRests:
            has_fermata = any([ne.isClassOrSubclass(('Fermata',)) for ne in n.expressions])
            if has_fermata:
                part_fermatas.append(1)
            else:
                part_fermatas.append(0)

            if n.isRest:
                part.append(0)
                all_chord = False
            else:
                if not n.isChord:
                    part.append(n.pitch.midi)
                    all_chord = False
            part_time.append(total_time + n.duration.quarterLength)
            total_time = part_time[-1]
            part_delta_time.append(n.duration.quarterLength)
        if all_chord:
            if verbose:
                print("Found a part with only chords, skipping...")
        else:
            parts.append(part)
            parts_times.append(part_time)
            parts_delta_times.append(part_delta_time)
            parts_fermatas.append(part_fermatas)
    return parts, parts_times, parts_delta_times, parts_fermatas


def music21_to_quantized(p, quantized_bin_size=0.125):
    """
    Convert from music21 score to quantized
    """
    r = music21_to_pitch_duration(p)
    parts, _, parts_durations = r
    pr = pitch_and_duration_to_quantized(parts, parts_durations, quantized_bin_size)
    return pr


def ribbons_from_piano_roll(piano_roll, ribbon_type, quantized_bin_size,
                            interval=12):
    from IPython import embed; embed(); raise ValueErro


def quantized_imlike_to_image_array(piano_roll_as_imarray, quantized_bin_size,
                                    plot_colors="default",
                                    background="white"):
    """
    piano_roll_as_imarray should be N H W C
    """
    import matplotlib as mpl
    from matplotlib.colors import colorConverter
    n_voices = piano_roll_as_imarray.shape[-1]
    if plot_colors == "default":
        if background == "black":
            # style dark_background from matplotlib
            # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle 
            colors = ['#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2', '#fdb462', '#b3de69', '#bc82bd', '#ccebc4', '#ffed6f']
            #colors = ["darkred", "steelblue", "forestgreen", "orchid"]
        elif background == "white":
            # https://www.sessions.edu/color-calculator-results/?colors=dea85b,5c4db0,b34aa3,8c963d
            #colors = ['#5C4DB0', '#DEA85B', '#B34AA3', '#8C963D']
            # https://www.sessions.edu/color-calculator-results/?colors=3b3a9c,cf9022,83a82c,802380
            colors = ['#3B3A9C', '#CF9022', '#83A82C', '#802380']
        elif background == "gray":
            # seaborn pastel style
            colors = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6']
            #colors = ["darkred", "steelblue", "forestgreen", "orchid"]
        else:
            raise ValueError("No defaults for background color {}".format(background))
    else:
        colors = plot_colors

    if n_voices > len(colors):
        raise ValueError("Need to provide at least as many colors as voices! {} colors but piano_roll_as_imarray.shape[-1] is {}".format(len(colors), piano_roll_as_imarray.shape[-1]))
    # zip will truncate!
    cmaps = [mpl.colors.LinearSegmentedColormap.from_list("my_cmap_{}".format(v), [background, c], 256)
             for c, v in zip(colors, list(range(n_voices)))]

    # one cmap per channel
    for cmap in cmaps:
        cmap._init()
        # lazy way to make zeros of right size
        alphas = np.linspace(0., 1., cmap.N + 3)
        cmap._lut[:, -1] = alphas
    # NO ALPHA SUPPORT
    joined = np.array([cmap(piano_roll_as_imarray[..., i])[..., :-1] for i, cmap in enumerate(cmaps)])
    as_imarray = joined.sum(axis=0) / float(n_voices)
    return as_imarray


def plot_piano_roll(piano_roll, quantized_bin_size,
                    pitch_bot=1, pitch_top=88,
                    colors=["red", "blue", "green", "purple"],
                    return_img_like=True,
                    ribbons=False,
                    ribbon_type="std",
                    axis_handle=None,
                    autorange=True,
                    autoscale_ratio=0.25,
                    show=False,
                    show_rest_channel=False):
    # piano roll should be time x voices
    # https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import colorConverter
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib as mpl

    if axis_handle is None:
        fig = Figure()
        ax = plt.gca()

    time_len = len(piano_roll)
    n_voices = len(piano_roll[0])
    n_colors = len(colors)
    if n_voices > n_colors:
        raise ValueError("Need as many colors as voices! Only gave {} colors for {} voices".format(n_colors, n_voices))
    if ribbons:
        ribbons_traces = ribbons_from_piano_roll(piano_roll, ribbon_type,
                                                 quantized_bin_size)
        from IPython import embed; embed(); raise ValueError()
    else:
       ribbons_traces = None
    # 0 always rest!
    voice_storage = np.zeros((time_len, pitch_top, n_voices))

    for v in range(n_voices):
        pitch_offset_values = [piano_roll[i][v] for i in range(time_len)]

        for n, pov in enumerate(pitch_offset_values):
            voice_storage[n, pov, v] = 255.

    cmaps = [mpl.colors.LinearSegmentedColormap.from_list("my_cmap_{}".format(v), ["white", c], 256)
             for c, v in zip(colors, list(range(n_voices)))]

    for cmap in cmaps:
        cmap._init()
        # lazy way to make zeros of right size
        alphas = np.linspace(0., 0.6, cmap.N + 3)
        cmap._lut[:, -1] = alphas

    nz = np.where(voice_storage != 0.)[1]
    nz = nz[nz >= pitch_bot]
    nz = nz[nz <= pitch_top]
    if autorange:
        mn = nz.min() - 6
        mx = nz.max() + 1 + 6
    else:
        if show_rest_channel:
            mn = 0
        else:
            mn = pitch_bot
        mx = pitch_top + 1

    for v in range(n_voices):
        ax.imshow(voice_storage[:, :, v].T, cmap=cmaps[v], interpolation=None)

    ax.set_ylim([mn, mx])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pitch")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(mx - mn, autoscale_ratio * len(voice_storage))
    ax.set_aspect(asp)
    if not show and return_img_like:
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        ax.axis("off")
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi() 
        width = int(width)
        height = int(height)
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        return img

    patch_list = [mpatches.Patch(color=c, alpha=.6, label="Voice {}".format(v)) for c, v in zip(colors, list(range(n_voices)))]
    ax.legend(handles=patch_list, bbox_to_anchor=(0., -.3, 1., .102), loc=1,
              ncol=1, borderaxespad=0.)
    if show:
        plt.show()
    return voice_storage, ribbons_traces



if __name__ == "__main__":
    # Won't work, but serves as a good example of usage
    """
    # fig 5, gradus ad parnassum
    notes = [["A3", "A3", "G3", "A3", "B3", "C4", "C4", "B3", "D4", "C#4", "D4"],
             ["D3", "F3", "E3", "D3", "G3", "F3", "A3", "G3", "F3", "E3", "D3"]]
    durations = [[4.] * len(notes[0]), [4.] * len(notes[1])]
    # can add harmonic nnotations as well to plot
    #chord_annotations = ["i", "I6", "IV", "V6", "I", "IV6", "I64", "V", "I"]
    """
    from analysis import notes_to_midi
    from analysis import fixup_parts_durations
    from analysis import intervals_from_midi
    from datasets import fetch_two_voice_species3

    ex = fetch_two_voice_species3()
    nd = ex[-2]["notes_and_durations"]
    notes = [[ndii[0] for ndii in ndi] for ndi in nd]
    durations = [[ndii[1] for ndii in ndi] for ndi in nd]

    #notes = ex[-2]["notes"]
    #durations = ex[-2]["durations"]
    # can we do all these automatically?
    parts = notes_to_midi(notes)
    interval_figures = intervals_from_midi(parts, durations)
    _, interval_durations = fixup_parts_durations(parts, durations)
    interval_durations = [interval_durations[0]]
    # need to figure out duration convention (maybe support floats and str both?)
    durations = [[int(di) for di in d] for d in durations]

    # treble, bass, treble_8, etc
    clefs = ["treble", "treble_8"]
    time_signatures = [(4, 4), (4, 4)]
    pitches_and_durations_to_pretty_midi([parts], [durations],
                                         save_dir="samples",
                                         name_tag="sample_{}.mid",
                                         default_quarter_length=240,
                                         voice_params="piano")

    # figure out plotting of tied notes
    # fix zoom
    plot_pitches_and_durations([parts], [durations],
                               interval_figures=interval_figures,
                               interval_durations=interval_durations,
                               use_clefs=clefs)

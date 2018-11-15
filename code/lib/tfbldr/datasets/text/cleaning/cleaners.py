""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from .eng_rules import rulebased_g2p


_whitespace_re = re.compile(r'\s+')
_apos_s_re = re.compile(r"'s")
_single_re = re.compile(r'["]')
_double_re = re.compile(r"[']")
_semicolon_re = re.compile(r';')
_paren_re = re.compile(r'[()]')
_bracket_re = re.compile(r'[\[\]]')
_dash_re = re.compile(r'--')
_comma_re = re.compile(r' , ')
_colon_re = re.compile(r':')
_period_re = re.compile(r'\.$')
_abbrev_re = re.compile(r'\.')
_US_re = re.compile(r' US')
_UK_re = re.compile(r' UK')
_FBI_re = re.compile(r' FBI')
_CIA_re = re.compile(r' CIA')
_NSA_re = re.compile(r' NSA')
_USA_re = re.compile(r' USA')
_USSR_re = re.compile(r' USSR')

# handle 22 -> 22nd???

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  text = re.sub(_USSR_re, ' U S S R', text)
  text = re.sub(_USA_re, ' U S A', text)
  text = re.sub(_US_re, ' U S', text)
  text = re.sub(_UK_re, ' U K', text)
  text = re.sub(_FBI_re, ' F B I', text)
  text = re.sub(_CIA_re, ' C I A', text)
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  unicode_content = text.decode('utf-8')
  return unidecode(unicode_content)


def collapse_spurious(text):
  text = re.sub(_apos_s_re, "-s", text)
  text = re.sub(_single_re, "", text)
  text = re.sub(_double_re, "", text)
  text = re.sub(_paren_re, "", text)
  text = re.sub(_semicolon_re, ",", text)
  text = re.sub(_dash_re, ",", text)
  text = re.sub(_colon_re, ", ", text)
  text = re.sub(_period_re, "", text)
  text = re.sub(_bracket_re, "", text)
  text = re.sub(_abbrev_re, " ", text)
  text = re.sub(_comma_re, ", ", text)
  text = re.sub(_comma_re, ", ", text)
  return text


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def rulebased_g2p_cleaners(text):
  text = convert_to_ascii(text)
  r = rulebased_g2p(text)
  text = "^".join(["&".join(ri[1]).lower() for ri in r])
  text = lowercase(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_spurious(text)
  text = collapse_whitespace(text)
  return text


def english_minimal_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = collapse_whitespace(text)
  return text


def english_phone_cleaners(text):
  '''Pipeline for English phones.'''
  return text

def english_phone_pause_cleaners(text):
  '''Pipeline for English phones.'''
  return text

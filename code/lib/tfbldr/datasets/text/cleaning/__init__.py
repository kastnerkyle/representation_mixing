""" from https://github.com/keithito/tacotron """
import re
import cleaners
from symbols import char_symbols
from symbols import phone_symbols
from symbols import pau_phone_symbols
from eng_rules import hybrid_g2p, rulebased_g2p


# Mappings from symbol to numeric ID and vice versa:
_char_symbol_to_id = {s: i for i, s in enumerate(char_symbols)}
_id_to_char_symbol = {i: s for i, s in enumerate(char_symbols)}

_phone_symbol_to_id = {s: i for i, s in enumerate(phone_symbols)}
_id_to_phone_symbol = {i: s for i, s in enumerate(phone_symbols)}

_pau_phone_symbol_to_id = {s: i for i, s in enumerate(pau_phone_symbols)}
_id_to_pau_phone_symbol = {i: s for i, s in enumerate(pau_phone_symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def get_vocabulary_sizes(cleaner_names):
  """
    if pause in name, return pause phone size
    if phone in name, return phone size
    else return char size
  """
  outs = []
  for cn in cleaner_names:
      if "pause" in cn:
          outs.append(len(_pau_phone_symbol_to_id))
      elif "phone" in cn:
          outs.append(len(_phone_symbol_to_id))
      else:
          outs.append(len(_char_symbol_to_id))
  # needed?
  if len(outs) == 1:
      outs = outs[0]
  return outs


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if any(["rule" in name for name in cleaner_names]):
      raise ValueError("IMPLEMENT RULE TRANFORM")
      sequence = []
      # Check for curly braces and treat their contents as ARPAbet:
      while len(text):
        m = _curly_re.match(text)
        if not m:
          sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
          break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

      # Append EOS token
      sequence.append(_symbol_to_id['~'])
      return sequence
  elif any(["pause" in name for name in cleaner_names]):
      sequence = []
      # Check for curly braces and treat their contents as ARPAbet:
      while len(text):
        m = _curly_re.match(text)
        if not m:
          sequence += _pau_phone_symbols_to_sequence(_clean_text(text, cleaner_names))
          break
        sequence += _pau_phone_symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
      # Append EOS token
      sequence.append(_phone_symbol_to_id['~'])
      return sequence
  elif any(["phone" in name for name in cleaner_names]):
      sequence = []
      # Check for curly braces and treat their contents as ARPAbet:
      while len(text):
        m = _curly_re.match(text)
        if not m:
          sequence += _phone_symbols_to_sequence(_clean_text(text, cleaner_names))
          break
        sequence += _phone_symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
      # Append EOS token
      sequence.append(_phone_symbol_to_id['~'])
      return sequence
  else:
      sequence = []
      # Check for curly braces and treat their contents as ARPAbet:
      while len(text):
        m = _curly_re.match(text)
        if not m:
          sequence += _char_symbols_to_sequence(_clean_text(text, cleaner_names))
          break
        sequence += _char_symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

      # Append EOS token
      sequence.append(_char_symbol_to_id['~'])
      return sequence


def sequence_to_text(sequence, cleaner_names):
  '''Converts a sequence of IDs back to a string'''
  if any(["rule" in name for name in cleaner_names]):
      raise ValueError("IMPLEMENT RULE TRANFORM")
  elif any(["pause" in name for name in cleaner_names]):
      result = ""
      space_id = _pau_phone_symbol_to_id[" "]
      pad_id = _pau_phone_symbol_to_id["_"]
      eos_id = _pau_phone_symbol_to_id["~"]
      special_ids = [_pau_phone_symbol_to_id[special] for special in ["1","2","3","4"]]
      for symbol_id in sequence:
          if symbol_id in [space_id, pad_id, eos_id]:
              result += _id_to_pau_phone_symbol[symbol_id]
          elif symbol_id in special_ids:
              result += _id_to_pau_phone_symbol[symbol_id]
          else:
              result += "@" + _id_to_pau_phone_symbol[symbol_id]
      return result
  elif any(["phone" in name for name in cleaner_names]):
      result = ""
      space_id = _phone_symbol_to_id[" "]
      pad_id = _phone_symbol_to_id["_"]
      eos_id = _phone_symbol_to_id["~"]
      special_ids = [_phone_symbol_to_id[special] for special in "!,:?"]
      for symbol_id in sequence:
          if symbol_id in [space_id, pad_id, eos_id] + special_ids:
              result += _id_to_phone_symbol[symbol_id]
          else:
              result += "@" + _id_to_phone_symbol[symbol_id]
      return result
  else:
      result = ''
      for symbol_id in sequence:
        if symbol_id in _id_to_char_symbol:
          s = _id_to_char_symbol[symbol_id]
          # Enclose ARPAbet back in curly braces:
          if len(s) > 1 and s[0] == '@':
            s = '{%s}' % s[1:]
          result += s
      return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _char_symbols_to_sequence(symbols):
  return [_char_symbol_to_id[s] for s in symbols if _char_should_keep_symbol(s)]

def _pau_phone_symbols_to_sequence(symbols):
  new = []
  specials = ["1", "2", "3", "4"]
  for ss in symbols.split(" "):
      if any([special in ss for special in specials]):
          all_special = [special for special in ss if special in specials]
          all_non_special = [nonspecial[1:] for nonspecial in ss if nonspecial not in specials]
          prev = []
          for ssi in ss.strip().split("@")[1:]:
              if any([special in ssi for special in specials]):
                  prev.append(re.sub("|".join(specials), "", ssi))
                  which_specials = [special for special in specials if special in ssi]
                  for p in prev:
                      new.append(p)
                  # ASSUME ONLY 1?
                  new.append(which_specials[0])
                  prev = []
              else:
                  prev.append(ssi)
      else:
          for ssi in ss.strip().split("@")[1:] + [" "]:
              new.append(ssi)
  return [_pau_phone_symbol_to_id[s] for s in new if _pau_phone_should_keep_symbol(s)]

def _phone_symbols_to_sequence(symbols):
  new = []
  for ss in symbols.split(" "):
      if any([special in ss for special in "!,:?"]):
          # special symbols only at start or back of chunk
          if ss[0] in "!,:?":
              for ssi in [ss[0]] + ss[1:].strip().split("@")[1:] + [" "]:
                  new.append(ssi)
          elif ss[-1] in "!,:?":
              for ssi in ss[:-1].strip().split("@")[1:] + [ss[-1]] + [" "]:
                  new.append(ssi)
      else:
          for ssi in ss.strip().split("@")[1:] + [" "]:
              new.append(ssi)
  #new = [ssi for ss in symbols.split(" ") for ssi in ss.strip().split("@")[1:] + [" "]][:-1]
  return [_phone_symbol_to_id[s] for s in new if _phone_should_keep_symbol(s)]

def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])

def _char_should_keep_symbol(s):
  return s in _char_symbol_to_id and s is not '_' and s is not '~'

def _pau_phone_should_keep_symbol(s):
  return s in _pau_phone_symbol_to_id and s is not '_' and s is not '~'

def _phone_should_keep_symbol(s):
  return s in _phone_symbol_to_id and s is not '_' and s is not '~'

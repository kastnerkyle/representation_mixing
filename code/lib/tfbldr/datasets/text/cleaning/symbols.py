""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''

import cmudict

_pad = '_'
_eos = '~'
# PUT IT BACK!!!

_phones = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh', ' ']
special = [s for s in "!\',-.:?"]
_pau_phones = _phones + [s for s in ["1","2","3","4"]]
_phones = _phones + special

_characters = 'abcdefghijklmnopqrstuvwxyz!\',-.:? '
_rules = 'abcdefghijklmnopqrstuvwxyz&^!\',-.:? '

#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\',-.:? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
char_symbols = [_pad, _eos] + list(_characters)# + _arpabet
phone_symbols = [_pad, _eos] + list(_phones)# + _arpabet
pau_phone_symbols = [_pad, _eos] + list(_pau_phones)
rule_symbols = [_pad, _eos] + list(_rules)# + _arpabet
